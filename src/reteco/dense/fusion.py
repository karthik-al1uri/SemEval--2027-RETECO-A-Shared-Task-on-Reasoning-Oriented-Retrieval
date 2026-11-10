"""Weighted reciprocal rank fusion of a sparse and a dense candidate list.

    score(d) = w_sparse / (k + rank_sparse(d)) + w_dense / (k + rank_dense(d))

with rank starting at 1 and a missing document contributing 0. Weights are
normalised so that w_sparse + w_dense = 1. Both inputs are depth-1000 candidate
runs under cache/dense/candidates/<name>/<split>.trec:

  dense        written by reteco.dense.retrieve
  bm25         baseline BM25, produced here from the existing indexes in
               cache/bm25/ (only loaded, never built or written)
  sparse_best  the sparse agent's tuned BM25, produced by
               src/reteco/dense/sparse_candidates.py

The config key `sparse` names the sparse input (default bm25).

    python -m reteco.dense.fusion --config configs/dense_hybrid.yaml

With `tune: true` in the config, (k, w_dense) is chosen by macro nDCG@10 on the
train split over the config grid (ties broken by Recall@100, then by smaller
k and w_dense); the chosen setting is applied unchanged to dev.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import yaml

from ..data import DOMAINS, REPO_ROOT, SPLITS, load_queries
from ..evaluate import append_result, evaluate_run, format_table
from ..runs import _ordered, read_run, write_run
from . import CACHE_DIR, CANDIDATES_DIR

BM25_K1, BM25_B = 0.9, 0.4


def bm25_candidates(split: str, depth: int = 1000) -> Path:
    """Depth-`depth` BM25 run for all 13 domains, from the cached indexes (read-only)."""
    path = CANDIDATES_DIR / "bm25" / f"{split}.trec"
    if path.exists():
        return path
    from ..analysis import LuceneEnglishAnalyzer
    from ..bm25 import BM25Index
    analyzer = LuceneEnglishAnalyzer()
    run = {}
    for domain in DOMAINS:
        idx_dir = BM25Index.cache_path(domain, BM25_K1, BM25_B)
        if not (idx_dir / "meta.json").exists():
            raise FileNotFoundError(f"no cached BM25 index at {idx_dir}; refusing to rebuild it")
        index = BM25Index.load(idx_dir)
        for q in load_queries(domain, split):
            run[q.id] = index.search(analyzer.analyze(q.query), depth)
        print(f"bm25 candidates {split} {domain}", flush=True)
        del index
    return write_run(run, path, tag="bm25", depth=depth)


def ranks(run: dict[str, dict[str, float]]) -> dict[str, dict[str, int]]:
    """Ranks in trec_eval order (score desc, docid desc) – the order the run file lists."""
    return {q: {d: r for r, (d, _) in enumerate(_ordered(docs, len(docs)), 1)}
            for q, docs in run.items()}


def rrf(rank_lists: list[dict[str, dict[str, int]]], weights: list[float], k: float,
        depth: int = 100) -> dict[str, dict[str, float]]:
    total = sum(weights)
    if total <= 0 or any(w < 0 for w in weights):
        raise ValueError(f"weights must be non-negative with a positive sum, got {weights}")
    weights = [w / total for w in weights]
    out = {}
    qids = set().union(*rank_lists)
    for q in qids:
        scores: dict[str, float] = {}
        for rl, w in zip(rank_lists, weights):
            if w == 0:
                continue
            for d, r in rl.get(q, {}).items():
                scores[d] = scores.get(d, 0.0) + w / (k + r)
        top = sorted(scores.items(), key=lambda x: -x[1])[:depth]
        out[q] = dict(top)
    return out


def sparse_candidates(name: str, split: str) -> Path:
    if name == "bm25":
        return bm25_candidates(split)
    path = CANDIDATES_DIR / name / f"{split}.trec"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; see src/reteco/dense/sparse_candidates.py")
    return path


def load_ranks(dense: str, split: str, domains, sparse: str = "bm25") -> tuple[dict, dict]:
    keep = {q.id for d in domains for q in load_queries(d, split)}
    bm = read_run(sparse_candidates(sparse, split))
    de = read_run(CANDIDATES_DIR / dense / f"{split}.trec")
    return (ranks({q: v for q, v in bm.items() if q in keep}),
            ranks({q: v for q, v in de.items() if q in keep}))


def tune(cfg: dict, domains) -> tuple[dict, list[dict]]:
    bm, de = load_ranks(cfg["dense"], "train", domains, cfg.get("sparse", "bm25"))
    rows = []
    for k, w in itertools.product(cfg["grid"]["k"], cfg["grid"]["w_dense"]):
        res = evaluate_run(rrf([bm, de], [1 - w, w], k, cfg.get("depth", 100)), "train", domains)
        rows.append({"k": k, "w_dense": w, "ndcg@10": res["macro_ndcg@10"],
                     "recall@100": res["macro_recall@100"]})
    best = max(rows, key=lambda r: (round(r["ndcg@10"], 6), round(r["recall@100"], 6),
                                    -r["k"], -r["w_dense"]))
    return best, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.dense.fusion")
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--domains", nargs="+", default=None, choices=DOMAINS)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    domains = args.domains or cfg.get("domains", DOMAINS)
    if domains == "all":
        domains = DOMAINS
    name = cfg["system_name"]

    if cfg.get("tune"):
        best, rows = tune(cfg, domains)
        grid_path = CACHE_DIR / "fusion" / f"{name}_grid_train.csv"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        with open(grid_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        k, w_dense = best["k"], best["w_dense"]
        print(f"tuned on train: k={k} w_dense={w_dense} nDCG@10={best['ndcg@10']:.4f} "
              f"R@100={best['recall@100']:.4f}  (grid: {grid_path})")
    else:
        k, w_dense = cfg["k"], cfg["w_dense"]
        best = None

    desc = (f"{cfg['description']}; RRF k={k} w_dense={w_dense} w_sparse={round(1 - w_dense, 4)}"
            + (" (tuned on train)" if best else ""))
    for split in args.splits:
        bm, de = load_ranks(cfg["dense"], split, domains, cfg.get("sparse", "bm25"))
        fused = rrf([bm, de], [1 - w_dense, w_dense], k, cfg.get("depth", 100))
        path = write_run(fused, REPO_ROOT / "runs" / split / f"{name}.trec", tag=name,
                         depth=cfg.get("depth", 100))
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"description": desc, "config": args.config, "domains": list(domains),
             "sparse": cfg.get("sparse", "bm25"), "dense": cfg["dense"], "k": k,
             "w_dense": w_dense}, indent=2) + "\n")
        res = evaluate_run(fused, split, domains)
        print(f"\n{name} [{split}]\n{format_table(res)}")
        if not args.no_log:
            append_result(res, name, desc, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
