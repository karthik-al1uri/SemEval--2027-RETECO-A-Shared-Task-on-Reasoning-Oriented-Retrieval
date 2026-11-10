"""Dense retrieval: encode (if needed), search per domain, write runs, score and log.

    python -m reteco.dense.retrieve --config configs/dense_bge_small.yaml --splits train dev
    python -m reteco.dense.retrieve --config configs/dense_bge_small.yaml --encode-only

Writes runs/<split>/<system_name>.trec (top `depth`, default 100) plus a deeper
candidate run (top `candidate_depth`, default 1000) under
cache/dense/candidates/<system_name>/<split>.trec for rank fusion. Each split is
scored with reteco.evaluate and appended to results/experiments.csv unless
--no-log is given.

doc_mode "first": one vector per document (first `doc_max_tokens` tokens),
searched with a FAISS IndexFlatIP. doc_mode "maxp": one vector per passage,
document score = max over its passages (exact inner products, reduced per doc).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from ..data import DOMAINS, REPO_ROOT, SMALL_DOMAINS, SPLITS, load_queries
from ..evaluate import append_result, evaluate_run, format_table
from ..runs import write_run
from . import CANDIDATES_DIR
from .encode import encode_domain, encode_queries, is_encoded, load_domain, load_model


def search_first(emb: np.ndarray, doc_ids: list[str], q: np.ndarray, k: int) -> list[dict[str, float]]:
    import faiss
    # faiss-cpu and torch each bundle libomp on macOS; multi-threaded faiss
    # search segfaults once torch is loaded. Flat search is fast enough on one thread.
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    scores, idx = index.search(q, min(k, len(doc_ids)))
    return [{doc_ids[i]: float(s) for s, i in zip(srow, irow) if i >= 0}
            for srow, irow in zip(scores, idx)]


def search_maxp(emb: np.ndarray, doc_ids: list[str], doc_index: np.ndarray, q: np.ndarray,
                k: int) -> list[dict[str, float]]:
    # Passages of a document are contiguous, so a segmented max gives per-doc MaxP.
    starts = np.flatnonzero(np.r_[True, doc_index[1:] != doc_index[:-1]])
    owners = doc_index[starts]
    out = []
    for i in range(0, len(q), 64):
        doc_scores = np.maximum.reduceat(q[i:i + 64] @ emb.T, starts, axis=1)
        kk = min(k, doc_scores.shape[1])
        for row in doc_scores:
            top = np.argpartition(-row, kk - 1)[:kk]
            out.append({doc_ids[owners[j]]: float(row[j]) for j in top})
    return out


def run(cfg: dict, splits, domains, config_path: str = "", log_results: bool = True,
        encode_only: bool = False) -> dict:
    model = None
    deep = cfg.get("candidate_depth", 1000)
    runs = {s: {} for s in splits}
    for domain in domains:
        t0 = time.time()
        if not is_encoded(cfg, domain):
            if model is None:
                model = load_model(cfg["model"], cfg.get("dtype", "float16"))
            print(f"encoding {domain} ...", flush=True)
            encode_domain(cfg, domain, model)
        if encode_only:
            continue
        if model is None:
            model = load_model(cfg["model"], cfg.get("dtype", "float16"))
        emb, doc_ids, doc_index, meta = load_domain(cfg, domain)
        t1 = time.time()
        for split in splits:
            qs = load_queries(domain, split)
            q = encode_queries(model, [x.query for x in qs], cfg)
            if cfg["doc_mode"] == "maxp":
                res = search_maxp(emb, doc_ids, doc_index, q, deep)
            else:
                res = search_first(emb, doc_ids, q, deep)
            for x, r in zip(qs, res):
                runs[split][x.id] = r
        print(f"{domain:<10} {meta['n_docs']:>8} docs {meta['n_units']:>8} units  "
              f"load {t1 - t0:5.1f}s  search {time.time() - t1:5.1f}s", flush=True)
        del emb
    if encode_only:
        return {}

    results = {}
    name = cfg["system_name"]
    for split in splits:
        path = write_run(runs[split], REPO_ROOT / "runs" / split / f"{name}.trec", tag=name,
                         depth=cfg.get("depth", 100))
        write_run(runs[split], CANDIDATES_DIR / name / f"{split}.trec", tag=name, depth=deep)
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"description": cfg["description"], "config": config_path, "domains": list(domains)},
            indent=2) + "\n")
        # Score only the depth-100 list that the run file holds.
        trimmed = {qid: dict(sorted(d.items(), key=lambda x: -x[1])[:cfg.get("depth", 100)])
                   for qid, d in runs[split].items()}
        res = evaluate_run(trimmed, split, domains)
        print(f"\n{name} [{split}]\n{format_table(res)}")
        if log_results:
            append_result(res, name, cfg["description"], config_path)
        results[split] = res
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.dense.retrieve")
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--domains", nargs="+", default=None, choices=DOMAINS,
                    help="default: config `domains` (all | small | list), else all 13")
    ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    domains = args.domains or cfg.get("domains", "all")
    if domains == "all":
        domains = DOMAINS
    elif domains == "small":
        domains = SMALL_DOMAINS
    run(cfg, args.splits, domains, args.config, not args.no_log, args.encode_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
