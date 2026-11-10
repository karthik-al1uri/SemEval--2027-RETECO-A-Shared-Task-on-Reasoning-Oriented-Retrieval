"""Near-duplicate demotion: a rescorer for any TREC run (CPU only).

Within each query's ranked list, a document is a near-duplicate when it repeats a
HIGHER-ranked document of the same list (original order) under either test:

    strict    identical after lowercasing and dropping everything but [a-z0-9]
              (case, punctuation and whitespace ignored)
    jaccard   word 5-gram (shingle) Jaccard similarity >= threshold

Near-duplicates are demoted, never deleted: each one is moved below the next
`skip` non-duplicate documents that follow it (skip=1: below the next
non-duplicate). Order among the moved documents is kept. The output run gives
rank-based scores (depth - rank + 1), so the written order is the new ranking.

Tuning (`--tune`) picks the threshold by train macro nDCG@10 and writes the
config; applying it needs only query ids, the input run and the documents.

    # tune on train (writes configs/dedup.yaml)
    python -m reteco.analysis_tools.dedup --tune --run runs/train/hybrid_best.trec
    # apply to any run, e.g. later a reranker output
    python -m reteco.analysis_tools.dedup --run runs/dev/hybrid_best.trec --split dev \
        --config configs/dedup.yaml

Document features are recomputed on each call (about a minute for train + dev):
cache/ is shared read-only, so nothing is cached.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

from ..data import DOMAINS, REPO_ROOT, SPLITS, load_qrels
from ..evaluate import append_result, evaluate_run, format_table
from ..runs import read_run, write_run
from .errors import fetch_docs
from .perquery import macro, per_query

SHINGLE = 5
_NON = re.compile(r"[^a-z0-9]+")
_M = np.uint64(0x9E3779B97F4A7C15)   # odd 64-bit multiplier for rolling shingle hashes


# ---------------------------------------------------------------- features --
def strict_key(text: str) -> int:
    return hash(_NON.sub("", text.lower()))


def shingle_hashes(text: str, vocab: dict[str, int]) -> np.ndarray:
    """Sorted unique 64-bit hashes of word 5-grams (whole text if shorter than 5 words)."""
    words = _NON.sub(" ", text.lower()).split()
    if not words:
        return np.zeros(0, dtype=np.uint64)
    ids = np.fromiter((vocab.setdefault(w, len(vocab)) for w in words), dtype=np.uint64,
                      count=len(words))
    n = max(1, len(ids) - SHINGLE + 1)
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(min(SHINGLE, len(ids))):
            h = h * _M + ids[k:k + n] + np.uint64(1)
    return np.unique(h)


def jaccard_matrix(sh: list[np.ndarray]) -> np.ndarray:
    """Exact pairwise Jaccard of shingle sets (n x n, dense)."""
    n = len(sh)
    sizes = np.array([len(s) for s in sh], dtype=np.float64)
    if n == 0 or sizes.sum() == 0:
        return np.zeros((n, n))
    allh = np.concatenate(sh)
    _, cols = np.unique(allh, return_inverse=True)
    rows = np.repeat(np.arange(n), sizes.astype(int))
    X = sp.csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)),
                      shape=(n, int(cols.max()) + 1))
    inter = (X @ X.T).toarray().astype(np.float64)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def ranked(docs: dict[str, float]) -> list[str]:
    """trec_eval order: score descending, ties by docid descending."""
    order = sorted(docs.items(), key=lambda x: x[0], reverse=True)
    order.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in order]


def profile(run: dict[str, dict[str, float]], root=None) -> dict[str, dict]:
    """Per query: ranked doc ids, max Jaccard to any higher-ranked doc (and which one), and
    whether the doc strictly repeats a higher-ranked one (and which one).

    Any threshold can then be applied without touching the documents again.
    """
    by_dom: dict[str, list[str]] = defaultdict(list)
    for q, docs in run.items():
        first = next(iter(docs))
        by_dom[first.split("/", 1)[0]].append(q)
    out = {}
    for dom in sorted(by_dom):
        qids = by_dom[dom]
        need = {d for q in qids for d in run[q]}
        texts = fetch_docs(need, root)
        vocab: dict[str, int] = {}
        feats = {d: (strict_key(t), shingle_hashes(t, vocab)) for d, t in texts.items()}
        for q in qids:
            lst = ranked(run[q])
            keys = [feats[d][0] if d in feats else hash(d) for d in lst]
            sh = [feats[d][1] if d in feats else np.zeros(0, dtype=np.uint64) for d in lst]
            J = jaccard_matrix(sh)
            n = len(lst)
            maxprev = np.zeros(n)
            argprev = np.full(n, -1)
            strict = np.zeros(n, dtype=bool)
            strict_of = np.full(n, -1)
            first: dict[int, int] = {}
            for i in range(n):
                if i:
                    argprev[i] = int(J[i, :i].argmax())
                    maxprev[i] = J[i, argprev[i]]
                if keys[i] in first:
                    strict[i], strict_of[i] = True, first[keys[i]]
                else:
                    first[keys[i]] = i
            out[q] = {"docs": lst, "maxprev": maxprev, "argprev": argprev, "strict": strict,
                      "strict_of": strict_of}
        print(f"  {dom}: {len(qids)} queries, {len(texts)} documents", file=sys.stderr)
    return out


# ---------------------------------------------------------------- demotion --
def is_dup(p: dict, threshold: float | None) -> np.ndarray:
    """threshold None = strict only; otherwise strict OR Jaccard >= threshold."""
    if threshold is None:
        return p["strict"].copy()
    return p["strict"] | (p["maxprev"] >= threshold)


def demote(docs: list[str], dup: np.ndarray, skip: int = 1) -> list[str]:
    """Move each duplicate below the next `skip` non-duplicates (skip<=0: to the end)."""
    out: list[str] = []
    pending: list[tuple[str, int]] = []      # (doc, non-duplicates still to pass)
    for d, flag in zip(docs, dup):
        if flag:
            pending.append((d, skip if skip > 0 else 10**9))
            continue
        out.append(d)
        still = []
        for pd, left in pending:
            left -= 1
            if left <= 0:
                out.append(pd)
            else:
                still.append((pd, left))
        pending = still
    out.extend(pd for pd, _ in pending)
    return out


def apply(prof: dict[str, dict], threshold: float | None, skip: int = 1,
          depth: int = 100) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    """New run (rank-based scores) and, per query, the demoted doc ids."""
    run, demoted = {}, {}
    for q, p in prof.items():
        dup = is_dup(p, threshold)
        new = demote(p["docs"], dup, skip)
        run[q] = {d: float(depth - r) for r, d in enumerate(new)}
        demoted[q] = [d for d, f in zip(p["docs"], dup) if f]
    return run, demoted


def setting_name(threshold: float | None) -> str:
    return "strict" if threshold is None else f"jaccard>={threshold:g}"


# ------------------------------------------------------------------ tuning --
THRESHOLDS = [None, 1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]


def gold_demotions(demoted: dict[str, list[str]], split: str, root=None) -> tuple[int, int]:
    """(# demoted documents that are gold, # demoted documents)."""
    gold = {}
    for d in DOMAINS:
        gold.update(load_qrels(d, split, root))
    n_gold = sum(1 for q, ds in demoted.items() for x in ds if gold.get(q, {}).get(x, 0) > 0)
    return n_gold, sum(len(v) for v in demoted.values())


def grid(prof: dict[str, dict], split: str, skips=(1,), root=None) -> list[dict]:
    rows = []
    for skip in skips:
        for t in THRESHOLDS:
            run, demoted = apply(prof, t, skip)
            pq = per_query(run, split, root=root)
            ng, nd = gold_demotions(demoted, split, root)
            rows.append({"setting": setting_name(t), "threshold": t, "skip": skip,
                         "ndcg@10": macro(pq, "ndcg@10"), "recall@100": macro(pq, "recall@100"),
                         "ndcg@10_no_iota": macro([r for r in pq if r["domain"] != "iota"],
                                                  "ndcg@10"),
                         "demoted": nd, "demoted_gold": ng})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reteco.analysis_tools.dedup")
    ap.add_argument("--run", required=True, type=Path, help="input TREC run (any system)")
    ap.add_argument("--split", choices=SPLITS, default="train")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "dedup.yaml")
    ap.add_argument("--tune", action="store_true",
                    help="grid-search the threshold on this (train) run and write --config")
    ap.add_argument("--grid-out", type=Path, help="with --tune: CSV of the whole grid")
    ap.add_argument("--skip", type=int, default=1,
                    help="with --tune: demotion depth to select the threshold for "
                         "(1 = below the next non-duplicate, 0 = to the end of the list)")
    ap.add_argument("--out", type=Path, help="output run (default runs/<split>/<stem>+dedup.trec)")
    ap.add_argument("--no-log", action="store_true", help="do not append to results/experiments.csv")
    args = ap.parse_args(argv)

    run_in = read_run(args.run)
    prof = profile(run_in)

    if args.tune:
        if args.split != "train":
            ap.error("--tune is only allowed on the train split")
        rows = grid(prof, "train", skips=(1, 3, 0))
        base = macro(per_query(run_in, "train"), "ndcg@10")
        for r in rows:
            print(f"skip={r['skip']} {r['setting']:<14} nDCG@10 {r['ndcg@10']:.4f} "
                  f"({r['ndcg@10'] - base:+.4f})  R@100 {r['recall@100']:.4f}  "
                  f"demoted {r['demoted']} (gold {r['demoted_gold']})")
        # Selection at the given skip: best train nDCG@10, ties (within 1e-4) broken
        # towards the stricter (higher) threshold. THRESHOLDS runs strict -> loose.
        cand = [r for r in rows if r["skip"] == args.skip]
        best_val = max(r["ndcg@10"] for r in cand)
        best = next(r for r in cand if r["ndcg@10"] >= best_val - 1e-4)
        where = {1: "below the next non-duplicate", 0: "to the end of the list"}.get(
            args.skip, f"below the next {args.skip} non-duplicates")
        cfg = {"system_name": {1: "dedup", 0: "dedup_end"}.get(args.skip, f"dedup_skip{args.skip}"),
               "description": (f"near-duplicate demotion ({best['setting']}, word 5-gram shingles; "
                               f"duplicates moved {where}); tuned on train"),
               "threshold": best["threshold"], "skip": args.skip, "depth": 100, "seed": 13,
               "tuned_on": {"split": "train", "input_run": str(args.run),
                            "train_macro_ndcg@10": round(best["ndcg@10"], 4),
                            "train_baseline": round(base, 4)}}
        args.config.write_text("# Tuned on train by `python -m reteco.analysis_tools.dedup --tune`.\n"
                               + yaml.safe_dump(cfg, sort_keys=False))
        print(f"selected {best['setting']} -> {args.config}")
        if args.grid_out:
            import csv
            with open(args.grid_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["baseline_ndcg@10"])
                w.writeheader()
                for r in rows:
                    w.writerow({**r, "threshold": "strict" if r["threshold"] is None else r["threshold"],
                                "baseline_ndcg@10": base})
        return 0

    cfg = yaml.safe_load(args.config.read_text())
    run, demoted = apply(prof, cfg["threshold"], cfg.get("skip", 1), cfg.get("depth", 100))
    name = f"{args.run.stem}+{cfg['system_name']}"
    out = args.out or REPO_ROOT / "runs" / args.split / f"{name}.trec"
    write_run(run, out, tag=name.replace("+", "_"), depth=cfg.get("depth", 100))
    desc = f"{cfg['description']} [input: {args.run.name}]"
    n_dem = sum(len(v) for v in demoted.values())
    out.with_suffix(".meta.json").write_text(json.dumps(
        {"description": desc, "config": str(args.config), "input_run": str(args.run),
         "demoted_documents": n_dem}, indent=2) + "\n")
    print(f"wrote {out} ({n_dem} documents demoted)")
    result = evaluate_run(read_run(out), args.split)
    print(format_table(result))
    if not args.no_log:
        append_result(result, out.stem, desc, str(args.config))
        print("appended to results/experiments.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
