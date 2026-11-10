"""Depth-1000 candidate lists from the sparse agent's `sparse_best` system, for fusion.

This file is run as a script against the *sparse* worktree's code, so that the
candidates come from exactly the code that produced sparse_best:

    PYTHONPATH=<sparse worktree>/src RETECO_DATA=<abs path to track1_tempo> \\
        .venv/bin/python -P src/reteco/dense/sparse_candidates.py \\
        --config <sparse worktree>/configs/sparse_best.yaml \\
        --out cache/dense/candidates/sparse_best

(-P keeps this folder off sys.path.) It only *loads* the cached raw-tf indexes
under cache/sparse/tf/ (never builds or saves them), uses the sparse config's
per-domain (k1, b) and query cleaning, and writes <out>/<split>.trec plus
<out>/meta.json. Nothing is written inside the sparse worktree or cache/sparse/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def main(argv=None) -> int:
    from reteco.data import DOMAINS, SPLITS, load_queries
    from reteco.runs import write_run
    from reteco.sparse.index import TFIndex
    from reteco.sparse.retrieve import search, tokenize_queries
    from reteco.sparse.run import domain_params

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--depth", type=int, default=1000)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    if cfg.get("rm3"):
        raise SystemExit("sparse config uses RM3; not supported here")
    out = args.out.resolve()
    if "candidates" not in out.parts:
        raise SystemExit(f"refusing to write outside a candidates folder: {out}")

    corpus = cfg.get("corpus", "raw")
    runs = {s: {} for s in args.splits}
    for domain in DOMAINS:
        t0 = time.time()
        path = TFIndex.cache_path(domain, corpus)
        if not (path / "meta.json").exists():
            raise SystemExit(f"no cached sparse index at {path}; refusing to build it")
        index = TFIndex.load(path)
        k1, b = domain_params(cfg, domain)
        weights = index.bm25(k1, b)
        for split in args.splits:
            qs = load_queries(domain, split)
            ranked = search(index, tokenize_queries(qs, cfg.get("query_clean")), k1, b,
                            args.depth, None, weights=weights)
            for q, r in zip(qs, ranked):
                runs[split][q.id] = dict(r)
        print(f"{domain:<10} k1={k1} b={b} {time.time() - t0:6.1f}s", flush=True)
        del index, weights
    for split in args.splits:
        write_run(runs[split], out / f"{split}.trec", tag="sparse_best", depth=args.depth)
    (out / "meta.json").write_text(json.dumps(
        {"source_config": str(Path(args.config).resolve()), "depth": args.depth,
         "corpus": corpus, "query_clean": cfg.get("query_clean")}, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
