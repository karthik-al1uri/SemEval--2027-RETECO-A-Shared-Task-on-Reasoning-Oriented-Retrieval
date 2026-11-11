"""Paired comparison of two playground runs on holdout: delta, domain-stratified bootstrap CI,
wins/losses, and the same without iota.

    python -m reteco.playground.compare runs/train/pg_e1.trec runs/train/pg_e5.trec
"""
from __future__ import annotations

import sys

import numpy as np

from ..runs import read_run
from .common import load_split, paired_bootstrap, per_query_ndcg


def main(argv=None) -> int:
    a_path, b_path = (argv or sys.argv[1:])[:2]
    a = per_query_ndcg(read_run(a_path), "holdout")
    b = per_query_ndcg(read_run(b_path), "holdout")
    lo, hi = paired_bootstrap(a, b)
    d = np.array([b[q] - a[q] for q in a])
    sp = load_split()["domains"]
    per = {dom: np.mean([b[q] - a[q] for q in v["holdout"]]) for dom, v in sp.items()}
    print(f"delta macro {np.mean(list(per.values())):+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"better {(d > 0).sum()} worse {(d < 0).sum()} same {(d == 0).sum()}")
    print("delta no iota", f"{np.mean([v for k, v in per.items() if k != 'iota']):+.4f}")
    print(" ".join(f"{k[:4]}={v:+.3f}" for k, v in per.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
