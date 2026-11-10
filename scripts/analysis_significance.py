"""Paired significance tests between systems on dev (macro over 13 domains).

    PYTHONPATH=src python scripts/analysis_significance.py

Reads results/analysis_per_query.csv (scripts/analysis_breakdown.py).
Writes results/analysis_significance.csv.

"Consecutive" pairs follow the progression bm25 -> bm25+temporal -> sparse_best
-> dense_best -> hybrid_best. Three more pairs answer the questions the
progression skips (do tuned sparse and dense each beat the baseline; does fusion
beat its sparse input). Holm correction is applied per metric over all seven pairs.
"""
from __future__ import annotations

import csv

import numpy as np

from reteco.analysis_tools import SYSTEMS
from reteco.analysis_tools.significance import (bootstrap_ci, holm, macro_diff,
                                                randomization_test)
from reteco.data import REPO_ROOT

OUT = REPO_ROOT / "results"
N_PERM = 10000
N_BOOT = 10000

PAIRS = [(SYSTEMS[i], SYSTEMS[i + 1], "consecutive") for i in range(len(SYSTEMS) - 1)] + [
    ("bm25", "sparse_best", "extra"),
    ("bm25", "dense_best", "extra"),
    ("sparse_best", "hybrid_best", "extra"),
]


def main(split: str = "dev") -> None:
    with open(OUT / "analysis_per_query.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    doms = [r["domain"] for r in rows]
    doms_no_iota = [d for d in doms if d != "iota"]
    keep = np.array([d != "iota" for d in doms])

    out = []
    for metric in ("ndcg@10", "recall@100"):
        res = []
        for a, b, kind in PAIRS:
            xa = np.array([float(r[f"{a}|{metric}"]) for r in rows])
            xb = np.array([float(r[f"{b}|{metric}"]) for r in rows])
            delta = macro_diff(xb, xa, doms)
            p = randomization_test(xb, xa, doms, n=N_PERM)
            lo, hi = bootstrap_ci(xb, xa, doms, n=N_BOOT)
            d_ni = macro_diff(xb[keep], xa[keep], doms_no_iota)
            p_ni = randomization_test(xb[keep], xa[keep], doms_no_iota, n=N_PERM)
            res.append([kind, a, b, metric, float(np.mean(xa)), float(np.mean(xb)), delta, lo, hi,
                        p, d_ni, p_ni, int((xb > xa).sum()), int((xb < xa).sum()),
                        int((xb == xa).sum())])
        adj = holm([r[9] for r in res])
        for r, pa in zip(res, adj):
            out.append(r[:10] + [pa] + r[10:])
            print(f"{metric:<10} {r[1]:>26} -> {r[2]:<26} d={r[6]:+.4f} "
                  f"[{r[7]:+.4f},{r[8]:+.4f}] p={r[9]:.4f} holm={pa:.4f} | no iota d={r[10]:+.4f} "
                  f"p={r[11]:.4f}")

    header = ["pair_kind", "system_a", "system_b", "metric", "micro_a", "micro_b", "macro_delta",
              "ci_low", "ci_high", "p_randomization", "p_holm", "macro_delta_no_iota",
              "p_randomization_no_iota", "n_b_better", "n_b_worse", "n_tied"]
    with open(OUT / "analysis_significance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split"] + header)
        for r in out:
            w.writerow([split] + [f"{x:.4f}" if isinstance(x, float) else x for x in r])
    print("wrote", OUT / "analysis_significance.csv")


if __name__ == "__main__":
    main()
