"""The one dev comparison at the end of a playground round: two dev run files, macro nDCG@10 over the
13 domains, domain-stratified paired bootstrap CI (10k) and paired randomization test (10k), per-domain
deltas, wins/losses, the same without iota, and R@100.

    python -m reteco.playground.dev_compare runs/dev/final.trec runs/dev/playground2_final.trec
"""
from __future__ import annotations

import sys

import numpy as np

from ..analysis_tools.perquery import macro, per_query
from ..analysis_tools.significance import bootstrap_ci, macro_diff, randomization_test
from ..runs import read_run


def main(argv=None) -> int:
    a_path, b_path = (argv or sys.argv[1:])[:2]
    a, b = per_query(read_run(a_path), "dev"), per_query(read_run(b_path), "dev")
    assert [r["qid"] for r in a] == [r["qid"] for r in b]
    dom = [r["domain"] for r in a]
    na, nb = [r["ndcg@10"] for r in a], [r["ndcg@10"] for r in b]
    print(f"A {a_path}: nDCG@10 {macro(a, 'ndcg@10'):.4f}  R@100 {macro(a, 'recall@100'):.4f}")
    print(f"B {b_path}: nDCG@10 {macro(b, 'ndcg@10'):.4f}  R@100 {macro(b, 'recall@100'):.4f}")
    lo, hi = bootstrap_ci(nb, na, dom)
    p = randomization_test(nb, na, dom)
    d = np.array(nb) - np.array(na)
    print(f"B - A: {macro_diff(nb, na, dom):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  randomization p = {p:.4f}  "
          f"better {(d > 0).sum()} worse {(d < 0).sum()} same {(d == 0).sum()}")
    keep = [i for i, x in enumerate(dom) if x != "iota"]
    sub = lambda v: [v[i] for i in keep]
    lo2, hi2 = bootstrap_ci(sub(nb), sub(na), sub(dom))
    print(f"without iota: {macro_diff(sub(nb), sub(na), sub(dom)):+.4f}  95% CI [{lo2:+.4f}, {hi2:+.4f}]")
    per = {}
    for x, v in zip(dom, d):
        per.setdefault(x, []).append(v)
    print(" ".join(f"{k}={np.mean(v):+.3f}({len(v)})" for k, v in per.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
