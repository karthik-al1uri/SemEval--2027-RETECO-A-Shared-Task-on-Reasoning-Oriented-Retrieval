"""Score interpolation between the first stage and a reranker.

    final(q, d) = a * norm(first_stage(q, d)) + (1 - a) * norm(reranker(q, d))

over the reranked top `depth` of each query, with a per-query normalisation
(min-max or z-score). a = 0 is the reranker alone, a = 1 the first stage
alone. Documents below the reranking depth keep their first-stage order
underneath the reranked block. (a, norm) is tuned on train macro nDCG@10.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..evaluate import evaluate_run

A_GRID = [round(x, 2) for x in np.arange(0, 1.0001, 0.05)]
NORMS = ["minmax", "zscore"]


def norm(x: np.ndarray, method: str) -> np.ndarray:
    if method == "minmax":
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)
    if method == "zscore":
        sd = x.std()
        return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)
    raise ValueError(f"unknown normalisation {method!r}")


def combine(cands: dict[str, list[tuple[str, float]]], rerank: dict[str, dict[str, float]],
            depth: int, a: float, method: str = "minmax", out_depth: int = 100) -> dict[str, dict[str, float]]:
    out = {}
    for q, docs in cands.items():
        top = docs[:depth]
        if not top:
            out[q] = {}
            continue
        fs = np.array([s for _, s in top], dtype=np.float64)
        ce = np.array([rerank[q][d] for d, _ in top], dtype=np.float64)
        f = a * norm(fs, method) + (1 - a) * norm(ce, method)
        run = {d: float(s) for (d, _), s in zip(top, f)}
        low = float(f.min()) - 1.0
        for i, (d, _) in enumerate(docs[depth:out_depth]):
            run[d] = low - 1e-3 * (i + 1)
        # keep the out_depth best
        out[q] = dict(sorted(run.items(), key=lambda x: -x[1])[:out_depth])
    return out


def tune(cands, rerank, depth: int, split: str = "train", domains=None,
         a_grid=A_GRID, norms=NORMS) -> tuple[dict, list[dict]]:
    """Grid over (a, norm) on `split` (train only); ties go to smaller a, then minmax."""
    assert split == "train", "interpolation weights are tuned on train only"
    rows = []
    for method, a in itertools.product(norms, a_grid):
        if a == 0 and method != norms[0]:
            continue  # a = 0 is the same ranking for every normalisation
        res = evaluate_run(combine(cands, rerank, depth, a, method), split, domains)
        rows.append({"a": a, "norm": method, "ndcg@10": res["macro_ndcg@10"],
                     "recall@100": res["macro_recall@100"]})
    best = max(rows, key=lambda r: (round(r["ndcg@10"], 6), -r["a"], r["norm"] == "minmax"))
    return best, rows
