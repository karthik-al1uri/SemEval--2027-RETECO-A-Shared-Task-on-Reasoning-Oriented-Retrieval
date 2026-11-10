"""Paired significance tests for macro-averaged (over domains) IR metrics.

The macro score is the mean over domains of the per-domain query means, so a
difference between two systems is a weighted sum of per-query differences,
sum_i w_i d_i with w_i = 1 / (n_domains * n_queries(domain(i))). Both tests
respect that weighting:

- paired randomization (sign-flip) test: each query's two scores are swapped
  with probability 1/2; two-sided p = (#{|T*| >= |T|} + 1) / (R + 1).
- paired bootstrap: queries are resampled with replacement within each domain
  (domain sizes fixed), giving a percentile CI for the macro difference.
"""
from __future__ import annotations

import numpy as np

from .. import SEED


def _weights(domains: list[str]) -> np.ndarray:
    doms = np.asarray(domains)
    uniq, inv, counts = np.unique(doms, return_inverse=True, return_counts=True)
    return 1.0 / (len(uniq) * counts[inv])


def macro_diff(a, b, domains) -> float:
    return float(np.sum(_weights(domains) * (np.asarray(a) - np.asarray(b))))


def randomization_test(a, b, domains, n: int = 10000, seed: int = SEED,
                       chunk: int = 2000) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    wd = _weights(domains) * d
    obs = abs(wd.sum())
    if not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n:
        m = min(chunk, n - done)
        signs = rng.integers(0, 2, size=(m, len(d)), dtype=np.int8) * 2 - 1
        hits += int(np.sum(np.abs(signs @ wd) >= obs - 1e-12))
        done += m
    return (hits + 1) / (n + 1)


def bootstrap_ci(a, b, domains, n: int = 10000, seed: int = SEED,
                 alpha: float = 0.05) -> tuple[float, float]:
    d = np.asarray(a, float) - np.asarray(b, float)
    doms = np.asarray(domains)
    rng = np.random.default_rng(seed)
    total = np.zeros(n)
    uniq = list(dict.fromkeys(domains))
    for dom in uniq:
        x = d[doms == dom]
        idx = rng.integers(0, len(x), size=(n, len(x)))
        total += x[idx].mean(axis=1)
    total /= len(uniq)
    lo, hi = np.quantile(total, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def query_bootstrap_ci(d, n: int = 10000, seed: int = SEED,
                       alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of a plain (query-level) mean difference."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
