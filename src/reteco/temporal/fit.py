"""Temporal-fit score between a query's time intervals and a document's.

For query intervals Q_i and document intervals D_j (closed year ranges) the
distance is the gap between the ranges (0 when they overlap):

    dist(i, j) = max(0, D_j.start - Q_i.end, Q_i.start - D_j.end)
    k(i, j)    = exp(-dist(i, j) / tau)

Three views of the fit, each in [0, 1]:

    doc_prox   mean_j max_i k(i, j)   share of the document's time mentions near the query time
    q_cover    mean_i max_j k(i, j)   share of the query's time anchors the document covers
    best       max_ij k(i, j)         closest single match (overlap -> 1)

    sim      = (w_doc * doc_prox + w_q * q_cover + w_best * best) / (w_doc + w_q + w_best)
    temporal = sim - center   if both query and document have time expressions
             = 0              otherwise ("no time information" is neutral)

A document (or query) without time expressions always scores exactly 0.
With center = 0 the score can only boost time-matching documents. With
center > 0, documents whose times are far from the query's go negative and
fall below time-less documents of equal retrieval score: the intended
"right topic, wrong time drops" behaviour, with no penalty for silence.

`PairFeatures` precomputes the per-pair minimum distances once so that
scoring for many (tau, weights, center) settings is vectorised.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FitParams:
    tau: float = 10.0
    w_doc: float = 1.0
    w_q: float = 1.0
    w_best: float = 0.0
    center: float = 0.25
    # Expression kinds to ignore on either side (e.g. "present", "relative").
    drop_kinds: tuple[str, ...] = ()
    # Ignore query/document intervals wider than this many years (None = keep all).
    max_width: int | None = None


def _filter(iv: list, p: FitParams) -> np.ndarray:
    rows = [(s, e) for s, e, k in iv
            if k not in p.drop_kinds and (p.max_width is None or e - s <= p.max_width)]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


def min_dists(q: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(per-doc-interval min distance to query, per-query-interval min distance to doc)."""
    dist = np.maximum(0.0, np.maximum(d[None, :, 0] - q[:, None, 1], q[:, None, 0] - d[None, :, 1]))
    return dist.min(axis=0), dist.min(axis=1)


class PairFeatures:
    """Flattened per-pair minimum distances for a list of (query, doc) pairs."""

    def __init__(self, q_times: list[list], d_times: list[list], p: FitParams):
        doc_parts, q_parts, doc_n, q_n = [], [], [], []
        qcache: dict[int, np.ndarray] = {}
        for qt, dt in zip(q_times, d_times):
            qa = qcache.get(id(qt))
            if qa is None:
                qa = qcache[id(qt)] = _filter(qt, p)
            da = _filter(dt, p)
            if len(qa) and len(da):
                dm, qm = min_dists(qa, da)
            else:
                dm = qm = np.empty(0)
            doc_parts.append(dm)
            q_parts.append(qm)
            doc_n.append(len(dm))
            q_n.append(len(qm))
        self.doc_n = np.asarray(doc_n)
        self.q_n = np.asarray(q_n)
        self.has = self.doc_n > 0
        self.doc_d = np.concatenate(doc_parts) if doc_parts else np.empty(0)
        self.q_d = np.concatenate(q_parts) if q_parts else np.empty(0)
        self.doc_seg = np.repeat(np.arange(len(doc_n)), self.doc_n)
        self.q_seg = np.repeat(np.arange(len(q_n)), self.q_n)

    def components(self, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.doc_n)
        kd = np.exp(-self.doc_d / tau)
        kq = np.exp(-self.q_d / tau)
        safe_dn = np.maximum(self.doc_n, 1)
        safe_qn = np.maximum(self.q_n, 1)
        doc_prox = np.bincount(self.doc_seg, kd, minlength=n) / safe_dn
        q_cover = np.bincount(self.q_seg, kq, minlength=n) / safe_qn
        best = np.zeros(n)
        if len(kd):
            np.maximum.at(best, self.doc_seg, kd)
        return doc_prox, q_cover, best

    def score(self, p: FitParams) -> np.ndarray:
        doc_prox, q_cover, best = self.components(p.tau)
        w = p.w_doc + p.w_q + p.w_best
        sim = (p.w_doc * doc_prox + p.w_q * q_cover + p.w_best * best) / w
        return np.where(self.has, sim - p.center, 0.0)


def temporal_fit(q_times: list, d_times: list, p: FitParams) -> float:
    """Temporal-fit score for a single pair (same definition as PairFeatures.score)."""
    return float(PairFeatures([q_times], [d_times], p).score(p)[0])
