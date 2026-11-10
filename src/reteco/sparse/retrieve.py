"""BM25 search with optional RM3 pseudo-relevance feedback.

RM3 (Abdul-Jaleel et al. 2004; parameters as in Anserini):

  1. retrieve the top `fb_docs` documents with BM25
  2. relevance model  P(w|R) ∝ sum_d  s(d) * tf(w, d) / |d|
     over the feedback docs, with s(d) the BM25 scores normalised to sum 1.
     Candidate terms must be [a-z0-9]{2,20} and have df / N <= `max_df`
     (Anserini drops terms in more than 10% of documents).
  3. keep the top `fb_terms` terms, L1-normalise
  4. q' = alpha * q_orig + (1 - alpha) * P(.|R), with q_orig the baseline BM25
     query weights L1-normalised, so alpha = 1 gives back plain BM25
  5. score(q', d) = sum_t q'(t) * w(t, d) with the BM25 document weights
"""
from __future__ import annotations

import re

import numpy as np

from ..analysis import LuceneEnglishAnalyzer
from ..data import Query
from .clean import clean_query
from .index import BM25Weights, TFIndex

_TERM_OK = re.compile(r"^[a-z0-9]{2,20}$")


def tokenize_queries(queries: list[Query], query_clean=(), analyzer=None) -> list[list[str]]:
    analyzer = analyzer or LuceneEnglishAnalyzer()
    steps = tuple(query_clean or ())
    return [analyzer.analyze(clean_query(q.query, steps) if steps else q.query) for q in queries]


def _l1(v: dict[int, float]) -> dict[int, float]:
    s = sum(v.values())
    return {k: x / s for k, x in v.items()} if s > 0 else {}


class RM3:
    def __init__(self, index: TFIndex, max_df: float = 0.1):
        self.index = index
        terms = index.terms
        ok = np.array([bool(_TERM_OK.match(t)) for t in terms])
        self.allowed = ok & (index.df / index.n_docs <= max_df)
        self.tf = index.tf  # CSR rows = documents

    def feedback_model(self, first_pass: list[tuple[int, float]], fb_docs: int, fb_terms: int) -> dict[int, float]:
        docs = first_pass[:fb_docs]
        if not docs:
            return {}
        rows = np.array([r for r, _ in docs])
        s = np.array([x for _, x in docs], dtype=np.float64)
        s = s / s.sum() if s.sum() > 0 else np.full(len(s), 1.0 / len(s))
        sub = self.tf[rows]
        dl = self.index.doc_len[rows]
        acc: dict[int, float] = {}
        for i in range(len(rows)):
            lo, hi = sub.indptr[i], sub.indptr[i + 1]
            if dl[i] <= 0:
                continue
            cols = sub.indices[lo:hi]
            vals = sub.data[lo:hi] * (s[i] / dl[i])
            keep = self.allowed[cols]
            for c, v in zip(cols[keep].tolist(), vals[keep].tolist()):
                acc[c] = acc.get(c, 0.0) + v
        top = sorted(acc.items(), key=lambda x: -x[1])[:fb_terms]
        return _l1(dict(top))

    @staticmethod
    def interpolate(q_orig: dict[int, float], fb: dict[int, float], alpha: float) -> dict[int, float]:
        q = {c: alpha * w for c, w in _l1(q_orig).items()}
        for c, w in fb.items():
            q[c] = q.get(c, 0.0) + (1.0 - alpha) * w
        return q


def search(index: TFIndex, token_lists: list[list[str]], k1: float, b: float, depth: int = 100,
           rm3: dict | None = None, weights: BM25Weights | None = None) -> list[list[tuple[str, float]]]:
    """Ranked (doc_id, score) lists, one per query."""
    weights = weights or index.bm25(k1, b)
    qvecs = [weights.query_vector(t) for t in token_lists]
    if rm3:
        fb_docs, fb_terms, alpha = rm3["fb_docs"], rm3["fb_terms"], rm3["alpha"]
        model = RM3(index, rm3.get("max_df", 0.1))
        first = weights.score_many(qvecs, depth=fb_docs)
        qvecs = [model.interpolate(q, model.feedback_model(f, fb_docs, fb_terms), alpha) if q else q
                 for q, f in zip(qvecs, first)]
    ranked = weights.score_many(qvecs, depth=depth)
    ids = index.doc_ids
    return [[(ids[r], s) for r, s in lst] for lst in ranked]
