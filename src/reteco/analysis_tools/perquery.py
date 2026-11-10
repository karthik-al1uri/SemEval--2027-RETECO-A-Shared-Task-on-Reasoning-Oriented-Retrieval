"""Per-query nDCG@10 and Recall@100 for existing TREC runs.

Same conventions as `reteco.evaluate`: every query in a domain's qrels counts and
a query missing from the run scores 0. The mean of `per_query` over a domain
therefore equals `evaluate.score_domain`, and the mean over domains equals the
logged macro score.
"""
from __future__ import annotations

from pathlib import Path

import pytrec_eval

from ..data import DOMAINS, REPO_ROOT, load_qrels
from ..evaluate import METRICS
from ..runs import read_run


def run_path(system: str, split: str) -> Path:
    return REPO_ROOT / "runs" / split / f"{system}.trec"


def per_query(run: dict[str, dict[str, float]], split: str, domains=None, root=None) -> list[dict]:
    """One row per query: domain, qid, ndcg@10, recall@100."""
    rows = []
    for d in domains or DOMAINS:
        qrels = load_qrels(d, split, root)
        sub = {q: run[q] for q in qrels if q in run}
        res = pytrec_eval.RelevanceEvaluator(qrels, METRICS).evaluate(sub) if sub else {}
        for q in sorted(qrels):
            r = res.get(q, {})
            rows.append({"domain": d, "qid": q,
                         "ndcg@10": r.get("ndcg_cut_10", 0.0),
                         "recall@100": r.get("recall_100", 0.0)})
    return rows


def per_query_systems(systems, split: str, root=None) -> dict[str, list[dict]]:
    """system -> per-query rows (all systems share the same row order)."""
    return {s: per_query(read_run(run_path(s, split)), split, root=root) for s in systems}


def macro(rows: list[dict], metric: str) -> float:
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["domain"], []).append(r[metric])
    return sum(sum(v) / len(v) for v in by.values()) / len(by)
