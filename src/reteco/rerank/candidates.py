"""First-stage candidate lists for reranking.

`hybrid_best` is rebuilt here at depth 1000 from the dense agent's depth-1000
inputs (cache/dense/candidates/{sparse_best,dense_best}/<split>.trec) with the
frozen RRF setting of configs/dense_hybrid_best.yaml (k=20, w_dense=0.5). Its
top 100 is the committed runs/<split>/hybrid_best.trec.

Candidate lists are returned in trec_eval order (score desc, docid desc), the
order a run file lists them, so "the top d" means the same thing everywhere.
"""
from __future__ import annotations

from pathlib import Path

from ..data import DOMAINS, load_queries
from ..dense import CANDIDATES_DIR
from ..dense.fusion import ranks, rrf
from ..runs import _ordered, read_run, write_run
from . import CACHE_DIR

FIRST_STAGES = {
    # name: how to build it from the depth-1000 inputs
    "hybrid_best": {"fuse": ["sparse_best", "dense_best"], "weights": [0.5, 0.5], "k": 20},
    "sparse_best": {"input": "sparse_best"},
    "dense_best": {"input": "dense_best"},
}


def candidate_path(name: str, split: str) -> Path:
    return CACHE_DIR / "candidates" / name / f"{split}.trec"


def build(name: str, split: str, depth: int = 1000) -> Path:
    spec = FIRST_STAGES[name]
    if "input" in spec:
        return CANDIDATES_DIR / spec["input"] / f"{split}.trec"
    path = candidate_path(name, split)
    if path.exists():
        return path
    rank_lists = [ranks(read_run(CANDIDATES_DIR / n / f"{split}.trec")) for n in spec["fuse"]]
    fused = rrf(rank_lists, spec["weights"], spec["k"], depth=depth)
    return write_run(fused, path, tag=name, depth=depth)


def load(name: str, split: str, domains=None, depth: int = 1000) -> dict[str, list[tuple[str, float]]]:
    """qid -> [(docid, first-stage score)] in run order, for the given domains."""
    keep = {q.id for d in (domains or DOMAINS) for q in load_queries(d, split)}
    run = read_run(build(name, split, depth))
    return {q: _ordered(docs, depth) for q, docs in run.items() if q in keep}
