"""Second-stage rerankers on top of a first-stage candidate list.

    python -m reteco.rerank.run --config configs/rerank_bge_base.yaml     cross-encoder (+ interpolation)
    python -m reteco.rerank.bench                                         cross-encoder throughput on MPS

Cross-encoder scores are cached per (model, query, document) under
cache/rerank/scores/, with the measured time of every scoring call, so that
latency per query can be reported for any reranking depth.
"""
from pathlib import Path

from ..data import REPO_ROOT

CACHE_DIR = REPO_ROOT / "cache" / "rerank"
