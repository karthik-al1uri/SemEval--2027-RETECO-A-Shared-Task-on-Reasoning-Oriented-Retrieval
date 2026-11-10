"""Dense first-stage retrieval (bi-encoders + FAISS) and BM25/dense rank fusion.

    python -m reteco.dense.bench                         hardware + encoding throughput
    python -m reteco.dense.retrieve --config configs/dense_bge_small.yaml
    python -m reteco.dense.fusion --dense dense_bge_small --tune
"""
from pathlib import Path

from ..data import REPO_ROOT

CACHE_DIR = REPO_ROOT / "cache" / "dense"
CANDIDATES_DIR = CACHE_DIR / "candidates"   # depth-1000 TREC runs used for fusion
