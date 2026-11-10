"""Sparse retrieval experiments: tuned BM25, RM3 feedback, corpus and query cleaning.

    python -m reteco.sparse.build  --corpus raw clean      # raw-tf indexes under cache/sparse/
    python -m reteco.sparse.tune   bm25 ...                # grid searches on train
    python -m reteco.sparse.run    --config configs/sparse_*.yaml --splits train dev

See docs/SPARSE.md for the results.
"""
