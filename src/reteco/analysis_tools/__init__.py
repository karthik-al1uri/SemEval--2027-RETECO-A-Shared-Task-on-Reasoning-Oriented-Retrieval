"""Post-hoc analysis of existing runs: query temporal types, per-type/per-domain
breakdowns, significance tests and error analysis.

Nothing here produces a retrieval system. Qrels are read only to score runs that
already exist; guidance labels are read only from guidance_train.jsonl, as
classifier training labels. guidance_dev.jsonl is never opened.
"""

# The systems compared in the analysis, in the order of the progression chart.
SYSTEMS = ["bm25", "bm25+temporal_fit_present", "sparse_best", "dense_best", "hybrid_best",
           "final"]

# Intermediate second-stage runs between hybrid_best and final (docs/RERANK.md). They are
# scored per query and tested for significance, but not shown as table/figure columns.
STAGES = ["rerank_minilm_interp", "rerank_minilm_interp+rerank_temporal_fit_present"]

# Short labels for tables and figures.
LABELS = {
    "bm25": "BM25",
    "bm25+temporal_fit_present": "BM25+temporal",
    "sparse_best": "Sparse (tuned)",
    "dense_best": "Dense",
    "hybrid_best": "Hybrid",
    "final": "Final",
    "rerank_minilm_interp": "Hybrid+CE",
    "rerank_minilm_interp+rerank_temporal_fit_present": "Hybrid+CE+temporal",
}
