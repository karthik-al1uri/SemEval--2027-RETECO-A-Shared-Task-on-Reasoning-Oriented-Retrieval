# SemEval-2027 RETECO: A Shared Task on Reasoning-Oriented Retrieval

This is an individual course project for CSCI 5832 (Natural Language Processing). It targets Track 1, Sub-track 1a (Temporal Retrieval) of SemEval-2027 Task 1 (RETECO). Systems are evaluated with nDCG@10.

- Task website: https://datascienceuibk.github.io/RETECO/
- Dataset: https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027
- Official starter kit: https://github.com/DataScienceUIBK/RETECO

Project rules (leakage, run files, logging, commits) are in `CLAUDE.md`.

## Data

The RETECO training and dev data (v1.1) is stored in `data/`. Sub-track 1a uses `data/track1_tempo/`. Statistics and examples are in `docs/DATA.md`.

```
huggingface-cli download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset --local-dir data/
```

Newer versions of `huggingface_hub` replace `huggingface-cli` with `hf`. The equivalent command is:

```
hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset --local-dir data/
```

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` also installs this repo as the `reteco` package (`src/reteco/`).

Dense retrieval (`src/reteco/dense/`) also needs `requirements-dense.txt` (torch, sentence-transformers, faiss-cpu) and Python 3.13, since torch and faiss wheels were not reliable on Python 3.14:

```
python3.13 -m venv .venv-dense
source .venv-dense/bin/activate
pip install -r requirements.txt -r requirements-dense.txt
```

The rerankers and the end-to-end pipeline (`src/reteco/rerank/`, `src/reteco/pipeline.py`) also run in this environment.

## Layout

```
src/reteco/
  data.py              loaders: queries (id, query only), streamed corpus, qrels
  runs.py              read/write TREC run files
  evaluate.py          pytrec_eval scorer + CLI, logs to results/experiments.csv
  analysis.py          Lucene EnglishAnalyzer approximation (no JDK)
  bm25.py              official BM25 baseline
  sparse/              tuned BM25, RM3, corpus and query cleaning
  dense/               bi-encoder retrieval and rank fusion
  temporal/            time extraction and temporal-fit rescoring
  rerank/              cross-encoder reranking and score interpolation
  analysis_tools/      query types, significance, error analysis, near-duplicate demotion
  pipeline.py          end-to-end final system and ablation
configs/               one YAML per system; configs/final.yaml is the final system
scripts/               data statistics and analysis scripts
runs/<split>/          TREC run files (top 100 per query); only selected systems are committed
results/
  experiments.csv      one row per scored run (CRLF line endings)
  ablation.csv         leave-one-out ablation of the final system (dev)
  analysis_*.csv       breakdowns, significance tests, error analysis, near-duplicates
docs/                  write-ups: DATA, SPARSE, DENSE, TEMPORAL, RERANK, ANALYSIS
tests/                 pytest suite
cache/                 indexes, embeddings, cross-encoder scores (gitignored)
```

## BM25 baseline

```
python -m reteco.bm25 --config configs/bm25.yaml --splits train dev
```

This builds one index per domain under `cache/bm25/` (about 2 minutes for all 13 domains on first run, then cached) and writes `runs/train/bm25.trec` and `runs/dev/bm25.trec`. Add `--domains iota quant law workplace cardano` for a quick smoke test.

It reproduces the official baseline: Lucene English analysis (StandardTokenizer, possessive and stopword filters, Porter stemming) and gensim's `LuceneBM25Model` scoring with k1=0.9, b=0.4. The analyzer is a pure-Python approximation, so no JDK or pyserini is needed.

| Macro nDCG@10 | train | dev |
| --- | ---: | ---: |
| Official BM25 (v1.1) | 0.1075 | 0.1147 |
| This repo | 0.1076 | 0.1148 |

## Final system

Tuned BM25 and bge-small dense retrieval fused with RRF → MiniLM cross-encoder on the top 200, interpolated with the fusion score → temporal-fit rescorer → near-duplicate demotion. Every setting was selected on train; dev was only scored.

| System | train nDCG@10 | dev nDCG@10 | dev R@100 | ms/query |
| --- | ---: | ---: | ---: | ---: |
| BM25 baseline (`bm25`) | 0.1076 | 0.1148 | 0.4045 | |
| BM25 + temporal (`bm25+temporal_fit_present`) | 0.1171 | 0.1304 | 0.4045 | |
| Tuned sparse (`sparse_best`) | 0.2435 | 0.2541 | 0.6531 | |
| Dense (`dense_best`) | 0.2261 | 0.2765 | 0.6259 | |
| Sparse + dense fusion (`hybrid_best`) | 0.2804 | 0.3189 | 0.7158 | |
| + MiniLM cross-encoder, interpolated (`rerank_minilm_interp`) | 0.2941 | 0.3316 | 0.7436 | 1053 |
| + temporal rescorer | 0.2976 | 0.3303 | 0.7436 | 1055 |
| **+ near-duplicate demotion = `final`** | **0.3025** | **0.3403** | **0.7436** | **1100** |

Macro over the 13 domains. Without iota (3 dev queries), `final` scores 0.3105 on dev. Its gain over `hybrid_best` on dev is +0.021 (95% CI [+0.009, +0.035]). Latency is per query on an Apple M4 (MPS); 96% of it is the cross-encoder. The leave-one-out ablation is in `results/ablation.csv`, and the write-ups are `docs/SPARSE.md`, `docs/DENSE.md`, `docs/TEMPORAL.md`, `docs/RERANK.md` and `docs/ANALYSIS.md`.

### Reproduce with one command

After downloading the data and creating `.venv-dense` (see Setup):

```
source .venv-dense/bin/activate
python -m reteco.pipeline --config configs/final.yaml              # runs/{train,dev}/final.trec, logged
```

This rebuilds every stage from `configs/final.yaml` and writes and scores `runs/train/final.trec` and `runs/dev/final.trec`. Only input-keyed caches under `cache/` are reused, and missing ones are rebuilt. From an empty `cache/` the run takes about 3 hours on an M4: 2.4 h to embed the corpus, about 30 min of cross-encoder scoring, and a few minutes of indexing. With the caches in place it takes a few minutes. Add `--ablation` to also write `results/ablation.csv` (dev).

## Evaluation

```
python -m reteco.evaluate runs/dev/bm25.trec --split dev
```

This prints per-domain and macro nDCG@10 (the official metric) and Recall@100 (a diagnostic), then appends a row to `results/experiments.csv`. Use `--no-log` to skip logging and `--domains ...` to score a subset. Queries missing from a run count as 0.

The scorer was checked against the official `starter_kit/scorer.py`. To repeat the check, clone the starter kit into `external/RETECO` (gitignored):

```
git clone https://github.com/DataScienceUIBK/RETECO external/RETECO
```

## Tests

```
source .venv-dense/bin/activate
pytest
```
