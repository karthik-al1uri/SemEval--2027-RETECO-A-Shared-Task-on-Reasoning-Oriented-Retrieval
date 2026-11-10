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
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dense.txt
```

## Layout

```
src/reteco/data.py       loaders: queries (id, query only), streamed corpus, qrels
src/reteco/runs.py       read/write TREC run files
src/reteco/evaluate.py   pytrec_eval scorer + CLI, logs to results/experiments.csv
src/reteco/analysis.py   Lucene EnglishAnalyzer approximation (no JDK)
src/reteco/bm25.py       BM25 baseline
configs/                 one YAML per system
runs/<split>/            TREC run files (top 100 per query)
results/experiments.csv  one row per scored run
cache/                   indexes etc. (gitignored)
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
pytest
```
