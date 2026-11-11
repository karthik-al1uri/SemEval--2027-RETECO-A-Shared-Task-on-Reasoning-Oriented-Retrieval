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
  playground/          tune/holdout experiment loop, cross-encoder and query-encoder fine-tuning
  pipeline.py          end-to-end final system and ablation
configs/               one YAML per system; configs/final.yaml is the final system, configs/final_v1.yaml the first one
scripts/               data statistics and analysis scripts, build_finetuned_models.sh
runs/<split>/          TREC run files (top 100 per query); only selected systems are committed
results/
  experiments.csv      one row per scored run (CRLF line endings)
  ablation.csv         leave-one-out ablation of final_v1 (dev)
  playground.csv       one row per playground experiment (tune/holdout of train)
  analysis_*.csv       breakdowns, significance tests, error analysis, near-duplicates
docs/                  write-ups: DATA, SPARSE, DENSE, TEMPORAL, RERANK, ANALYSIS, PLAYGROUND
tests/                 pytest suite
cache/                 indexes, embeddings, cross-encoder scores, fine-tuned models (gitignored)
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

`final` (`configs/final.yaml`): tuned BM25 and bge-small dense retrieval with a **fine-tuned query encoder**, fused with RRF → MiniLM cross-encoder **fine-tuned on train qrels with hard negatives** on the top 200, interpolated with the fusion score → temporal-fit rescorer → near-duplicate demotion. It came out of the playground loop (`docs/PLAYGROUND.md`), which tuned on an 80% "tune" part of train and judged on the other 20% ("holdout"). The previous final system, with off-the-shelf models, is kept as `final_v1` (`configs/final_v1.yaml`). Every setting was selected on train; dev was only scored.

| System | train nDCG@10 | dev nDCG@10 | dev R@100 | ms/query |
| --- | ---: | ---: | ---: | ---: |
| BM25 baseline (`bm25`) | 0.1076 | 0.1148 | 0.4045 | |
| BM25 + temporal (`bm25+temporal_fit_present`) | 0.1171 | 0.1304 | 0.4045 | |
| Tuned sparse (`sparse_best`) | 0.2435 | 0.2541 | 0.6531 | |
| Dense (`dense_best`) | 0.2261 | 0.2765 | 0.6259 | |
| Sparse + dense fusion (`hybrid_best`) | 0.2804 | 0.3189 | 0.7158 | |
| + MiniLM cross-encoder, interpolated (`rerank_minilm_interp`) | 0.2941 | 0.3316 | 0.7436 | 1053 |
| + temporal rescorer | 0.2976 | 0.3303 | 0.7436 | 1055 |
| + near-duplicate demotion = `final_v1` | 0.3025 | 0.3403 | 0.7436 | 1100 |
| **+ fine-tuned cross-encoder and query encoder, re-tuned fusion = `final`** | 0.4219 † (partly in-sample) | **0.3679** | 0.7397 | 1111 |

Macro over the 13 domains. Latency is per query on an Apple M4 (MPS); about 96% of it is the cross-encoder.

- **`final` vs `final_v1` on dev: +0.0276** (95% CI [+0.010, +0.046], domain-stratified paired bootstrap), +0.0353 without iota (3 dev queries; 0.3457 vs 0.3105). 226 dev queries improve and 123 get worse, and 9 of 13 domains improve. On the train holdout the gain was +0.052 (0.3155 → 0.3671); holdout overstates it because the 5 kept changes were picked from 15 tries on the same queries.
- † **Partly in-sample.** Both fine-tuned models of `final` were trained on the tune part of train (80% of its queries), so its train score is inflated and not comparable with the rows above. Its fair train-side estimate is the 20% holdout: 0.3671 (`final_v1`: 0.3155).
- `final_v1`'s gain over `hybrid_best` on dev is +0.021 (95% CI [+0.009, +0.035]). Its leave-one-out ablation is in `results/ablation.csv`. The write-ups are `docs/SPARSE.md`, `docs/DENSE.md`, `docs/TEMPORAL.md`, `docs/RERANK.md`, `docs/ANALYSIS.md` and `docs/PLAYGROUND.md`.

### Reproduce

After downloading the data and creating `.venv-dense` (see Setup):

```
source .venv-dense/bin/activate
scripts/build_finetuned_models.sh                                   # fresh clone: build the fine-tuned models first
python -m reteco.pipeline --config configs/final.yaml               # runs/{train,dev}/final.trec, logged
python -m reteco.pipeline --config configs/final_v1.yaml            # runs/{train,dev}/final_v1.trec
```

**On a fresh clone, run `scripts/build_finetuned_models.sh` before the pipeline.** `final` loads two fine-tuned models from `cache/playground/`, which is gitignored: the cross-encoder (`ce/minilm_ft_q128_all/epoch1`) and the bge-small query encoder (`qenc/bge_q_all/epoch2`). The script rebuilds both from scratch, using train only. It takes about 25 minutes on an M4 once the first-stage caches exist. From an empty `cache/` it first builds the BM25 indexes and embeds the corpus (about 2.5 h).

**Expected run-to-run variation.** Training on MPS is not bit-reproducible even with seed 13. A test rebuild with the script (31 min on the M4) gave models within 3.4e-5 (cross-encoder) and 7.7e-6 (query encoder) of the canonical weights, at **holdout nDCG@10 0.3667 vs 0.3671 (−0.0004, 95% CI [−0.0011, 0.0000])**. 240 of 243 holdout queries scored the same. Dev was not scored with the rebuilt models. The canonical models are the ones every logged `final` number comes from. A rebuilt copy should land within about ±0.001 of them.

The pipeline rebuilds every stage from its config and writes and scores the train and dev runs. Only input-keyed caches under `cache/` are reused, and missing ones are rebuilt. With the models and caches in place a run takes a few minutes; from an empty `cache/` it takes about 3 hours (2.4 h to embed the corpus, about 30 min of cross-encoder scoring). Add `--ablation` to also write `results/ablation.csv` (dev).

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
