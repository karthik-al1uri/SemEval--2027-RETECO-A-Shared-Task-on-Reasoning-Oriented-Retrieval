# Project rules

SemEval-2027 Task 1 (RETECO), Track 1, Sub-track 1a: Temporal Retrieval. Data v1.1 lives in `data/track1_tempo/<domain>/`.

## Task

- For each query in `examples_{train,dev}.jsonl`, rank documents from the SAME domain's `documents.jsonl` (`{id, content}`).
- Metric: nDCG@10 via `pytrec_eval`, macro-averaged over the 13 domains.

## Leakage rules

- System input at prediction time is only `id` and `query`.
- Never use qrels, `gold_ids`, `gold_answers` or `guidance_*.jsonl` as input at prediction time.
- `guidance_train.jsonl` and `qrels_train.txt` may be used as TRAINING labels only.
- Nothing from dev may be used for training.
- Tune on train; report on dev. Dev is checked, not optimized against.

## Outputs

- Every system writes a TREC run file (`qid Q0 docid rank score tag`), top 100 per query, to `runs/<split>/<system_name>.trec`. Run files over 50 MB are gitignored.
- Every scored run is appended to `results/experiments.csv`: date, system_name, split, macro nDCG@10, per-domain scores, short description, config path.
- Configs live in `configs/*.yaml`. Fixed random seed 13.

## Workflow

- Smoke-test new methods on small domains first (iota, quant, law, workplace, cardano), then run all 13.
- Cache expensive things (indexes, embeddings, LLM outputs) under `cache/` (gitignored).

## Git

- Commit messages are short and plain (e.g. "Add BM25 baseline").
- No AI attribution, no Co-Authored-By lines.
- Never push without asking.
