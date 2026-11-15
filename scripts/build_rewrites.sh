#!/usr/bin/env bash
# Rebuild the local LLM query rewrites that configs/final_r2.yaml needs. They live in the
# gitignored cache/llm_local/, so a fresh clone must run this before the final_r2 pipeline:
#
#   scripts/build_rewrites.sh            (fills in missing rewrites; uses .venv-dense, PYTHON=... to override)
#   scripts/build_rewrites.sh --fresh    (deletes the cached rewrites first)
#   python -m reteco.pipeline --config configs/final_r2.yaml
#
# Qwen/Qwen2.5-1.5B-Instruct (downloaded once from the Hugging Face hub, then run locally; no API)
# turns each post into a short search query (<= 16 words), from the query text only (the system
# input at prediction time). Greedy decoding, fp16 on MPS, batches of 16. Output:
#   cache/llm_local/Qwen-Qwen2.5-1.5B-Instruct/query/{train,dev}/<domain>.jsonl
# About 15 minutes for the 1,730 train + dev queries on an Apple M4.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv-dense/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
OUT=cache/llm_local/Qwen-Qwen2.5-1.5B-Instruct/query

if [[ "${1:-}" == "--fresh" ]]; then
    rm -rf "$OUT/train" "$OUT/dev"
fi
"$PY" -m reteco.playground.llm_rewrite --model Qwen/Qwen2.5-1.5B-Instruct --mode query --splits train dev
echo "done; now: python -m reteco.pipeline --config configs/final_r2.yaml"
