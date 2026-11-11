#!/usr/bin/env bash
# Rebuild, from scratch, the two fine-tuned models that configs/final.yaml needs. They live in
# the gitignored cache/playground/, so a fresh clone must run this before the pipeline:
#
#   scripts/build_finetuned_models.sh          (uses .venv-dense; set PYTHON=... to override)
#   python -m reteco.pipeline --config configs/final.yaml
#
# 1. cache/playground/ce/minilm_ft_q128_all/epoch1   MiniLM cross-encoder (configs/pg_e20.yaml)
# 2. cache/playground/qenc/bge_q_all/epoch2          bge-small query encoder (configs/pg_e20_dense.yaml)
#
# Training data is TRAIN only: the 968 "tune" queries of configs/playground_split.json, with
# positives from qrels_train and hard negatives from the first-stage top 200 (docs/PLAYGROUND.md).
# Seed 13 throughout. About 25 minutes on an Apple M4 once the first-stage caches exist; on a
# fresh clone step 1 first builds the BM25 indexes and embeds the corpus with bge-small (~2.5 h).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv-dense/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

CE=cache/playground/ce/minilm_ft_q128_all
QENC=cache/playground/qenc/bge_q_all
rm -rf "$CE" "$QENC"

echo "== 1/2 cross-encoder -> $CE/epoch1"
# Hard negatives come from final_v1's first stage. The LR schedule spans 2 epochs; only epoch 1 is used.
"$PY" -m reteco.playground.finetune_ce --config configs/final_v1.yaml --name minilm_ft_q128_all \
    --epochs 2 --stop-after-epoch 1 --n-neg 7 --groups-per-step 4 --max-query-tokens 128 --train-on tune

echo "== 2/2 query encoder -> $QENC/epoch2"
"$PY" -m reteco.playground.finetune_qenc --config configs/pg_e5.yaml --name bge_q_all \
    --epochs 2 --train-on tune

test -f "$CE/epoch1/config.json" && test -f "$QENC/epoch2/config.json"
echo "done; now: python -m reteco.pipeline --config configs/final.yaml"
