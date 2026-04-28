#!/usr/bin/env bash
# Embedding-loss lambda sweep: runs baseline + 3 lambda values sequentially.
# Usage: bash sweep_embed_lambda.sh
# Override duration:  MAX_WALLCLOCK_SECONDS=600 bash sweep_embed_lambda.sh
# Override log freq:  TRAIN_LOG_EVERY=10 bash sweep_embed_lambda.sh

set -euo pipefail

WALLCLOCK="${MAX_WALLCLOCK_SECONDS:-360}"
LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
VAL_EVERY="${VAL_LOSS_EVERY:-500}"
SEED="${SEED:-1337}"
GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=== Embed-lambda sweep  date:${DATE_TAG}  wallclock:${WALLCLOCK}s  gpus:${GPUS} ==="

run_experiment() {
    local lambda="$1"
    local tag="$2"
    local run_id="sweep_${DATE_TAG}_${tag}"
    echo ""
    echo "--- Starting: ${run_id}  EMBED_LOSS_LAMBDA=${lambda} ---"
    EMBED_LOSS_LAMBDA="${lambda}" \
    RUN_ID="${run_id}" \
    MAX_WALLCLOCK_SECONDS="${WALLCLOCK}" \
    TRAIN_LOG_EVERY="${LOG_EVERY}" \
    VAL_LOSS_EVERY="${VAL_EVERY}" \
    SEED="${SEED}" \
    torchrun --nproc_per_node="${GPUS}" train_gpt.py
    echo "--- Done: ${run_id} ---"
}

run_experiment "0.1"  "lambda0p1"
run_experiment "0.4"  "lambda0p4"
run_experiment "1.6"  "lambda1p6"
run_experiment "6.4"  "lambda6p4"

echo ""
echo "=== Sweep complete. Logs in logs/ ==="
echo "Log files for this sweep:"
ls -1 logs/sweep_${DATE_TAG}_*.txt 2>/dev/null || echo "(log files use run_id prefix in filename)"
