#!/usr/bin/env bash
# Warm-start embed loss sweep (untied embeddings).
# Runs embed loss for the first N steps then drops to CE-only.
# Compares against untied baseline (lambda=0) as control.
#
# Usage: bash sweep_warmstart.sh

set -euo pipefail

WALLCLOCK="${MAX_WALLCLOCK_SECONDS:-600}"
LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
VAL_EVERY="${VAL_LOSS_EVERY:-500}"
SEED="${SEED:-1337}"
GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=== Warm-start sweep  date:${DATE_TAG}  wallclock:${WALLCLOCK}s  gpus:${GPUS} ==="

run() {
    local tag="$1"; shift
    local run_id="ws_${DATE_TAG}_${tag}"
    echo ""
    echo "--- Starting: ${run_id} ---"
    env \
        RUN_ID="${run_id}" \
        MAX_WALLCLOCK_SECONDS="${WALLCLOCK}" \
        TRAIN_LOG_EVERY="${LOG_EVERY}" \
        VAL_LOSS_EVERY="${VAL_EVERY}" \
        SEED="${SEED}" \
        TIE_EMBEDDINGS=0 \
        "$@" \
        torchrun --nproc_per_node="${GPUS}" train_gpt.py
    echo "--- Done: ${run_id} ---"
}

# Warm-start: lambda=0.1 for first N steps, then CE-only
run "ws50"       EMBED_LOSS_LAMBDA=0.1 EMBED_LOSS_CUTOFF_STEP=50
run "ws100"      EMBED_LOSS_LAMBDA=0.1 EMBED_LOSS_CUTOFF_STEP=100
run "ws200"      EMBED_LOSS_LAMBDA=0.1 EMBED_LOSS_CUTOFF_STEP=200

echo ""
echo "=== Warm-start sweep complete. Logs in logs/ ==="
ls -1 logs/ws_${DATE_TAG}_*.txt 2>/dev/null || echo "(check logs/ manually)"
