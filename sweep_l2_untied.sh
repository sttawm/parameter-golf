#!/usr/bin/env bash
# L2 embed loss with untied weights, small lambda calibrated for safe embed_frac.
# Usage: bash sweep_l2_untied.sh

set -euo pipefail

WALLCLOCK="${MAX_WALLCLOCK_SECONDS:-600}"
LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
VAL_EVERY="${VAL_LOSS_EVERY:-500}"
SEED="${SEED:-1337}"
GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=== L2 untied sweep  date:${DATE_TAG}  wallclock:${WALLCLOCK}s ==="

run() {
    local tag="$1"; shift
    local run_id="l2u_${DATE_TAG}_${tag}"
    echo ""
    echo "--- Starting: ${run_id} ---"
    env \
        RUN_ID="${run_id}" \
        MAX_WALLCLOCK_SECONDS="${WALLCLOCK}" \
        TRAIN_LOG_EVERY="${LOG_EVERY}" \
        VAL_LOSS_EVERY="${VAL_EVERY}" \
        SEED="${SEED}" \
        TIE_EMBEDDINGS=0 \
        EMBED_LOSS_L2=1 \
        "$@" \
        torchrun --nproc_per_node="${GPUS}" train_gpt.py
    echo "--- Done: ${run_id} ---"
}

run "lam0p1" EMBED_LOSS_LAMBDA=0.1

echo ""
echo "=== Done. Log in logs/ ==="
