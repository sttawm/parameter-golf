#!/usr/bin/env bash
# Two 600s runs with ONLY embed loss (no CE term): tied vs untied weights.
# Diagnostic: does embed loss gradient alone decrease CE loss?
# Val eval is disabled — train CE is logged each step at no extra cost.
#
# Usage: bash sweep_embed_only.sh

set -euo pipefail

WALLCLOCK="${MAX_WALLCLOCK_SECONDS:-600}"
LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
SEED="${SEED:-1337}"
GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=== Embed-only runs  date:${DATE_TAG}  wallclock:${WALLCLOCK}s ==="

run() {
    local tag="$1"; shift
    local run_id="eo_${DATE_TAG}_${tag}"
    echo ""
    echo "--- Starting: ${run_id} ---"
    env \
        RUN_ID="${run_id}" \
        MAX_WALLCLOCK_SECONDS="${WALLCLOCK}" \
        TRAIN_LOG_EVERY="${LOG_EVERY}" \
        VAL_LOSS_EVERY=0 \
        SEED="${SEED}" \
        EMBED_LOSS_ONLY=1 \
        EMBED_LOSS_LAMBDA=0.1 \
        "$@" \
        torchrun --nproc_per_node="${GPUS}" train_gpt.py
    echo "--- Done: ${run_id} ---"
}

run "untied" TIE_EMBEDDINGS=0

echo ""
echo "=== Done. Logs in logs/eo_${DATE_TAG}_*.txt ==="
