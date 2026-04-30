#!/usr/bin/env bash
# CE + uniformity sweep: L_ce + gamma*L_uniform, no embed aux loss.
# Untied weights, gamma=0.1/0.5/1.0, 1 seed, 200 steps.
#
# Usage: bash sweep_ce_uniform.sh

set -euo pipefail

SEED="${SEED:-1337}"
GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=== CE+Uniform sweep  date:${DATE_TAG}  gpus:${GPUS}  seed:${SEED} ==="

run() {
    local tag="$1"; shift
    local run_id="ceu_${DATE_TAG}_${tag}"
    echo ""
    echo "--- Starting: ${run_id} ---"
    env \
        RUN_ID="${run_id}" \
        ITERATIONS=200 \
        MAX_WALLCLOCK_SECONDS=0 \
        TRAIN_LOG_EVERY=5 \
        VAL_LOSS_EVERY=50 \
        TIE_EMBEDDINGS=0 \
        SEED="${SEED}" \
        "$@" \
        torchrun --nproc_per_node="${GPUS}" train_gpt.py
    echo "--- Done: ${run_id} ---"
}

run "g01"  UNIFORM_LOSS_GAMMA=0.1
run "g05"  UNIFORM_LOSS_GAMMA=0.5
run "g1"   UNIFORM_LOSS_GAMMA=1.0

echo ""
echo "=== CE+Uniform sweep complete. Logs in logs/ ==="
ls -1 logs/ceu_${DATE_TAG}_*.txt 2>/dev/null || echo "(check logs/ manually)"
