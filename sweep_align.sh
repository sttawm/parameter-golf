#!/usr/bin/env bash
# Alignment loss sweep: soft weight tying via cos(E[i], W[i]) penalty.
# Untied weights, α=0.5/1.0/2.0, 2 seeds, 200 steps.
# Baseline (α=0) available from existing convu_*_lam0_* logs.
#
# Story: instead of hard-tying (E=W) or fully untied, this lets the model
# choose how much to align input/output embeddings per token based on CE.
#
# Usage: bash sweep_align.sh

set -euo pipefail

GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=== Align loss sweep  date:${DATE_TAG}  gpus:${GPUS} ==="

run() {
    local tag="$1"; shift
    local run_id="align_${DATE_TAG}_${tag}"
    echo ""
    echo "--- Starting: ${run_id} ---"
    env \
        RUN_ID="${run_id}" \
        ITERATIONS=200 \
        MAX_WALLCLOCK_SECONDS=0 \
        TRAIN_LOG_EVERY=5 \
        VAL_LOSS_EVERY=50 \
        TIE_EMBEDDINGS=0 \
        "$@" \
        torchrun --nproc_per_node="${GPUS}" train_gpt.py
    echo "--- Done: ${run_id} ---"
}

for SEED in 1337 42; do
    run "a05_s${SEED}"  ALIGN_LOSS_ALPHA=0.5 SEED=${SEED}
    run "a1_s${SEED}"   ALIGN_LOSS_ALPHA=1.0 SEED=${SEED}
    run "a2_s${SEED}"   ALIGN_LOSS_ALPHA=2.0 SEED=${SEED}
done

echo ""
echo "=== Align sweep complete. Logs in logs/ ==="
ls -1 logs/align_${DATE_TAG}_*.txt 2>/dev/null || echo "(check logs/ manually)"
