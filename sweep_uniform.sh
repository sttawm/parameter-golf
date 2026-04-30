#!/usr/bin/env bash
# Uniformity loss sweep: Wang & Isola (2020) spread term on token embeddings.
# CE only (no embed aux loss), γ=0/0.01/0.1/1.0, untied weights, 200 steps, 2 seeds.
# Goal: does forcing embedding spread help CE convergence?
#
# Usage: bash sweep_uniform.sh

set -euo pipefail

GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=== Uniform loss sweep  date:${DATE_TAG}  gpus:${GPUS} ==="

run() {
    local tag="$1"; shift
    local run_id="unif_${DATE_TAG}_${tag}"
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
    run "g0_s${SEED}"    UNIFORM_LOSS_GAMMA=0.0  SEED=${SEED}
    run "g001_s${SEED}"  UNIFORM_LOSS_GAMMA=0.01 SEED=${SEED}
    run "g01_s${SEED}"   UNIFORM_LOSS_GAMMA=0.1  SEED=${SEED}
    run "g1_s${SEED}"    UNIFORM_LOSS_GAMMA=1.0  SEED=${SEED}
done

echo ""
echo "=== Uniform sweep complete. Logs in logs/ ==="
ls -1 logs/unif_${DATE_TAG}_*.txt 2>/dev/null || echo "(check logs/ manually)"
