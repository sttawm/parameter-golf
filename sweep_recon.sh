#!/usr/bin/env bash
# Reconstruction loss sweep: CE(U(E[i]), i) as auxiliary loss.
# Untied weights, β=0.01/0.05/0.2, 2 seeds, 200 steps each.
#
# Usage: bash sweep_recon.sh

set -euo pipefail

GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=== Recon loss sweep  date:${DATE_TAG}  gpus:${GPUS} ==="

run() {
    local tag="$1"; shift
    local run_id="recon_${DATE_TAG}_${tag}"
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
    run "b0_s${SEED}"    RECON_LOSS_BETA=0.0  SEED=${SEED}
    run "b01_s${SEED}"   RECON_LOSS_BETA=0.01 SEED=${SEED}
    run "b05_s${SEED}"   RECON_LOSS_BETA=0.05 SEED=${SEED}
    run "b0p2_s${SEED}"  RECON_LOSS_BETA=0.2  SEED=${SEED}
done

echo ""
echo "=== Recon sweep complete. Logs in logs/ ==="
ls -1 logs/recon_${DATE_TAG}_*.txt 2>/dev/null || echo "(check logs/ manually)"
