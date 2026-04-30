#!/usr/bin/env bash
# Reconstruction loss sweep: CE(U(E[i]), i) as auxiliary loss.
# Untied weights, β=0.01/0.05/0.2, single seed, full 10-minute runs.
# Baseline (β=0) is available from existing convu_*_lam0_* logs.
#
# Usage: bash sweep_recon.sh

set -euo pipefail

SEED="${SEED:-1337}"
GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=== Recon loss sweep  date:${DATE_TAG}  gpus:${GPUS}  seed:${SEED} ==="

run() {
    local tag="$1"; shift
    local run_id="recon_${DATE_TAG}_${tag}"
    echo ""
    echo "--- Starting: ${run_id} ---"
    env \
        RUN_ID="${run_id}" \
        MAX_WALLCLOCK_SECONDS=600 \
        TRAIN_LOG_EVERY=20 \
        VAL_LOSS_EVERY=500 \
        TIE_EMBEDDINGS=0 \
        SEED="${SEED}" \
        "$@" \
        torchrun --nproc_per_node="${GPUS}" train_gpt.py
    echo "--- Done: ${run_id} ---"
}

run "b01"   RECON_LOSS_BETA=0.01
run "b05"   RECON_LOSS_BETA=0.05
run "b0p2"  RECON_LOSS_BETA=0.2

echo ""
echo "=== Recon sweep complete. Logs in logs/ ==="
ls -1 logs/recon_${DATE_TAG}_*.txt 2>/dev/null || echo "(check logs/ manually)"
