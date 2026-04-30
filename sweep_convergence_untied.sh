#!/usr/bin/env bash
# Early convergence diagnostic (untied embeddings): λ=0, 0.1, 0.4, 1.6 × 2 seeds, 200 steps each.
# Goal: compare embed loss effect on early convergence with untied weights.
#
# Usage: bash sweep_convergence_untied.sh

set -euo pipefail

GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=== Convergence sweep (untied)  date:${DATE_TAG}  gpus:${GPUS} ==="

run() {
    local tag="$1"; shift
    local run_id="convu_${DATE_TAG}_${tag}"
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
    run "lam0_s${SEED}"   EMBED_LOSS_LAMBDA=0.0 SEED=${SEED}
    run "lam0p1_s${SEED}" EMBED_LOSS_LAMBDA=0.1 SEED=${SEED}
    run "lam0p4_s${SEED}" EMBED_LOSS_LAMBDA=0.4 SEED=${SEED}
    run "lam1p6_s${SEED}" EMBED_LOSS_LAMBDA=1.6 SEED=${SEED}
done

echo ""
echo "=== Convergence sweep (untied) complete. Logs in logs/ ==="
ls -1 logs/convu_${DATE_TAG}_*.txt 2>/dev/null || echo "(check logs/ manually)"
