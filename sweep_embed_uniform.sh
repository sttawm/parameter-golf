#!/usr/bin/env bash
# Embed + uniformity sweep: lambda*L_embed + gamma*L_uniform, NO CE.
# Tests whether uniform spread prevents embedding collapse and lets
# the embed aux loss signal actually lower CE.
# Untied weights, lambda=1.0 fixed, gamma=0.5/2.0/8.0, 1 seed, 200 steps.
#
# Usage: bash sweep_embed_uniform.sh

set -euo pipefail

SEED="${SEED:-1337}"
GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=== Embed+Uniform sweep  date:${DATE_TAG}  gpus:${GPUS}  seed:${SEED} ==="

run() {
    local tag="$1"; shift
    local run_id="eu_${DATE_TAG}_${tag}"
    echo ""
    echo "--- Starting: ${run_id} ---"
    env \
        RUN_ID="${run_id}" \
        ITERATIONS=200 \
        MAX_WALLCLOCK_SECONDS=0 \
        TRAIN_LOG_EVERY=5 \
        VAL_LOSS_EVERY=50 \
        TIE_EMBEDDINGS=0 \
        EMBED_LOSS_ONLY=1 \
        EMBED_LOSS_LAMBDA=1.0 \
        SEED="${SEED}" \
        "$@" \
        torchrun --nproc_per_node="${GPUS}" train_gpt.py
    echo "--- Done: ${run_id} ---"
}

run "g05"  UNIFORM_LOSS_GAMMA=0.5
run "g2"   UNIFORM_LOSS_GAMMA=2.0
run "g8"   UNIFORM_LOSS_GAMMA=8.0

echo ""
echo "=== Embed+Uniform sweep complete. Logs in logs/ ==="
ls -1 logs/eu_${DATE_TAG}_*.txt 2>/dev/null || echo "(check logs/ manually)"
