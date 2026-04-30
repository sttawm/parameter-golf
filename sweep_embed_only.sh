#!/usr/bin/env bash
# Single 600s run with ONLY embed loss (no CE term).
# Diagnostic: does embed loss gradient alone decrease CE loss?
# Val eval is disabled — train CE is logged each step at no extra cost
# (logits already computed for embed loss, CE cached in forward pass).
#
# Usage: bash sweep_embed_only.sh

set -euo pipefail

WALLCLOCK="${MAX_WALLCLOCK_SECONDS:-600}"
LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
SEED="${SEED:-1337}"
GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)
RUN_ID="eo_${DATE_TAG}"

echo "=== Embed-only run  id:${RUN_ID}  wallclock:${WALLCLOCK}s ==="

env \
    RUN_ID="${RUN_ID}" \
    MAX_WALLCLOCK_SECONDS="${WALLCLOCK}" \
    TRAIN_LOG_EVERY="${LOG_EVERY}" \
    VAL_LOSS_EVERY=0 \
    SEED="${SEED}" \
    TIE_EMBEDDINGS=1 \
    EMBED_LOSS_ONLY=1 \
    EMBED_LOSS_LAMBDA=0.1 \
    torchrun --nproc_per_node="${GPUS}" train_gpt.py

echo ""
echo "=== Done. Log in logs/${RUN_ID}.txt ==="
