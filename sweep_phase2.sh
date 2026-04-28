#!/usr/bin/env bash
# Phase 2 sweep — run after untied-cosine λ=0.1 (sweep 2) completes.
#
# Parts:
#   1. Untied cosine: λ=0 (baseline only; λ=0.1–6.4 already done in sweep 2)
#   2. Tied L2:       λ=4, 16, 64, 256  (L2-calibrated: cosine/L2 ratio ≈62x at init, conservative half-step)
#   3. Untied top-K:  λ=0.1, K=8, 32, 128  (compare approximation overhead vs full softmax)
#
# Usage: bash sweep_phase2.sh
# Override duration:  MAX_WALLCLOCK_SECONDS=600 bash sweep_phase2.sh

set -euo pipefail

WALLCLOCK="${MAX_WALLCLOCK_SECONDS:-600}"
LOG_EVERY="${TRAIN_LOG_EVERY:-20}"
VAL_EVERY="${VAL_LOSS_EVERY:-500}"
SEED="${SEED:-1337}"
GPUS="${NPROC_PER_NODE:-1}"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=== Phase 2 sweep  date:${DATE_TAG}  wallclock:${WALLCLOCK}s  gpus:${GPUS} ==="

run() {
    local tag="$1"; shift          # remaining args are KEY=VAL env overrides
    local run_id="p2_${DATE_TAG}_${tag}"
    echo ""
    echo "--- Starting: ${run_id} ---"
    env \
        RUN_ID="${run_id}" \
        MAX_WALLCLOCK_SECONDS="${WALLCLOCK}" \
        TRAIN_LOG_EVERY="${LOG_EVERY}" \
        VAL_LOSS_EVERY="${VAL_EVERY}" \
        SEED="${SEED}" \
        "$@" \
        torchrun --nproc_per_node="${GPUS}" train_gpt.py
    echo "--- Done: ${run_id} ---"
}

# ── Part 1: Untied cosine baseline ────────────────────────────────────────────
# λ=0.1, 0.4, 1.6, 6.4 already covered by sweep 2 (221812); only baseline needed.
echo ""
echo "======================================================"
echo " Part 1: Untied cosine baseline (λ=0)"
echo "======================================================"

run "untied_cos_baseline" TIE_EMBEDDINGS=0 EMBED_LOSS_LAMBDA=0.0

# ── Part 2: Tied L2 ───────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Part 2: Tied L2  (λ=4, 16, 64, 256)"
echo "======================================================"

run "tied_l2_4"   TIE_EMBEDDINGS=1 EMBED_LOSS_L2=1 EMBED_LOSS_LAMBDA=4.0
run "tied_l2_16"  TIE_EMBEDDINGS=1 EMBED_LOSS_L2=1 EMBED_LOSS_LAMBDA=16.0
run "tied_l2_64"  TIE_EMBEDDINGS=1 EMBED_LOSS_L2=1 EMBED_LOSS_LAMBDA=64.0
run "tied_l2_256" TIE_EMBEDDINGS=1 EMBED_LOSS_L2=1 EMBED_LOSS_LAMBDA=256.0

# ── Part 3: Untied top-K ──────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Part 3: Untied top-K  (λ=0.1, K=8, 32, 128)"
echo "======================================================"

run "untied_topk8"   TIE_EMBEDDINGS=0 EMBED_LOSS_LAMBDA=0.1 EMBED_LOSS_TOPK=8
run "untied_topk32"  TIE_EMBEDDINGS=0 EMBED_LOSS_LAMBDA=0.1 EMBED_LOSS_TOPK=32
run "untied_topk128" TIE_EMBEDDINGS=0 EMBED_LOSS_LAMBDA=0.1 EMBED_LOSS_TOPK=128

echo ""
echo "======================================================"
echo " Phase 2 complete. Logs in logs/"
echo "======================================================"
ls -1 logs/p2_${DATE_TAG}_*.txt 2>/dev/null || echo "(check logs/ manually)"
