#!/usr/bin/env bash
# 1-GPU sweep: 3 seeds of baseline + 3 seeds of ceeu (lambda=1.0, gamma=2.0).
# 13,000 steps each, no wallclock cap. ~87 min/run, ~9 hrs total.
#
# Usage: bash run_1gpu_sweep.sh

set -euo pipefail

cd /workspace/parameter-golf/parameter-golf
mkdir -p logs

DATE_TAG=$(date +%Y%m%d_%H%M%S)
GPUS=1

run() {
    local run_id="$1"; shift
    echo ""
    echo "=== Starting: ${run_id} ==="
    env \
        RUN_ID="${run_id}" \
        ITERATIONS=13000 \
        MAX_WALLCLOCK_SECONDS=0 \
        TRAIN_LOG_EVERY=100 \
        VAL_LOSS_EVERY=1000 \
        "$@" \
        torchrun --nproc_per_node=${GPUS} train_gpt.py
    echo "=== Done: ${run_id} ==="
}

# Baseline: 3 seeds
run "baseline_1gpu_${DATE_TAG}_s1337"  TIE_EMBEDDINGS=0  SEED=1337
run "baseline_1gpu_${DATE_TAG}_s42"    TIE_EMBEDDINGS=0  SEED=42
run "baseline_1gpu_${DATE_TAG}_s1234"  TIE_EMBEDDINGS=0  SEED=1234

# ceeu (lambda=1.0, gamma=2.0): 3 seeds
run "ceeu_1gpu_${DATE_TAG}_lam1_g2_s1337"  TIE_EMBEDDINGS=0  EMBED_LOSS_LAMBDA=1.0  UNIFORM_LOSS_GAMMA=2.0  SEED=1337
run "ceeu_1gpu_${DATE_TAG}_lam1_g2_s42"    TIE_EMBEDDINGS=0  EMBED_LOSS_LAMBDA=1.0  UNIFORM_LOSS_GAMMA=2.0  SEED=42
run "ceeu_1gpu_${DATE_TAG}_lam1_g2_s1234"  TIE_EMBEDDINGS=0  EMBED_LOSS_LAMBDA=1.0  UNIFORM_LOSS_GAMMA=2.0  SEED=1234

echo ""
echo "=== All runs complete. Logs in logs/ ==="
ls -1 logs/*${DATE_TAG}*.txt 2>/dev/null
