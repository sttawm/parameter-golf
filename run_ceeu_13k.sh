#!/usr/bin/env bash
# ceeu 13k-step run for per-step comparison against baseline.
# CE + embed aux (lambda=1.0) + uniform (gamma=2.0), untied, seed=1337.
# MAX_WALLCLOCK_SECONDS=0 so it runs all 13,000 steps (~18 min on 8xH100).
#
# Usage: bash run_ceeu_13k.sh

set -euo pipefail

cd /workspace/parameter-golf/parameter-golf

echo "=== ceeu 13k-step run ==="

export RUN_ID="ceeu_8gpu_13k_$(date +%Y%m%d_%H%M%S)_lam1_g2_s1337"
export TIE_EMBEDDINGS=0
export EMBED_LOSS_LAMBDA=1.0
export UNIFORM_LOSS_GAMMA=2.0
export SEED=1337
export ITERATIONS=13682
export MAX_WALLCLOCK_SECONDS=0
export TRAIN_LOG_EVERY=100
export VAL_LOSS_EVERY=1000

echo "run_id: ${RUN_ID}"
echo "Starting torchrun..."

nohup torchrun --nproc_per_node=8 train_gpt.py \
    > /workspace/parameter-golf/parameter-golf/logs/${RUN_ID}.txt 2>&1 &

echo "PID=$!"
echo "Log: logs/${RUN_ID}.txt"
