#!/usr/bin/env bash
# Overnight local experiment: baseline vs ceeu with 8192-vocab tokenizer.
# 5 hours each. Run on M1 Pro via MLX.
#
# Usage: bash run_8192_overnight.sh

set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate
mkdir -p logs

DATA_PATH="./data/datasets/fineweb10B_sp8192"
TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model"
DATE_TAG=$(date +%Y%m%d_%H%M%S)

run() {
    local run_id="$1"; shift
    echo ""
    echo "=== Starting: ${run_id} ==="
    env \
        RUN_ID="${run_id}" \
        DATA_PATH="${DATA_PATH}" \
        TOKENIZER_PATH="${TOKENIZER_PATH}" \
        VOCAB_SIZE=8192 \
        MAX_WALLCLOCK_SECONDS=18000 \
        TRAIN_LOG_EVERY=200 \
        VAL_LOSS_EVERY=1000 \
        MLX_EAGER_EVAL=0 \
        VAL_BATCH_SIZE=65536 \
        LOGIT_CHUNK_TOKENS=512 \
        "$@" \
        python3 train_gpt_mlx.py 2>&1 | tee "logs/${run_id}.txt"
    echo "=== Done: ${run_id} ==="
}

run "baseline_8192_${DATE_TAG}" \
    TIE_EMBEDDINGS=1

run "ceeu_8192_${DATE_TAG}_lam1_g2" \
    TIE_EMBEDDINGS=0 \
    EMBED_LOSS_LAMBDA=1.0 \
    UNIFORM_LOSS_GAMMA=2.0

echo ""
echo "=== All runs complete. Logs in logs/ ==="
ls -1 logs/*${DATE_TAG}*.txt 2>/dev/null
