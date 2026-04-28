#!/usr/bin/env bash
# Runs all three experiment sweeps back to back:
#   1. Tied embeddings, cosine loss   (600s × 5 runs ≈ 50 min)
#   2. Untied embeddings, cosine loss (600s × 5 runs ≈ 50 min)
#   3. Untied embeddings, L2 loss     (600s × 5 runs ≈ 50 min)
#
# Usage: bash run_all_sweeps.sh

set -euo pipefail

echo "======================================================"
echo " Sweep 1/3: tied embeddings, cosine loss"
echo "======================================================"
bash sweep_embed_lambda.sh

echo "======================================================"
echo " Sweep 2/3: untied embeddings, cosine loss"
echo "======================================================"
TIE_EMBEDDINGS=0 bash sweep_embed_lambda.sh

echo "======================================================"
echo " Sweep 3/3: untied embeddings, L2 loss"
echo "======================================================"
TIE_EMBEDDINGS=0 EMBED_LOSS_L2=1 bash sweep_embed_lambda.sh

echo "======================================================"
echo " All sweeps complete. Logs in logs/"
echo "======================================================"
