#!/usr/bin/env bash
# Download docs_selected.jsonl and build the 8192-vocab dataset.
# Run once. Takes ~30-60 min depending on CPU.
#
# Usage: bash setup_8192.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Downloading docs_selected.jsonl ==="
python3 data/cached_challenge_fineweb.py --variant sp1024 --with-docs

echo "=== Training 8192-vocab tokenizer and exporting shards ==="
python3 data/download_hf_docs_and_tokenize.py \
    --repo-id willdepueoai/parameter-golf \
    --remote-root datasets \
    --output-root /tmp/param_golf_8192 \
    --tokenizer-config data/tokenizer_specs_8192.json \
    --skip-byte

echo "=== Copying tokenizer and dataset into data/ ==="
mkdir -p data/tokenizers data/datasets
cp /tmp/param_golf_8192/tokenizers/fineweb_8192_bpe.model data/tokenizers/
cp /tmp/param_golf_8192/tokenizers/fineweb_8192_bpe.vocab data/tokenizers/ 2>/dev/null || true
cp -r /tmp/param_golf_8192/datasets/fineweb10B_sp8192 data/datasets/

echo "=== Done. Dataset at data/datasets/fineweb10B_sp8192/ ==="
ls data/datasets/fineweb10B_sp8192/ | head -5
