#!/bin/bash
# Downloads and extracts the LAMA T-REx dataset.
# Run once before lama_finetune.py.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Downloading LAMA data..."
curl -L -o /tmp/lama_data.zip https://dl.fbaipublicfiles.com/LAMA/data.zip

echo "Extracting..."
unzip -q /tmp/lama_data.zip -d /tmp/lama_extracted

echo "Moving T-REx data..."
mv /tmp/lama_extracted/data/TREx "$SCRIPT_DIR/data"

echo "Cleaning up..."
rm -rf /tmp/lama_data.zip /tmp/lama_extracted

echo "Done. Data at $SCRIPT_DIR/data/"
