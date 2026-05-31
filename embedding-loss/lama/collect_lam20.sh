#!/bin/bash
# Collect lama_results_cosine_lam20.csv from all pods and merge.
# Usage: bash collect_lam20.sh

KEY="$HOME/.ssh/id_ed25519"
LAMA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"

declare -a PODS=(
  "FILL_HOST_1  FILL_PORT_1"
  "FILL_HOST_2  FILL_PORT_2"
  "FILL_HOST_3  FILL_PORT_3"
  "FILL_HOST_4  FILL_PORT_4"
  "FILL_HOST_5  FILL_PORT_5"
)

TMPDIR=$(mktemp -d)
i=0
for pod in "${PODS[@]}"; do
  read -r host port <<< "$pod"
  echo "Fetching from $host:$port..."
  scp $SSH_OPTS -P $port \
    "root@$host:/root/lama_results_cosine_lam20.csv" \
    "$TMPDIR/pod${i}.csv" 2>/dev/null && echo "  OK" || echo "  (not found)"
  i=$((i+1))
done

python3 - <<'EOF'
import os, glob
import pandas as pd

tmpdir = os.environ.get("TMPDIR_PY")
files = glob.glob(f"{tmpdir}/pod*.csv")
if not files:
    print("No CSVs found.")
    exit(1)

dfs = [pd.read_csv(f) for f in files]
merged = pd.concat(dfs, ignore_index=True).drop_duplicates()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lama_results_cosine_lam20.csv")
merged.to_csv(out, index=False)
n_final = len(merged[merged["epoch"] == "final"])
print(f"Merged {len(files)} files → {n_final} completed runs → {out}")
EOF
export TMPDIR_PY="$TMPDIR"

rm -rf "$TMPDIR"
