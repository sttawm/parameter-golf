#!/bin/bash
# Pull CSVs from all pods and merge into combined files.
# Usage: bash collect_results.sh
# Safe to run mid-run — gets whatever is done so far.

set -e
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"
LAMA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$LAMA_DIR/pod_results"
mkdir -p "$OUT_DIR"

declare -a PODS=(
  "194.68.245.8:22034"
  "194.68.245.209:22105"
  "194.68.245.55:22121"
  "69.30.85.61:22071"
  "194.68.245.6:22007"
  "69.30.85.67:22103"
  "194.68.245.4:22160"
  "69.30.85.104:22078"
)

# ── Pull CSVs ─────────────────────────────────────────────────────────────────
for pod in "${PODS[@]}"; do
  IFS=: read host port <<< "$pod"
  tag="${host}_${port}"
  echo "Pulling $tag..."
  scp $SSH_OPTS -P $port "root@$host:/root/lama_results_cosine_multi.csv" \
    "$OUT_DIR/cos_${tag}.csv" 2>/dev/null || echo "  [cosine missing]"
  scp $SSH_OPTS -P $port "root@$host:/root/lama_results_l2_multi.csv" \
    "$OUT_DIR/l2_${tag}.csv" 2>/dev/null || echo "  [l2 missing]"
done

# ── Merge with Python ─────────────────────────────────────────────────────────
python3 - <<'PYEOF'
import pandas as pd, glob, os

out = os.path.dirname(os.path.abspath(__file__))
pod_dir = os.path.join(out, "pod_results")

def merge(pattern, out_path):
    files = glob.glob(os.path.join(pod_dir, pattern))
    if not files:
        print(f"No files for {pattern}")
        return
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    # dedup by (seed, lambda, step/epoch) — keep last (most recent)
    combined = combined.drop_duplicates(subset=["seed", "lambda", "step"], keep="last")
    combined.to_csv(out_path, index=False)
    finals = combined[combined["epoch"] == "final"]
    print(f"{out_path}: {len(combined)} rows, {len(finals)} final runs")
    print(finals.groupby("lambda")["seed"].count().to_string())
    print()

merge("cos_*.csv", os.path.join(out, "lama_results_cosine_merged.csv"))
merge("l2_*.csv",  os.path.join(out, "lama_results_l2_merged.csv"))
PYEOF

echo "Done. Results in pod_results/ and merged CSVs in lama/."
