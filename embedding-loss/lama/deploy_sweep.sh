#!/bin/bash
# Deploy lama_sweep_topup.py to all 8 RunPod instances and launch runs.
# Run from: /Users/sttawm/dev/parameter-golf/embedding-loss/lama/
# Usage: bash deploy_sweep.sh

set -e
KEY="$HOME/.ssh/id_ed25519"
LAMA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Pod definitions: "host port seeds..." ────────────────────────────────────
declare -a PODS=(
  "194.68.245.8   22034  3000"
  "194.68.245.209 22105  4000"
  "194.68.245.55  22121  5000"
  "69.30.85.61    22071  6000"
  "194.68.245.6   22007  42 123"
  "69.30.85.67    22103  456 789"
  "194.68.245.4   22160  1011 2024"
  "69.30.85.104   22078  7000 8000"
)

SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"

setup_and_run() {
  local host=$1 port=$2; shift 2; local seeds="$@"
  local tag="$host:$port"
  echo "[$tag] Deploying (seeds: $seeds)..."

  # Copy script and existing cosine CSV
  scp $SSH_OPTS -P $port \
    "$LAMA_DIR/lama_sweep_topup.py" \
    "$LAMA_DIR/lama_results_cosine_multi.csv" \
    "root@$host:/root/" 2>&1 | sed "s/^/[$tag] /"

  # Setup and launch (all in one SSH session)
  ssh $SSH_OPTS -p $port root@$host bash <<REMOTE 2>&1 | sed "s/^/[$tag] /"; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "[$tag] FAILED"; return 1; }
set -e
pip install -q torch transformers tqdm pandas

# Download LAMA T-REx data if not present
if [ ! -d /root/data ]; then
  echo "Downloading LAMA data..."
  curl -sL -o /tmp/lama_data.zip https://dl.fbaipublicfiles.com/LAMA/data.zip
  python3 -c "import zipfile; zipfile.ZipFile('/tmp/lama_data.zip').extractall('/tmp/lama_ex')"
  mv /tmp/lama_ex/data/TREx /root/data
  rm -rf /tmp/lama_data.zip /tmp/lama_ex
  echo "Data ready."
fi

cd /root
nohup python lama_sweep_topup.py --seeds $seeds > run.log 2>&1 &
echo "Launched PID \$!"
REMOTE

  echo "[$tag] Done."
}

# Launch all pods in parallel
for pod in "${PODS[@]}"; do
  read -r host port seeds <<< "$pod"
  setup_and_run $host $port $seeds &
done

wait
echo ""
echo "All pods launched. Monitor with:"
echo "  ssh root@<host> -p <port> -i $KEY 'tail -f /root/run.log'"
