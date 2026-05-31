#!/bin/bash
# Deploy lama_lam20.py (cosine λ=20 with grad clipping) to RunPod instances.
# Usage: bash deploy_lam20.sh
#
# Spin up 5 pods (RunPod, any GPU), then fill in host/port below and run.

set -e
KEY="$HOME/.ssh/id_ed25519"
LAMA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Pod definitions: "host port seeds..." ─────────────────────────────────────
# Fill these in once pods are running. 10 seeds split across 5 pods (2 each).
declare -a PODS=(
  "FILL_HOST_1  FILL_PORT_1  42 123"
  "FILL_HOST_2  FILL_PORT_2  456 789"
  "FILL_HOST_3  FILL_PORT_3  1011 2024"
  "FILL_HOST_4  FILL_PORT_4  3000 4000"
  "FILL_HOST_5  FILL_PORT_5  5000 6000"
)

SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=15"

setup_and_run() {
  local host=$1 port=$2; shift 2; local seeds="$@"
  local tag="$host:$port"
  echo "[$tag] Deploying (seeds: $seeds)..."

  scp $SSH_OPTS -P $port \
    "$LAMA_DIR/lama_lam20.py" \
    "root@$host:/root/" 2>&1 | sed "s/^/[$tag] /"

  ssh $SSH_OPTS -p $port root@$host bash <<REMOTE 2>&1 | sed "s/^/[$tag] /"; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "[$tag] FAILED"; return 1; }
set -e
pip install -q torch transformers tqdm pandas

if [ ! -d /root/data ]; then
  echo "Downloading LAMA data..."
  curl -sL -o /tmp/lama_data.zip https://dl.fbaipublicfiles.com/LAMA/data.zip
  python3 -c "import zipfile; zipfile.ZipFile('/tmp/lama_data.zip').extractall('/tmp/lama_ex')"
  mv /tmp/lama_ex/data/TREx /root/data
  rm -rf /tmp/lama_data.zip /tmp/lama_ex
  echo "Data ready."
fi

cd /root
nohup python lama_lam20.py --seeds $seeds > run.log 2>&1 &
echo "Launched PID \$!"
REMOTE

  echo "[$tag] Done."
}

for pod in "${PODS[@]}"; do
  read -r host port seeds <<< "$pod"
  setup_and_run $host $port $seeds &
done

wait
echo ""
echo "All pods launched. Monitor with:"
echo "  ssh root@<host> -p <port> -i $KEY 'tail -f /root/run.log'"
echo ""
echo "Collect results with:"
echo "  bash collect_lam20.sh"
