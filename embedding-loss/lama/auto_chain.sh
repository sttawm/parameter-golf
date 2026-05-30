#!/bin/bash
# Auto-chains experiments on RunPod:
#   1. Waits for L2 sweep (lama_results.csv) to complete all 6 lambdas
#   2. Runs cosine sweep (lama_finetune_cosine.py)
#   3. Runs per-relation breakdown (lama_per_relation.py)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/auto_chain.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "Auto-chain started. Watching for L2 sweep completion..."

# ── Wait for L2 sweep: all 6 lambdas need a 'final' row ───────────────────────
EXPECTED_LAMBDAS="0.0 0.1 0.5 1.0 2.0 4.0"
while true; do
    sleep 120
    if [ ! -f "$SCRIPT_DIR/lama_results.csv" ]; then
        log "lama_results.csv not found yet, waiting..."
        continue
    fi
    done=0
    for lam in $EXPECTED_LAMBDAS; do
        if grep -q "^${lam},final," "$SCRIPT_DIR/lama_results.csv" 2>/dev/null; then
            done=$((done + 1))
        fi
    done
    log "L2 sweep: $done/6 lambdas complete"
    if [ "$done" -eq 6 ]; then
        log "L2 sweep complete!"
        break
    fi
done

# ── Run cosine sweep ──────────────────────────────────────────────────────────
log "Starting cosine sweep..."
python "$SCRIPT_DIR/lama_finetune_cosine.py" 2>&1 | tee -a "$LOG"
log "Cosine sweep complete!"

# ── Run per-relation breakdown ────────────────────────────────────────────────
log "Starting per-relation breakdown..."
python "$SCRIPT_DIR/lama_per_relation.py" 2>&1 | tee -a "$LOG"
log "Per-relation breakdown complete!"

log "All experiments done!"
