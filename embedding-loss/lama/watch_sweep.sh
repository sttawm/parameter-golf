#!/bin/bash
# Polls all pods every 20 min, refreshes the bar chart, sends a Mac notification.
# Run in background: bash watch_sweep.sh &
# Stop with: kill $(cat /tmp/watch_sweep.pid)

echo $$ > /tmp/watch_sweep.pid
LAMA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL=1200  # 20 minutes

notify() {
  osascript -e "display notification \"$1\" with title \"Sweep update\" sound name \"Submarine\""
}

echo "Watcher started (PID $$). Polling every ${INTERVAL}s."
notify "Watcher started — polling every 20 min"

while true; do
  sleep $INTERVAL
  cd "$LAMA_DIR"

  bash collect_results.sh > /tmp/watch_sweep.log 2>&1
  python3 plot_val_bar.py >> /tmp/watch_sweep.log 2>&1

  DONE=$(python3 -c "
import pandas as pd, os
total = 0
for f in ['lama_results_cosine_merged.csv', 'lama_results_l2_merged.csv',
          'lama_results_cosine_multi.csv', 'lama_results_l2_multi.csv']:
    p = os.path.join('$LAMA_DIR', f)
    if os.path.exists(p):
        df = pd.read_csv(p)
        n = len(df[df['epoch']=='final'])
        if n > total: total = n
print(total)
" 2>/dev/null || echo "?")

  notify "Chart updated — ${DONE} runs complete"
  open "$LAMA_DIR/val_bar_chart.png"
  echo "[$(date '+%H:%M')] Refreshed — ${DONE} runs done"
done
