#!/usr/bin/env python3
"""
Average CE loss over seeds and plot early convergence by lambda.

Usage:
    python plot_convergence.py logs/conv_<date>_*.txt
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

train_re  = re.compile(r"step:(\d+)/\d+ train_loss:([\d.]+)")
comps_re  = re.compile(r"step:(\d+) lambda:[\d.]+ ce:([\d.]+)")
lambda_re = re.compile(r"embed_loss_lambda:([\d.]+)")


def parse_log(path: Path) -> tuple[float, dict[int, float]]:
    lam = 0.0
    ce: dict[int, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = lambda_re.search(line)
            if m:
                lam = float(m.group(1))
                continue
            m = comps_re.search(line)
            if m:
                ce[int(m.group(1))] = float(m.group(2))
                continue
            m = train_re.search(line)
            if m:
                step = int(m.group(1))
                if step not in ce:
                    ce[step] = float(m.group(2))
    return lam, ce


def main(paths: list[Path]) -> None:
    # Group runs by lambda
    groups: dict[float, list[dict[int, float]]] = defaultdict(list)
    for p in paths:
        lam, ce = parse_log(p)
        groups[lam].append(ce)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors

    for i, lam in enumerate(sorted(groups)):
        runs = groups[lam]
        all_steps = sorted(set(s for r in runs for s in r))
        # Average CE across seeds at each step where all runs have a value
        avg, steps_used = [], []
        for s in all_steps:
            vals = [r[s] for r in runs if s in r]
            if vals:
                avg.append(np.mean(vals))
                steps_used.append(s)

        label = f"λ={lam} (n={len(runs)})" if lam > 0 else f"baseline (n={len(runs)})"
        ax.plot(steps_used, avg, label=label, color=colors[i], linewidth=2)

        # Plot individual seed runs as thin lines
        for run in runs:
            s_sorted = sorted(run)
            ax.plot(s_sorted, [run[s] for s in s_sorted],
                    color=colors[i], linewidth=0.5, alpha=0.4)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("CE loss", fontsize=12)
    ax.set_title("Early convergence: CE loss by lambda (avg ± seeds)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = Path("convergence_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main([Path(p) for p in sys.argv[1:]])
