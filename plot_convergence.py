#!/usr/bin/env python3
"""
Average val_bpb and CE loss over seeds and plot early convergence by lambda.

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
val_re    = re.compile(r"^step:(\d+)/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)")
comps_re  = re.compile(r"step:(\d+) lambda:[\d.]+ ce:([\d.]+)")
lambda_re = re.compile(r"embed_loss_lambda:([\d.]+)")


def parse_log(path: Path) -> tuple[float, dict[int, float], dict[int, float]]:
    lam = 0.0
    ce: dict[int, float] = {}
    val: dict[int, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = lambda_re.search(line)
            if m:
                lam = float(m.group(1))
                continue
            m = val_re.match(line)
            if m:
                val[int(m.group(1))] = float(m.group(2))
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
    return lam, ce, val


def avg_over_seeds(runs: list[dict]) -> tuple[list[int], list[float]]:
    all_steps = sorted(set(s for r in runs for s in r))
    steps_out, avg_out = [], []
    for s in all_steps:
        vals = [r[s] for r in runs if s in r]
        if vals:
            steps_out.append(s)
            avg_out.append(np.mean(vals))
    return steps_out, avg_out


def main(paths: list[Path]) -> None:
    groups: dict[float, list[tuple]] = defaultdict(list)
    for p in paths:
        lam, ce, val = parse_log(p)
        groups[lam].append((ce, val))

    colors = plt.cm.tab10.colors
    fig, (ax_val, ax_ce) = plt.subplots(1, 2, figsize=(13, 5),
                                         gridspec_kw={"wspace": 0.3})
    fig.suptitle("Early convergence (200 steps, tied, avg over 2 seeds)",
                 fontsize=13, fontweight="bold")

    for i, lam in enumerate(sorted(groups)):
        runs = groups[lam]
        n = len(runs)
        color = colors[i]
        label = f"λ={lam} (n={n})" if lam > 0 else f"baseline (n={n})"

        ce_runs  = [r[0] for r in runs]
        val_runs = [r[1] for r in runs]

        # val_bpb
        val_steps, val_avg = avg_over_seeds(val_runs)
        ax_val.plot(val_steps, val_avg, label=label, color=color, linewidth=2.5, marker="o", markersize=5)
        for vr in val_runs:
            s = sorted(vr); ax_val.plot(s, [vr[x] for x in s], color=color, linewidth=0.5, alpha=0.35)

        # train CE
        ce_steps, ce_avg = avg_over_seeds(ce_runs)
        ax_ce.plot(ce_steps, ce_avg, label=label, color=color, linewidth=2)
        for cr in ce_runs:
            s = sorted(cr); ax_ce.plot(s, [cr[x] for x in s], color=color, linewidth=0.5, alpha=0.35)

    for ax, ylabel, title in [
        (ax_val, "val_bpb",  "val_bpb vs step"),
        (ax_ce,  "CE loss",  "Train CE loss vs step"),
    ]:
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
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
