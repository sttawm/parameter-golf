#!/usr/bin/env python3
"""
Plot CE and uniformity loss terms for the uniform sweep.
Pass uniform sweep logs and optionally baseline logs (gamma=0).

Usage:
    python plot_uniform.py logs/unif_*.txt [--baseline logs/convu_*_lam0_*.txt]
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

train_re  = re.compile(r"step:(\d+)/\d+ train_loss:([\d.]+)")
val_re    = re.compile(r"^step:(\d+)/\d+ val_loss:([\d.]+)")
unif_re   = re.compile(r"step:(\d+) gamma:([\d.]+) uniform:([-\d.]+) emb_norm:([\d.]+)")
gamma_re  = re.compile(r"uniform_loss_gamma:([\d.]+)")


def parse_log(path: Path):
    gamma = 0.0
    total, unif, norm, val = {}, {}, {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = gamma_re.search(line)
            if m:
                gamma = float(m.group(1))
                continue
            m = train_re.search(line)
            if m:
                total[int(m.group(1))] = float(m.group(2))
                continue
            m = val_re.match(line)
            if m:
                val[int(m.group(1))] = float(m.group(2))
                continue
            m = unif_re.search(line)
            if m:
                step = int(m.group(1))
                unif[step] = float(m.group(3))
                norm[step] = float(m.group(4))
    # Back out CE from total: total = CE + gamma*uniform
    ce = {}
    for s in total:
        if s in unif:
            ce[s] = total[s] - gamma * unif[s]
        else:
            ce[s] = total[s]  # gamma=0 baseline
    return gamma, ce, unif, norm, val


def avg(runs):
    all_steps = sorted(set(s for r in runs for s in r))
    steps_out, vals_out = [], []
    for s in all_steps:
        vs = [r[s] for r in runs if s in r]
        if vs:
            steps_out.append(s)
            vals_out.append(np.mean(vs))
    return steps_out, vals_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--baseline", nargs="*", type=Path, default=[])
    args = parser.parse_args()

    # Group by gamma
    groups: dict[float, list] = defaultdict(list)
    for p in args.logs:
        gamma, ce, unif, norm, val = parse_log(p)
        groups[gamma].append((ce, unif, norm, val))
    for p in (args.baseline or []):
        _, ce, unif, norm, val = parse_log(p)
        groups[0.0].append((ce, unif, norm, val))

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Uniformity loss sweep — UNTIED weights (200 steps, avg 2 seeds)",
                 fontsize=12, fontweight="bold")

    for i, gamma in enumerate(sorted(groups)):
        runs = groups[gamma]
        color = colors[i]
        n = len(runs)
        label = f"baseline (n={n})" if gamma == 0.0 else f"γ={gamma} (n={n})"

        ce_runs  = [r[0] for r in runs]
        val_runs = [r[3] for r in runs]
        unif_runs = [r[1] for r in runs]
        norm_runs = [r[2] for r in runs]

        # Val CE
        vs, va = avg(val_runs)
        axes[0].plot(vs, va, label=label, color=color, linewidth=2, marker="o", markersize=4)
        for vr in val_runs:
            s = sorted(vr)
            axes[0].plot(s, [vr[x] for x in s], color=color, linewidth=0.5, alpha=0.35)

        # Scaled uniform term (γ · uniform) — only for non-baseline
        if gamma > 0.0:
            scaled = [{s: gamma * v for s, v in r.items()} for r in unif_runs]
            us, ua = avg(scaled)
            axes[1].plot(us, ua, label=label, color=color, linewidth=2)
            for ur in scaled:
                s = sorted(ur)
                axes[1].plot(s, [ur[x] for x in s], color=color, linewidth=0.5, alpha=0.35)

        # Embedding norm
        ns, na = avg(norm_runs)
        if ns:
            axes[2].plot(ns, na, label=label, color=color, linewidth=2)

    axes[0].set_title("Val CE loss")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("CE loss (nats)")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Scaled uniform term (γ · L_uniform)")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("γ · log-mean-exp (nats)")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Embedding norm")
    axes[2].set_xlabel("Step"); axes[2].set_ylabel("Mean L2 norm")
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path("uniform_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
