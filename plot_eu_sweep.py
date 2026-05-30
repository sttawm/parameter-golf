#!/usr/bin/env python3
"""Reproduce 'Train CE (embed runs not optimizing CE)' chart.

Plots train CE for embed+uniform runs alongside the collapsed embed-only run
and the CE-only untied baseline, to show embedding loss acts as a weak surrogate.
"""

import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = Path("logs")

train_re = re.compile(r"^step:(\d+)/\d+ train_loss:([\d.]+)")
ce_re    = re.compile(r"^step:(\d+) lambda:[\d.]+ ce:([\d.]+)")
val_re   = re.compile(r"^step:(\d+)/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)")
gamma_re = re.compile(r"^uniform_loss_gamma:([\d.]+)")


def parse_eu(path: Path):
    gamma, ce, val = 0.0, {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = gamma_re.match(line)
            if m:
                gamma = float(m.group(1)); continue
            m = ce_re.match(line)
            if m:
                ce[int(m.group(1))] = float(m.group(2)); continue
            m = val_re.match(line)
            if m:
                val[int(m.group(1))] = float(m.group(2))
    return gamma, ce, val


def parse_ce_baseline(path: Path):
    ce = {}
    with open(path) as f:
        for line in f:
            m = train_re.match(line.strip())
            if m:
                ce[int(m.group(1))] = float(m.group(2))
    return ce


def avg_series(runs: list[dict]) -> tuple[list, list]:
    all_steps = sorted(set(s for r in runs for s in r))
    steps_out, vals_out = [], []
    for s in all_steps:
        vs = [r[s] for r in runs if s in r]
        if vs:
            steps_out.append(s); vals_out.append(np.mean(vs))
    return steps_out, vals_out


# ── Load eu logs ─────────────────────────────────────────────────────────────
eu_groups: dict[float, list[dict]] = defaultdict(list)
eu_val: dict[float, list[float]]   = defaultdict(list)

for p in sorted(LOG_DIR.glob("eu_*.txt")):
    gamma, ce, val = parse_eu(p)
    if gamma > 0 and ce:
        eu_groups[gamma].append(ce)
        if val:
            eu_val[gamma].append(val[max(val.keys())])  # final val_bpb

# ── Load embed-only (collapsed) logs ─────────────────────────────────────────
eo_runs = []
for p in LOG_DIR.glob("eo_*_untied.txt"):
    _, ce, _ = parse_eu(p)          # same format: ce: field
    if ce:
        eo_runs.append(ce)

# ── Load CE-only untied baseline ─────────────────────────────────────────────
base_runs = []
for p in LOG_DIR.glob("convu_*_lam0_*.txt"):
    ce = parse_ce_baseline(p)
    if ce:
        base_runs.append(ce)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("white")

# embed-only (collapsed) — dashed gray; label over flat region ~step 60
if eo_runs:
    s, v = avg_series(eo_runs)
    ax.plot(s, v, color="gray", linestyle="--", linewidth=1.5)
    lx = 140
    ax.text(lx, np.interp(lx, s, v) + 0.45, "Loss = Embedding-Similarity",
            color="gray", fontsize=8.5, va="bottom", ha="center")

# CE-only baseline — black; label near the right end
if base_runs:
    s, v = avg_series(base_runs)
    ax.plot(s, v, color="black", linewidth=2.0)
    lx = 140
    ax.text(lx, np.interp(lx, s, v) + 0.15, "Loss = CE",
            color="black", fontsize=8.5, va="bottom", ha="center")

# embed+uniform — green; label centered over mid stretch
s, v = avg_series(eu_groups[8.0])
ax.plot(s, v, color="tab:green", linewidth=1.8)
lx = 140
ax.text(lx, np.interp(lx, s, v) + 0.30, "Loss = Embedding-Similarity + Embedding-Uniformity-Bias",
        color="tab:green", fontsize=8.5, va="bottom", ha="center")

ax.set_xlim(0, 200)
ax.set_title("Embedding loss as a weak CE surrogate", fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("step", fontsize=10)
ax.set_ylabel("CE / train loss (nats)", fontsize=10)
ax.grid(True, alpha=0.25)
ax.tick_params(labelsize=9)

plt.tight_layout()
out = Path("eu_sweep_plot_regen.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
