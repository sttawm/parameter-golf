#!/usr/bin/env python3
import re, statistics
import matplotlib.pyplot as plt
import numpy as np

pattern = re.compile(r'lam=([0-9.]+) s=\d+ step=\d+:.*\|\s+(\d+)/852.*?([0-9]+\.[0-9]+)it/s')

speeds = {'0.0': [], '1.0': []}
with open("/Users/sttawm/dev/parameter-golf/embedding-loss/lama/targeted2.log_local", 'r', errors='ignore') as f:
    for line in f:
        m = pattern.search(line)
        if m:
            lam, batch, its = m.group(1), int(m.group(2)), float(m.group(3))
            if batch > 200 and lam in speeds:
                speeds[lam].append(its)

med = {lam: statistics.median(v) for lam, v in speeds.items()}
std = {lam: statistics.stdev(v)  for lam, v in speeds.items()}
rel_diff = (med['1.0'] - med['0.0']) / med['0.0'] * 100

print(f"λ=0.0: median={med['0.0']:.3f} it/s  (n={len(speeds['0.0']):,})")
print(f"λ=1.0: median={med['1.0']:.3f} it/s  (n={len(speeds['1.0']):,})")
print(f"Relative difference: {rel_diff:+.1f}%")

colors = {"0.0": "#f8d7da", "1.0": "#a5d6a7"}
labels = {"0.0": "CE-only\n(λ=0.0)", "1.0": "Cosine embed\n(λ=1.0)"}

fig, ax = plt.subplots(figsize=(5, 4))
for i, lam in enumerate(["0.0", "1.0"]):
    ax.bar(i, med[lam], color=colors[lam], edgecolor="grey", linewidth=0.5,
           width=0.5)
    ax.text(i, med[lam] + 0.01, f"{med[lam]:.3f} it/s",
            ha="center", va="bottom", fontsize=9)

ax.annotate(f"Relative: {rel_diff:+.1f}%",
            xy=(0.5, (med['0.0'] + med['1.0']) / 2),
            ha="center", va="center", fontsize=10,
            color="dimgrey",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey"))

ax.set_xticks([0, 1]); ax.set_xticklabels([labels["0.0"], labels["1.0"]])
ax.set_ylabel("Training speed (it/s, median)")
ax.set_ylim(4.0, 4.35)
ax.set_title("BERT-base-uncased · T-REx (LAMA)\nTraining throughput (steady-state, batch > 200)", fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/speed_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
