#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

N_BINS = 25
DATA = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/gpt2_embed_corr_data.pkl"
OUT  = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/gpt2_embed_corr_preview.png"

with open(DATA, "rb") as f:
    d = pickle.load(f)

combos = [
    ("lpr", "cos", "log p_i − log p_top  (log-prob ratio)", "Cosine distance to top token"),
    ("lpr", "l2",  "log p_i − log p_top  (log-prob ratio)", "L2 distance to top token"),
    ("pr",  "cos", "p_i / p_top  (probability ratio)",      "Cosine distance to top token"),
    ("pr",  "l2",  "p_i / p_top  (probability ratio)",      "L2 distance to top token"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.patch.set_facecolor("white")
fig.suptitle("GPT-2-medium  ·  top-100 tokens  ·  500 FineWeb contexts  ·  6.1M pairs",
             fontsize=12, y=1.01)

for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flat, combos):
    x = d[xcol]; y = d[ycol]

    hb = ax.hexbin(x, y, gridsize=60, cmap="Blues", bins="log",
                   mincnt=1, linewidths=0.15, alpha=0.9)
    plt.colorbar(hb, ax=ax, label="pair count (log₁₀)", shrink=0.85)

    edges = np.percentile(x, np.linspace(0, 100, N_BINS + 1))
    edges = np.unique(edges)
    ctrs, mus, sigs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < 30: continue
        ctrs.append((lo + hi) / 2); mus.append(y[m].mean()); sigs.append(y[m].std())
    ctrs = np.array(ctrs); mus = np.array(mus); sigs = np.array(sigs)
    ax.plot(ctrs, mus, color="crimson", lw=2.0, zorder=3)
    ax.fill_between(ctrs, mus - sigs, mus + sigs, color="crimson", alpha=0.20, zorder=2)

    rp, _ = stats.pearsonr(x, y); rs, _ = stats.spearmanr(x, y)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.97, 0.97, f"Pearson r = {rp:.3f}\nSpearman ρ = {rs:.3f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
