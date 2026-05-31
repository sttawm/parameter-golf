#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

N_BINS  = 25
OUT_DIR = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama"

panels = [
    ("GPT-2-medium",         "gpt2",        "rank", "l2",  50,  "Rank by probability (1 = most probable)", "L2 distance to top token",      "vocab 50,257"),
    ("parameter-golf (λ=0)", "pg_baseline", "rank", "l2",  50,  "Rank by probability (1 = most probable)", "L2 distance to top token",      "vocab 1,024"),
    ("bert-base-uncased",    "bert",        "rank", "cos", 100, "Rank by probability (1 = most probable)", "Cosine distance to top token",  "vocab 30,522"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("white")

for ax, (title, slug, xcol, ycol, top_k, xlabel, ylabel, vocab) in zip(axes, panels):
    d    = pickle.load(open(f"{OUT_DIR}/{slug}_embed_corr_data.pkl", "rb"))
    mask = d["rank"] <= top_k
    x    = d[xcol][mask]
    y    = d[ycol][mask]

    hb = ax.hexbin(x, y, gridsize=60, cmap="Blues", bins="log",
                   mincnt=1, linewidths=0.15, alpha=0.9)
    if ax is axes[-1]:
        plt.colorbar(hb, ax=ax, label="pair count (log₁₀)", shrink=0.85)

    edges = np.percentile(x, np.linspace(0, 100, N_BINS + 1))
    edges = np.unique(edges)
    ctrs, mus, sigs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < 30: continue
        ctrs.append((lo + hi) / 2); mus.append(y[m].mean()); sigs.append(y[m].std())
    ctrs = np.array(ctrs); mus = np.array(mus); sigs = np.array(sigs)
    ax.plot(ctrs, mus, color="crimson", lw=2.0, zorder=3, label="bin mean")
    ax.fill_between(ctrs, mus - sigs, mus + sigs,
                    color="crimson", alpha=0.20, zorder=2, label="±1 std")
    ax.legend(fontsize=8, loc="lower left")

    rp, _ = stats.pearsonr(x, y); rs, _ = stats.spearmanr(x, y)

    # % difference between mean embedding distance at rank 1 vs rank top_k
    pct_diff = (mus[-1] - mus[0]) / abs(mus[0]) * 100

    ax.set_title(f"{title}  ·  top-{top_k}  ·  {vocab}", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.97, 0.97,
            f"Pearson r = {rp:.3f}\nSpearman ρ = {rs:.3f}\n+{pct_diff:.0f}% rank 1→{top_k}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

plt.tight_layout()
out = f"{OUT_DIR}/best_three_embed_corr.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
