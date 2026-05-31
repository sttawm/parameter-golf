#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

N_BINS  = 25
TOP_K   = 50
OUT_DIR = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama"

d    = pickle.load(open(f"{OUT_DIR}/bert_embed_corr_data.pkl", "rb"))
mask = d["rank"] <= TOP_K
x    = d["lpr"][mask]
y    = d["cos"][mask]

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("white")

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
ax.plot(ctrs, mus, color="crimson", lw=2.0, zorder=3, label="bin mean")
ax.fill_between(ctrs, mus - sigs, mus + sigs,
                color="crimson", alpha=0.20, zorder=2, label="±1 std")
ax.legend(fontsize=9, loc="lower left")

rp, _ = stats.pearsonr(x, y); rs, _ = stats.spearmanr(x, y)
n_pairs = mask.sum()
ax.set_title(f"bert-base-uncased  ·  top-{TOP_K} tokens  ·  {n_pairs:,} pairs",
             fontsize=11, fontweight="bold", pad=8)
ax.set_xlabel("log p_i − log p_top  (log-prob ratio)", fontsize=10)
ax.set_ylabel("Cosine distance to top token", fontsize=10)
ax.tick_params(labelsize=9)
ax.text(0.97, 0.97, f"Pearson r = {rp:.3f}\nSpearman ρ = {rs:.3f}",
        transform=ax.transAxes, fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

plt.tight_layout()
out = f"{OUT_DIR}/bert_lpr_cos_top{TOP_K}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}  (Pearson r={rp:.3f}  Spearman ρ={rs:.3f})")
