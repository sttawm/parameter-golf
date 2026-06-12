#!/usr/bin/env python3
"""
Single-panel version of the embedding-probability distance correlation chart.
X-axis: log(log p_top - log p_i)  =  log(-log_prob_ratio)
Y-axis: cosine distance to top token
Trims the flat region (tokens very close to top, i.e. small -log_prob_ratio).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def pearsonr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum()))

def spearmanr(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pearsonr(ra, rb)

CSV  = "/Users/sttawm/dev/parameter-golf/embedding_prob_data.csv"
OUT  = "/Users/sttawm/dev/parameter-golf/gpt-2-loglog_embed_corr.png"
N_BINS = 40

print("Loading data…")
df = pd.read_csv(CSV, usecols=["log_prob_ratio", "cos_dist"])

# log_prob_ratio = log p_i - log p_top  <=  0
# new x: log(log p_top - log p_i) = log(-log_prob_ratio)
# drop entries where -log_prob_ratio is zero or nearly zero (flat region)
lpr = df["log_prob_ratio"].values
mask = lpr < -1e-6          # remove tokens nearly identical to top (the flat part)
lpr  = lpr[mask]
cos  = df["cos_dist"].values[mask]

x = np.log(-lpr)            # always finite now since -lpr > 1e-6
y = cos

print(f"  {len(x):,} pairs after trimming flat region")

# Pearson / Spearman on log-log axis
rp = pearsonr(x, y)
rs = spearmanr(x, y)
print(f"  Pearson r = {rp:.3f}   Spearman ρ = {rs:.3f}")

# Binned mean ± std
edges = np.percentile(x, np.linspace(0, 100, N_BINS + 1))
edges = np.unique(edges)
ctrs, mus, sigs = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (x >= lo) & (x < hi)
    if m.sum() < 50:
        continue
    ctrs.append((lo + hi) / 2)
    mus.append(y[m].mean())
    sigs.append(y[m].std())

ctrs = np.array(ctrs)
mus  = np.array(mus)
sigs = np.array(sigs)

fig, ax = plt.subplots(figsize=(7, 5))

hb = ax.hexbin(x, y, gridsize=80, cmap="Blues", bins="log",
               mincnt=1, linewidths=0.2, alpha=0.85)
plt.colorbar(hb, ax=ax, label="pair count (log₁₀)")

ax.plot(ctrs, mus, color="crimson", lw=2.0, zorder=3, label="bin mean")
ax.fill_between(ctrs, mus - sigs, mus + sigs,
                color="crimson", alpha=0.22, zorder=2, label="±1 std")

ax.set_xlabel(r"$\log(\log p_{\rm top} - \log p_i)$")
ax.set_ylabel("Cosine distance to top token")
ax.set_title(
    f"GPT-2-medium  ·  Pearson r = {rp:.3f}   Spearman ρ = {rs:.3f}",
    fontsize=11,
)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
