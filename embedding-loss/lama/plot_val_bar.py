#!/usr/bin/env python3
"""
Bar chart: mean best-val-accuracy per (lambda, loss_type), with std error bars.

Each bar = mean of per-seed max test_acc. The 'final' row in each CSV already
stores best_acc (the best val checkpoint), so we read directly from those rows.

Works on partial data — bars appear as seeds complete.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

LAMA_DIR = Path(__file__).parent
LAMBDAS  = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0]

# ── Load data ─────────────────────────────────────────────────────────────────
def load(path, label):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = df[df["epoch"] == "final"].copy()
    df["lambda"]    = pd.to_numeric(df["lambda"], errors="coerce")
    df["loss_type"] = label
    return df.dropna(subset=["lambda", "test_acc"])

# Prefer merged (post-collection) files, fall back to originals
cos_path = LAMA_DIR / ("lama_results_cosine_merged.csv"
                       if (LAMA_DIR / "lama_results_cosine_merged.csv").exists()
                       else "lama_results_cosine_multi.csv")
l2_path  = LAMA_DIR / ("lama_results_l2_merged.csv"
                       if (LAMA_DIR / "lama_results_l2_merged.csv").exists()
                       else "lama_results_l2_multi.csv")

cos = load(cos_path, "cosine")
l2  = load(l2_path,  "l2")

# Append high-lambda runs if present
lam20_path = LAMA_DIR / "lama_results_cosine_lam20.csv"
cos = pd.concat([cos, load(lam20_path, "cosine")], ignore_index=True).drop_duplicates()

# λ=0.0 is identical regardless of loss type — pool into a single "baseline" entry
baseline = pd.concat([
    cos[cos["lambda"] == 0.0],
    l2[l2["lambda"]   == 0.0],
], ignore_index=True)
baseline["loss_type"] = "baseline"

cos = cos[cos["lambda"] != 0.0]
l2  = l2[l2["lambda"]  != 0.0]
df  = pd.concat([baseline, cos, l2], ignore_index=True)

if df.empty:
    print("No completed runs yet.")
    exit(0)

# ── Aggregate ─────────────────────────────────────────────────────────────────
stats = (df.groupby(["loss_type", "lambda"])["test_acc"]
           .agg(mean="mean", std="std", n="count")
           .reset_index())
stats["std"] = stats["std"].fillna(0)

print(stats.to_string(index=False))

# ── Planned seed counts per (loss_type, lambda) ───────────────────────────────
PLANNED = {
    ("baseline", 0.0): 28,   # pooled cosine+L2 λ=0, no more running
    ("cosine",   1.0): 25,   # already had 25 seeds pre-sweep
    ("cosine",   0.1): 12,   # 10 default seeds + 2 bonus (pod 8: 7000, 8000)
    ("cosine",   0.5): 12,
    ("cosine",   2.0): 12,
    ("cosine",   4.0): 12,
    ("l2",       0.1): 12,
    ("l2",       0.5): 12,
    ("l2",       1.0): 12,
    ("l2",       2.0): 12,
    ("l2",       4.0): 12,
    ("l2",       8.0): 12,
}

# ── Plot ──────────────────────────────────────────────────────────────────────
# Order: cosine lambdas ascending, then L2 lambdas ascending (baseline first)
def make_label(r):
    if r["loss_type"] == "baseline": return "baseline\n(λ=0)"
    prefix = "cos" if r["loss_type"] == "cosine" else "L2"
    return f"{prefix}  λ={r['lambda']}"
stats["label"] = stats.apply(make_label, axis=1)
# Baseline first, then remaining bars sorted descending by mean performance
baseline_rows = stats[stats["loss_type"] == "baseline"]
other_rows    = stats[stats["loss_type"] != "baseline"].sort_values("mean", ascending=False)
stats = pd.concat([baseline_rows, other_rows], ignore_index=True)

N    = len(stats)
X    = np.arange(N)
BAR_W = 0.6

# Distinct color per bar using a qualitative palette
palette = plt.cm.tab20(np.linspace(0, 1, max(N, 2)))

fig, ax = plt.subplots(figsize=(max(8, N * 0.9), 5))
fig.patch.set_facecolor("white")

for i, row in stats.iterrows():
    bar = ax.bar(i, row["mean"], BAR_W,
                 yerr=row["std"], capsize=4,
                 color=palette[i], alpha=0.88,
                 error_kw={"elinewidth": 1.4, "ecolor": "#444"})
    ax.text(i, row["mean"] + row["std"] + 0.0003,
            f"n={int(row['n'])}", ha="center", va="bottom",
            fontsize=8, color="#333333")

ax.set_xticks(X)
ax.set_xticklabels(stats["label"], fontsize=9.5, rotation=30, ha="right")
ax.set_ylabel("Best val accuracy (mean ± std)", fontsize=11)
ax.set_title("LAMA T-REx: best val accuracy by configuration",
             fontsize=12, fontweight="bold", pad=10)
# Baseline dotted line (λ=0.0, cosine — same as λ=0.0 L2 since no embed loss)
baseline_rows = stats[stats["loss_type"] == "baseline"]
if not baseline_rows.empty:
    baseline_mean = baseline_rows.iloc[0]["mean"]
    ax.axhline(baseline_mean, color="#555", linewidth=1.2, linestyle=":",
               label=f"Baseline mean ({baseline_mean:.4f})", zorder=0)
    ax.legend(fontsize=9)

ax.grid(True, alpha=0.25, axis="y")
ax.tick_params(labelsize=9)

# Tight y-axis around actual data range
valid_means = stats["mean"].dropna()
valid_stds  = stats["std"].fillna(0)
pad = (valid_means.max() - valid_means.min()) * 0.8 + valid_stds.max()
ax.set_ylim(valid_means.min() - pad, valid_means.max() + pad * 1.5)

plt.tight_layout()
out = LAMA_DIR / "val_bar_chart.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
