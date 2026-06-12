#!/usr/bin/env python3
"""
Per-anchor-token correlation analysis.

For each unique anchor token (top predicted token at a position) that appears
in at least MIN_PAIRS rows, compute the Spearman correlation between
log_prob_ratio and cos_out (embed_out cosine distance).

A more negative correlation means: when this token is the top prediction,
tokens with lower probability tend to be further away in embed_out space
— the geometry is well-ordered by probability.

Outputs
-------
token_correlation_plot.png   — horizontal bar chart, top/bottom tokens
token_correlation_table.csv  — full ranked table
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
CSV_IN    = "embedding_prob_pythia_data.csv"
OUT_PLOT  = "token_correlation_plot.png"
OUT_TABLE = "token_correlation_table.csv"
MIN_PAIRS = 200   # minimum rows per anchor token for a reliable estimate
N_SHOW    = 25    # top and bottom N tokens to plot

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {CSV_IN}…")
df = pd.read_csv(CSV_IN, usecols=["log_prob_ratio", "cos_out", "anchor_token"])
print(f"  {len(df):,} rows, {df['anchor_token'].nunique():,} unique anchor tokens")

# ── Per-token correlation ─────────────────────────────────────────────────────
print("Computing per-anchor-token Spearman correlations…")
results = []
for token, grp in df.groupby("anchor_token"):
    if len(grp) < MIN_PAIRS:
        continue
    r, p = stats.spearmanr(grp["log_prob_ratio"], grp["cos_out"])
    results.append({
        "anchor_token": token,
        "spearman_r":   r,
        "p_value":      p,
        "n_pairs":      len(grp),
    })

tbl = pd.DataFrame(results).sort_values("spearman_r")
tbl.to_csv(OUT_TABLE, index=False)
print(f"  {len(tbl):,} tokens with ≥{MIN_PAIRS} pairs → {OUT_TABLE}")

# ── Heuristic token categories ────────────────────────────────────────────────
def categorize(tok):
    s = tok.strip()
    if not s:
        return "whitespace/empty"
    if re.fullmatch(r'[\W_]+', s):
        return "punctuation/symbol"
    if re.fullmatch(r'[\d.,%-]+', s):
        return "number"
    if tok.startswith(" "):
        return "word (space-prefixed)"
    return "subword (no space)"

tbl["category"] = tbl["anchor_token"].apply(categorize)

CAT_COLORS = {
    "word (space-prefixed)":  "#4e79a7",
    "subword (no space)":     "#f28e2b",
    "punctuation/symbol":     "#59a14f",
    "number":                 "#e15759",
    "whitespace/empty":       "#bab0ac",
}

# ── Plot ──────────────────────────────────────────────────────────────────────
top    = tbl.head(N_SHOW)        # most negative r  (strongest correlation)
bottom = tbl.tail(N_SHOW)        # least negative r (weakest correlation)
both   = pd.concat([top, bottom])

fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=False)
fig.suptitle(
    f"Per-anchor-token Spearman r  (log_prob_ratio vs embed_out cosine distance)\n"
    f"Pythia-410M · {len(tbl):,} tokens with ≥{MIN_PAIRS} pairs",
    fontsize=12,
)

for ax, subset, title in [
    (axes[0], top,    f"Top {N_SHOW} — strongest correlation\n(more negative = geometry tracks probability)"),
    (axes[1], bottom, f"Bottom {N_SHOW} — weakest correlation\n(near zero = geometry unrelated to probability)"),
]:
    colors = [CAT_COLORS.get(c, "#bab0ac") for c in subset["category"]]
    bars = ax.barh(
        y=range(len(subset)),
        width=subset["spearman_r"].values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(
        [repr(t) for t in subset["anchor_token"]],
        fontsize=9, fontfamily="monospace",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Spearman r")
    ax.set_title(title, fontsize=10)

    # annotate n_pairs
    for i, (_, row) in enumerate(subset.iterrows()):
        ax.text(
            row["spearman_r"] + (0.005 if row["spearman_r"] >= 0 else -0.005),
            i, f"n={row['n_pairs']:,}",
            va="center",
            ha="left" if row["spearman_r"] >= 0 else "right",
            fontsize=7, color="#555555",
        )

# shared legend
legend_patches = [
    mpatches.Patch(color=c, label=cat)
    for cat, c in CAT_COLORS.items()
    if cat in tbl["category"].values
]
fig.legend(handles=legend_patches, title="Token category",
           loc="lower center", ncol=len(legend_patches),
           fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"Saved plot → {OUT_PLOT}")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n── Correlation by category (median Spearman r) ──")
print(
    tbl.groupby("category")["spearman_r"]
    .agg(["median", "mean", "count"])
    .sort_values("median")
    .to_string()
)

print(f"\n── Top {N_SHOW} tokens (strongest correlation) ──")
print(top[["anchor_token", "spearman_r", "n_pairs", "category"]].to_string(index=False))

print(f"\n── Bottom {N_SHOW} tokens (weakest correlation) ──")
print(bottom[["anchor_token", "spearman_r", "n_pairs", "category"]].to_string(index=False))

plt.show()
