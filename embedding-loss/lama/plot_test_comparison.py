#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LAMA_DIR = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama"

# ── Load cosine sweep (CE baseline + CE+Emb λ=1) ─────────────────────────────
df_cos = pd.read_csv(f"{LAMA_DIR}/lama_results_cosine_multi.csv")
df_cos["lambda"] = pd.to_numeric(df_cos["lambda"], errors="coerce")
df_cos["test_acc"] = pd.to_numeric(df_cos["test_acc"], errors="coerce")

# Zero-shot: epoch == 0 rows (pre fine-tuning eval)
zs_rows = df_cos[pd.to_numeric(df_cos["epoch"], errors="coerce") == 0].dropna(subset=["test_acc"])
zs_v = zs_rows["test_acc"].values

finals = df_cos[df_cos["epoch"] == "final"].dropna(subset=["test_acc", "lambda"])
ce   = finals[finals["lambda"] == 0.0]["test_acc"].values
cos1 = finals[finals["lambda"] == 1.0]["test_acc"].values

# ── Load EU (embedding + uniformity, no CE) ───────────────────────────────────
try:
    df_eu = pd.read_csv(f"{LAMA_DIR}/lama_results_eu.csv")
    eu_finals = df_eu[df_eu["epoch"] == "final"].dropna(subset=["test_acc"])
    eu = eu_finals["test_acc"].values
except FileNotFoundError:
    eu = np.array([])

# ── Order: Zero-shot, CE only, CE + Emb, Emb only ────────────────────────────
groups = [zs_v, ce, cos1, eu]
labels = ["Zero-shot\n(no fine-tuning)", "CE only", "CE + Emb", "Emb only*"]
colors = ["#e0e0e0", "#f8d7da", "#a5d6a7", "#90caf9"]

fig, ax = plt.subplots(figsize=(8, 5.4))
fig.patch.set_facecolor("white")

for i, (grp, lbl, col) in enumerate(zip(groups, labels, colors)):
    if len(grp) == 0:
        continue
    m  = grp.mean()
    se = grp.std(ddof=1) / np.sqrt(len(grp)) if len(grp) > 1 else 0
    ax.bar(i, m, yerr=se, capsize=5, color=col,
           edgecolor="#888", linewidth=0.5, width=0.55,
           error_kw=dict(elinewidth=1.5, ecolor="#333"))
    ax.text(i, m + se + 0.001,
            f"{m:.1%}\n(n={len(grp)})",
            ha="center", va="bottom", fontsize=9, color="#333")

ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_ylabel("Test accuracy (LAMA T-REx)", fontsize=11)
ax.set_title("BERT-base-uncased  ·  T-REx factual recall", fontsize=12, fontweight="bold", pad=10)
ax.set_ylim(0.44, 0.73)
ax.grid(axis="y", alpha=0.2)

# Key: CE and Emb definitions
ax.text(0.98, 0.98,
        "CE = Cross-Entropy     Emb = Embedding-Similarity",
        transform=ax.transAxes, fontsize=9, va="top", ha="right",
        color="#444",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9))

# Footnote
fig.text(0.5, -0.04,
         "* Emb only also includes a uniformity loss term to prevent embedding collapse\n"
         "  (without it, all embeddings converge to a single point)",
         ha="center", va="top", fontsize=8, color="#777", style="italic")

plt.tight_layout()
out = f"{LAMA_DIR}/test_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
