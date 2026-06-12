#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LAMA_DIR = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama"
df = pd.read_csv(f"{LAMA_DIR}/topk_results.csv")

CE_COLOR  = "#f8d7da"
EMB_COLOR = "#a5d6a7"
CE_LINE   = "#c9606e"
EMB_LINE  = "#4d9a5f"

TOPKS = [1, 3, 5, 10, 20]

ce  = df[df["lambda"] == 0.0]
emb = df[df["lambda"] == 1.0]

# ── Figure 1: Top-k accuracy ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.5))
fig.patch.set_facecolor("white")

for sub, lc, fc, label in [
    (ce,  CE_LINE,  CE_COLOR,  f"CE only  (n={len(ce)})"),
    (emb, EMB_LINE, EMB_COLOR, f"CE + Emb  (n={len(emb)})"),
]:
    means = [sub[f"top{k}"].mean() for k in TOPKS]
    ses   = [sub[f"top{k}"].std(ddof=1) / np.sqrt(len(sub)) for k in TOPKS]
    ax.plot(TOPKS, means, color=lc, lw=2, marker="o", markersize=6, label=label)
    ax.fill_between(TOPKS,
                    [m - s for m, s in zip(means, ses)],
                    [m + s for m, s in zip(means, ses)],
                    color=fc, alpha=0.4)

ax.set_xlabel("k", fontsize=11)
ax.set_ylabel("Top-k accuracy (test set)", fontsize=11)
ax.set_title("Top-k factual recall · BERT-base · T-REx", fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xticks(TOPKS)
ax.legend(fontsize=9.5)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{LAMA_DIR}/topk_plot.png", dpi=150, bbox_inches="tight")
print("Saved → topk_plot.png")

# ── Figure 2: Train / test / gap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.5))
fig.patch.set_facecolor("white")

labels   = ["CE only", "CE + Emb"]
x        = np.array([0.0, 0.6])
bar_w    = 0.18

for i, (sub, fc, lc) in enumerate([(ce, CE_COLOR, CE_LINE), (emb, EMB_COLOR, EMB_LINE)]):
    xi = x[i]
    for j, (col, label, offset) in enumerate([
        ("train_acc", "Train",  -bar_w),
        ("top1",      "Test",    bar_w),
    ]):
        m  = sub[col].mean()
        se = sub[col].std(ddof=1) / np.sqrt(len(sub))
        bar = ax.bar(xi + offset, m, width=bar_w*1.8, color=fc, edgecolor=lc,
                     linewidth=1.0, yerr=se, capsize=4,
                     error_kw=dict(elinewidth=1.3, ecolor="#555"))
        ax.text(xi + offset, m + se + 0.005, f"{m:.1%}",
                ha="center", va="bottom", fontsize=8.5, color="#333")

    # Gap annotation
    gap_m  = (sub["train_acc"] - sub["top1"]).mean()
    gap_se = (sub["train_acc"] - sub["top1"]).std(ddof=1) / np.sqrt(len(sub))
    ax.annotate("", xy=(xi + bar_w, sub["top1"].mean()),
                    xytext=(xi + bar_w, sub["train_acc"].mean()),
                arrowprops=dict(arrowstyle="<->", color=lc, lw=1.5))
    ax.text(xi + bar_w + 0.06, (sub["train_acc"].mean() + sub["top1"].mean()) / 2,
            f"gap\n{gap_m:.1%}", ha="left", va="center", fontsize=8, color=lc)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Train vs test accuracy · BERT-base · T-REx", fontsize=11, fontweight="bold")
ax.set_ylim(0.55, 1.02)
ax.grid(axis="y", alpha=0.2)

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=CE_COLOR, ec=CE_LINE, label="Train acc"),
                   Patch(fc=CE_COLOR, ec=CE_LINE, alpha=0.4, label="Test acc")],
          labels=["Train acc", "Test acc"], fontsize=9)

plt.tight_layout()
plt.savefig(f"{LAMA_DIR}/train_test_gap_plot.png", dpi=150, bbox_inches="tight")
print("Saved → train_test_gap_plot.png")
