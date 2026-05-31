#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/sttawm/dev/parameter-golf/embedding-loss/lama/lama_results_cosine_multi.csv")

# Best val acc per (seed, lambda) from training rows
train = df[(df["epoch"] != "final")].copy()
train["val_acc"] = pd.to_numeric(train["val_acc"], errors="coerce")
train["lambda"]  = train["lambda"].astype(str)
best_val = train.groupby(["seed", "lambda"])["val_acc"].max().reset_index()
best_val.columns = ["seed", "lambda", "best_val_acc"]

# Test acc from final rows
finals = df[df["epoch"] == "final"].copy()
finals["test_acc"] = pd.to_numeric(finals["test_acc"], errors="coerce")
finals["lambda"]   = finals["lambda"].astype(str)

merged = finals[["seed", "lambda", "test_acc"]].merge(best_val, on=["seed", "lambda"])

# Only compare 0.0 and 1.0
merged = merged[merged["lambda"].isin(["0.0", "1.0"])]

colors = {"1.0": "#a5d6a7", "0.0": "#f8d7da"}
labels = {"1.0": "Cosine embed (λ=1.0)", "0.0": "CE-only (λ=0.0)"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Best val accuracy vs test accuracy per seed  ·  12 seeds", fontsize=12)

# Left: scatter val vs test
ax = axes[0]
for lam in ["0.0", "1.0"]:
    sub = merged[merged["lambda"] == lam]
    ax.scatter(sub["best_val_acc"], sub["test_acc"],
               color=colors[lam], edgecolors="grey", linewidths=0.8,
               s=60, label=labels[lam], zorder=3)
    # mean point
    ax.scatter(sub["best_val_acc"].mean(), sub["test_acc"].mean(),
               color=colors[lam], edgecolors="black", linewidths=1.5,
               s=150, marker="D", zorder=4)

lims = [merged[["best_val_acc", "test_acc"]].min().min() - 0.002,
        merged[["best_val_acc", "test_acc"]].max().max() + 0.002]
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4, label="val = test")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Best val accuracy"); ax.set_ylabel("Test accuracy")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
ax.set_title("Val vs test per seed  (diamonds = mean)", fontsize=10)

# Right: grouped bar - val and test side by side
ax = axes[1]
x = np.array([0, 1])
width = 0.3
for i, lam in enumerate(["0.0", "1.0"]):
    sub = merged[merged["lambda"] == lam]
    vm, vs = sub["best_val_acc"].mean(), sub["best_val_acc"].std()
    tm, ts = sub["test_acc"].mean(), sub["test_acc"].std()
    offset = (i - 0.5) * width
    ax.bar(x[0] + offset, vm, width, yerr=vs, capsize=4,
           color=colors[lam], edgecolor="grey", linewidth=0.5,
           error_kw=dict(elinewidth=1.2))
    ax.bar(x[1] + offset, tm, width, yerr=ts, capsize=4,
           color=colors[lam], edgecolor="grey", linewidth=0.5,
           error_kw=dict(elinewidth=1.2), label=labels[lam])
    # annotate
    for xpos, m, s in [(x[0]+offset, vm, vs), (x[1]+offset, tm, ts)]:
        ax.text(xpos, m + s + 0.001, f"{m:.2%}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x); ax.set_xticklabels(["Best val accuracy", "Test accuracy"])
ax.set_ylim(0.64, 0.70)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
ax.set_title("Mean ± std", fontsize=10)

plt.tight_layout()
out = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/val_vs_test.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# Print val→test gaps
print("\nVal → test gap (val - test, lower = less overfitting):")
for lam in ["0.0", "1.0"]:
    sub = merged[merged["lambda"] == lam]
    gap = (sub["best_val_acc"] - sub["test_acc"]).mean()
    print(f"  λ={lam}: mean gap = {gap:+.4f} ({gap:.2%})")
