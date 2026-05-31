#!/usr/bin/env python3
"""
Train CE vs Val CE over training steps — shows generalization gap.
A smaller train-val gap = less overfitting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/sttawm/dev/parameter-golf/embedding-loss/lama/lama_results_cosine_multi.csv")

train_rows = df[(df["epoch"] != "final") & df["lambda"].astype(str).isin(["0.0", "1.0"])].copy()
train_rows["train_ce"]  = pd.to_numeric(train_rows["train_ce"],  errors="coerce")
train_rows["val_ce"]    = pd.to_numeric(train_rows["val_ce"],    errors="coerce")
train_rows["step"]      = pd.to_numeric(train_rows["step"],      errors="coerce")
train_rows["lambda"]    = train_rows["lambda"].astype(str)
train_rows = train_rows.dropna(subset=["train_ce", "val_ce", "step"])

colors = {"1.0": "#4caf50", "0.0": "#e57373"}
labels = {"1.0": "λ=1.0 (Cosine embed)", "0.0": "λ=0.0 (CE-only)"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("BERT-base-uncased · T-REx (LAMA)  ·  Train vs Val CE  ·  12 seeds", fontsize=12)

# Left: train CE and val CE curves, both lambdas
ax = axes[0]
for lam in ["0.0", "1.0"]:
    sub = train_rows[train_rows["lambda"] == lam]
    grouped = sub.groupby("step")[["train_ce", "val_ce"]].agg(["mean", "std"]).reset_index()
    grouped.columns = ["step", "train_mean", "train_std", "val_mean", "val_std"]
    grouped = grouped[grouped["train_std"].notna()]

    col = colors[lam]
    ax.plot(grouped["step"], grouped["train_mean"], color=col, lw=1.5, ls="--",
            label=f"{labels[lam]} — train")
    ax.fill_between(grouped["step"],
                    grouped["train_mean"] - grouped["train_std"],
                    grouped["train_mean"] + grouped["train_std"],
                    color=col, alpha=0.10)
    ax.plot(grouped["step"], grouped["val_mean"], color=col, lw=2.0, ls="-",
            label=f"{labels[lam]} — val")
    ax.fill_between(grouped["step"],
                    grouped["val_mean"] - grouped["val_std"],
                    grouped["val_mean"] + grouped["val_std"],
                    color=col, alpha=0.15)

ax.set_xlabel("Training step")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("Train (dashed) vs Val (solid)  ·  mean ± std", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.25)

# Right: mean train-val CE gap at each step
ax = axes[1]
for lam in ["0.0", "1.0"]:
    sub = train_rows[train_rows["lambda"] == lam].copy()
    sub["gap"] = sub["val_ce"] - sub["train_ce"]
    grouped = sub.groupby("step")["gap"].agg(["mean", "std"]).reset_index()

    col = colors[lam]
    ax.plot(grouped["step"], grouped["mean"], color=col, lw=2.0, label=labels[lam])
    ax.fill_between(grouped["step"],
                    grouped["mean"] - grouped["std"],
                    grouped["mean"] + grouped["std"],
                    color=col, alpha=0.18)

ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
ax.set_xlabel("Training step")
ax.set_ylabel("Val CE − Train CE  (generalization gap)")
ax.set_title("Generalization gap  ·  smaller = less overfitting", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

# Print summary at final step
print("Generalization gap (val CE - train CE) at last logged step:")
for lam in ["0.0", "1.0"]:
    sub = train_rows[train_rows["lambda"] == lam].copy()
    sub["gap"] = sub["val_ce"] - sub["train_ce"]
    last = sub.groupby("seed")["step"].max().reset_index()
    last = last.merge(sub[["seed", "step", "gap"]], on=["seed", "step"])
    print(f"  λ={lam}: mean gap = {last['gap'].mean():+.4f}  std={last['gap'].std():.4f}")

plt.tight_layout()
out = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/generalization_gap.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
