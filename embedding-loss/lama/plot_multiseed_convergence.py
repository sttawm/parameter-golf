#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

df = pd.read_csv("/Users/sttawm/dev/parameter-golf/embedding-loss/lama/lama_results_cosine_multi.csv")

# Training rows only (not 'final')
train = df[df["epoch"] != "final"].copy()
train["step"] = pd.to_numeric(train["step"])
train["val_acc"] = pd.to_numeric(train["val_acc"])
train["lambda"] = train["lambda"].astype(str)
train = train.dropna(subset=["step", "val_acc"])

lambdas = ["1.0", "0.0"]
colors  = {"1.0": "#4caf50", "0.0": "#e57373"}

n_seeds = train[train["lambda"].isin(lambdas)].groupby("lambda")["seed"].nunique().min()

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_title(f"BERT-base-uncased · T-REx (LAMA)  ·  Val accuracy vs training step  ·  mean ± std ({n_seeds} seeds)", fontsize=12)

labels = {"1.0": "Cosine embed (λ=1.0)", "0.0": "CE-only (λ=0.0)"}
for lam in lambdas:
    col  = colors[lam]
    rows = train[train["lambda"] == lam]
    grouped = rows.groupby("step")["val_acc"].agg(["mean", "std", "count"])
    grouped = grouped[grouped["count"] >= 2].sort_index()
    if grouped.empty:
        continue

    steps = grouped.index.values
    mean  = grouped["mean"].values
    std   = grouped["std"].fillna(0).values

    ls = "--" if lam == "0.0" else "-"
    ax.plot(steps, mean, color=col, lw=2.2, ls=ls, label=labels[lam], marker="o", markersize=3)
    ax.fill_between(steps, mean - std, mean + std, color=col, alpha=0.15)

ax.set_xlabel("Training step")
ax.set_ylabel("Val accuracy")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.25)

plt.tight_layout()
out = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/multiseed_convergence.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
