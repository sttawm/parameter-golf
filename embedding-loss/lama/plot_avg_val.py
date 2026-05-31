#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_csv("/Users/sttawm/dev/parameter-golf/embedding-loss/lama/lama_results_cosine_multi.csv")

train_rows = df[(df["epoch"] != "final") & df["lambda"].astype(str).isin(["0.0", "1.0"])].copy()
train_rows["val_acc"] = pd.to_numeric(train_rows["val_acc"], errors="coerce")
train_rows["lambda"]  = train_rows["lambda"].astype(str)
train_rows = train_rows.dropna(subset=["val_acc"])

# Mean val accuracy across all eval steps, per (seed, lambda)
avg_val = train_rows.groupby(["seed", "lambda"])["val_acc"].mean().reset_index()
avg_val.columns = ["seed", "lambda", "avg_val_acc"]

a = avg_val[avg_val["lambda"] == "1.0"]["avg_val_acc"].values
b = avg_val[avg_val["lambda"] == "0.0"]["avg_val_acc"].values

# Welch t-test
na, nb = len(a), len(b)
ma, mb = a.mean(), b.mean()
va, vb = a.var(ddof=1), b.var(ddof=1)
se  = math.sqrt(va/na + vb/nb)
t   = (ma - mb) / se
p   = 2 * (1 - 0.5*(1 + math.erf(abs(t)/math.sqrt(2))))

print(f"lambda=0.0: mean={mb:.4f}  std={b.std():.4f}  n={nb}")
print(f"lambda=1.0: mean={ma:.4f}  std={a.std():.4f}  n={na}")
print(f"Welch t={t:.3f}  p={p:.3f}")

fig, ax = plt.subplots(figsize=(5, 5))
groups = [b, a]
labels = ["CE-only\n(λ=0.0)", "Cosine embed\n(λ=1.0)"]
colors = ["#f8d7da", "#a5d6a7"]

for i, (grp, lbl, col) in enumerate(zip(groups, labels, colors)):
    m, s = grp.mean(), grp.std()
    ax.bar(i, m, yerr=s, capsize=6, color=col,
           edgecolor="grey", linewidth=0.5, width=0.5,
           error_kw=dict(elinewidth=1.5, ecolor="black"))
    ax.text(i, m + s + 0.001, f"{m:.2%}\n±{s:.2%}",
            ha="center", va="bottom", fontsize=9)

ax.set_xticks(range(2)); ax.set_xticklabels(labels)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
ax.set_ylabel("Mean val accuracy (all checkpoints)")
ax.set_title(
    f"BERT-base-uncased · T-REx (LAMA)  ·  {na} seeds\n"
    f"Average val acc across all eval steps  ·  t={t:.2f}, p={p:.3f}",
    fontsize=10)
ax.grid(axis="y", alpha=0.3)

# set ylim based on data
lo = min(b.mean(), a.mean()) - b.std() - 0.01
hi = max(b.mean(), a.mean()) + a.std() + 0.015
ax.set_ylim(lo, hi)

plt.tight_layout()
out = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/avg_val_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
