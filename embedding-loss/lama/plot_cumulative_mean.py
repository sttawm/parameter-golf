#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("lama_results_cosine_multi.csv")
finals = df[df["epoch"] == "final"].copy()
finals["test_acc"] = pd.to_numeric(finals["test_acc"])
finals["lambda"] = finals["lambda"].astype(str)

fig, ax = plt.subplots(figsize=(7, 5))

colors = {"1.0": "#a5d6a7", "0.0": "#f8d7da"}
labels = {"1.0": "Cosine embed (λ=1.0)", "0.0": "CE-only (λ=0.0)"}

for lam in ["0.0", "1.0"]:
    vals = finals[finals["lambda"] == lam].sort_values("seed")["test_acc"].values
    n = len(vals)
    cum_mean = np.cumsum(vals) / np.arange(1, n + 1)
    ax.plot(np.arange(1, n + 1), cum_mean * 100,
            color=colors[lam], lw=2, marker="o", ms=5,
            label=f"{labels[lam]}  (final mean: {vals.mean()*100:.2f}%)")

ax.set_xlabel("Number of seeds in average")
ax.set_ylabel("Cumulative mean test accuracy (%)")
ax.set_title("BERT-base-uncased · T-REx (LAMA)\nCumulative mean test accuracy vs seeds", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
out = "lama_cumulative_mean.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
