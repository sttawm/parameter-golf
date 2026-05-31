#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/sttawm/dev/parameter-golf/embedding-loss/lama/lama_results_cosine_multi.csv")

finals = df[(df["epoch"] == "final") & df["lambda"].astype(str).isin(["0.0", "1.0"])].copy()
finals["step"]     = pd.to_numeric(finals["step"])
finals["test_acc"] = pd.to_numeric(finals["test_acc"])
finals["lambda"]   = finals["lambda"].astype(str)

colors = {"1.0": "#a5d6a7", "0.0": "#f8d7da"}
labels = {"1.0": "Cosine embed (λ=1.0)", "0.0": "CE-only (λ=0.0)"}

fig, ax = plt.subplots(figsize=(8, 5))

for lam in ["0.0", "1.0"]:
    sub = finals[finals["lambda"] == lam].sort_values("seed")
    ax.scatter(sub["step"], sub["test_acc"],
               color=colors[lam], edgecolors="grey", linewidths=0.8,
               s=70, label=labels[lam], zorder=3)
    mean_step = sub["step"].mean()
    std_step  = sub["step"].std()
    ax.axvline(mean_step, color=colors[lam], lw=1.5, ls="--", alpha=0.8, zorder=2)
    ax.axvspan(mean_step - std_step, mean_step + std_step,
               color=colors[lam], alpha=0.18, zorder=1)

import math

a = finals[finals["lambda"] == "1.0"]["step"].values
b = finals[finals["lambda"] == "0.0"]["step"].values
na, nb = len(a), len(b)

# Welch t-test for means
va, vb = a.var(ddof=1), b.var(ddof=1)
t = (a.mean() - b.mean()) / math.sqrt(va/na + vb/nb)
p_mean = 2*(1 - 0.5*(1 + math.erf(abs(t)/math.sqrt(2))))

# F-test for variances (log-normal approximation)
F = vb / va
se_lnF = math.sqrt(2/(nb-1) + 2/(na-1))
z = math.log(F) / se_lnF
p_var = 2*(1 - 0.5*(1 + math.erf(abs(z)/math.sqrt(2))))

stats_text = (
    f"Means:    {b.mean():.0f} vs {a.mean():.0f} steps  (p={p_mean:.3f})\n"
    f"Std devs: {b.std():.0f} vs {a.std():.0f} steps  (p={p_var:.3f})"
)
ax.text(0.97, 0.05, stats_text, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgrey", alpha=0.9),
        family="monospace")

ax.set_xlabel("Step of best val accuracy")
ax.set_ylabel("Test accuracy")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
n = finals["lambda"].value_counts().max()
ax.set_title(f"Step of best val checkpoint vs test accuracy  ·  {n} seeds", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

plt.tight_layout()
out = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/best_step_scatter.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

print("\nBest step stats:")
for lam in ["0.0", "1.0"]:
    sub = finals[finals["lambda"] == lam]
    print(f"  λ={lam}: mean={sub['step'].mean():.0f}  std={sub['step'].std():.0f}  "
          f"min={sub['step'].min():.0f}  max={sub['step'].max():.0f}")
