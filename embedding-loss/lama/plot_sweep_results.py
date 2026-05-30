#!/usr/bin/env python3
"""
Analyze and plot the lambda sweep results (cosine + L2).

Produces:
  sweep_results.png  — test accuracy by lambda, cosine vs L2, mean ± std
  sweep_curves.png   — val accuracy training curves averaged across seeds
  sweep_summary.csv  — mean/std/n per (lambda, loss_type)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

LAMA_DIR = Path(__file__).parent
COS_CSV  = LAMA_DIR / "lama_results_cosine_merged.csv"
L2_CSV   = LAMA_DIR / "lama_results_l2_merged.csv"

# Fall back to unmerged files if merged not yet available
if not COS_CSV.exists():
    COS_CSV = LAMA_DIR / "lama_results_cosine_multi.csv"
if not L2_CSV.exists():
    L2_CSV = LAMA_DIR / "lama_results_l2_multi.csv"

LAMBDAS  = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0]
COLORS   = {"cosine": "#4C72B0", "l2": "#DD8452"}


def load_finals(path, label):
    if not Path(path).exists():
        print(f"Missing: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["epoch"] == "final"].copy()
    df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")
    df = df.dropna(subset=["lambda", "test_acc"])
    df["loss_type"] = label
    return df


cos = load_finals(COS_CSV, "cosine")
l2  = load_finals(L2_CSV,  "l2")
all_df = pd.concat([cos, l2], ignore_index=True)

if all_df.empty:
    print("No data found. Run collect_results.sh first.")
    exit(1)

# ── Summary table ─────────────────────────────────────────────────────────────
summary = (all_df.groupby(["lambda", "loss_type"])["test_acc"]
           .agg(["mean", "std", "count"])
           .reset_index())
summary.columns = ["lambda", "loss_type", "mean", "std", "n"]
summary["se"] = summary["std"] / np.sqrt(summary["n"])
summary.to_csv(LAMA_DIR / "sweep_summary.csv", index=False)
print(summary.to_string(index=False))

# ── Plot 1: test accuracy bar/scatter by lambda ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")

x_base = np.array(LAMBDAS)
offset = {"cosine": -0.04, "l2": 0.04}

for loss_type, color in COLORS.items():
    sub = summary[summary["loss_type"] == loss_type].set_index("lambda")
    xs, ys, errs = [], [], []
    for lam in LAMBDAS:
        if lam in sub.index:
            xs.append(lam + offset[loss_type])
            ys.append(sub.loc[lam, "mean"])
            errs.append(sub.loc[lam, "se"] * 1.96)   # 95% CI
    if xs:
        ax.errorbar(xs, ys, yerr=errs, fmt="o", color=color,
                    capsize=4, capthick=1.5, linewidth=1.5,
                    label=f"{loss_type} (±95% CI)", markersize=6)

# Individual seed dots (jittered)
rng = np.random.default_rng(0)
for loss_type, color in COLORS.items():
    sub = all_df[all_df["loss_type"] == loss_type]
    jitter = rng.uniform(-0.025, 0.025, len(sub))
    ax.scatter(sub["lambda"] + offset[loss_type] + jitter, sub["test_acc"],
               alpha=0.25, s=18, color=color, zorder=1)

ax.set_xlabel("λ (embedding loss weight)", fontsize=11)
ax.set_ylabel("Test accuracy", fontsize=11)
ax.set_title("LAMA T-REx: Effect of embedding loss weight", fontsize=12,
             fontweight="bold", pad=10)
ax.legend(fontsize=9)
ax.set_xticks(LAMBDAS)
ax.grid(True, alpha=0.25, axis="y")
ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig(LAMA_DIR / "sweep_results.png", dpi=150, bbox_inches="tight")
print("Saved sweep_results.png")

# ── Plot 2: averaged val-accuracy training curves ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.patch.set_facecolor("white")

for ax, (loss_type, color) in zip(axes, COLORS.items()):
    path = COS_CSV if loss_type == "cosine" else L2_CSV
    if not Path(path).exists():
        ax.set_title(f"{loss_type} — no data")
        continue
    df = pd.read_csv(path)
    df = df[df["epoch"] != "final"].copy()
    df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")
    df = df.dropna(subset=["lambda", "step", "val_acc"])

    lam_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(LAMBDAS)))
    for lam, lc in zip(LAMBDAS, lam_colors):
        sub = df[df["lambda"] == lam]
        if sub.empty: continue
        grouped = sub.groupby("step")["val_acc"].agg(["mean", "std"]).reset_index()
        ax.plot(grouped["step"], grouped["mean"], color=lc, linewidth=1.8,
                label=f"λ={lam}")
        ax.fill_between(grouped["step"],
                        grouped["mean"] - grouped["std"],
                        grouped["mean"] + grouped["std"],
                        alpha=0.12, color=lc)

    ax.set_title(f"{loss_type} loss", fontsize=11, fontweight="bold")
    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Val accuracy (mean ± std)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)

plt.suptitle("Validation accuracy curves by λ", fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(LAMA_DIR / "sweep_curves.png", dpi=150, bbox_inches="tight")
print("Saved sweep_curves.png")
plt.close("all")
