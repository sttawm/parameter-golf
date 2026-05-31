#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LAMA_DIR = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama"

# Exact colors from bar chart
COLORS = {
    "ce":  "#f8d7da",
    "co":  "#a5d6a7",
    "eu":  "#90caf9",
    "zs":  "#e0e0e0",
}

# ── Load ──────────────────────────────────────────────────────────────────────
df_cos = pd.read_csv(f"{LAMA_DIR}/lama_results_cosine_multi.csv")
df_cos["lambda"]  = pd.to_numeric(df_cos["lambda"],  errors="coerce")
df_cos["val_acc"] = pd.to_numeric(df_cos["val_acc"],  errors="coerce")
df_cos["step"]    = pd.to_numeric(df_cos["step"],     errors="coerce")

df_eu = pd.read_csv(f"{LAMA_DIR}/lama_results_eu.csv")
df_eu["val_acc"] = pd.to_numeric(df_eu["val_acc"], errors="coerce")
df_eu["step"]    = pd.to_numeric(df_eu["step"],    errors="coerce")

# Zero-shot mean (all conditions start from same pretrained weights)
zs_mean = pd.to_numeric(
    df_cos[pd.to_numeric(df_cos["epoch"], errors="coerce") == 0]["test_acc"],
    errors="coerce"
).mean()

def smooth(arr, w=2):
    return pd.Series(arr).rolling(w, min_periods=1).mean().values

def curve(df):
    df = df[df["epoch"] != "final"].dropna(subset=["step", "val_acc"])
    g   = df.groupby("step")["val_acc"]
    mu  = g.mean()
    std = g.std().fillna(0)
    # Prepend step=0 at zero-shot level
    x   = np.concatenate([[0], mu.index.values])
    m   = np.concatenate([[zs_mean], smooth(mu.values)])
    s   = np.concatenate([[0], smooth(std.values)])
    return x, m, s, df["seed"].nunique()

ce_x, ce_mu, ce_std, ce_n = curve(df_cos[df_cos["lambda"] == 0.0])
co_x, co_mu, co_std, co_n = curve(df_cos[df_cos["lambda"] == 1.0])
eu_x, eu_mu, eu_std, eu_n = curve(df_eu)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor("white")

LINE_COLORS = {
    "#f8d7da": "#c9606e",  # pastel pink → medium rose
    "#a5d6a7": "#4d9a5f",  # pastel green → medium green
    "#90caf9": "#4a88c7",  # pastel blue → medium blue
}

def plot_band(ax, x, mu, std, color, label, n):
    ax.plot(x, mu, color=LINE_COLORS[color], lw=2.0, label=f"{label}  (n={n})")
    ax.fill_between(x, mu - std, mu + std, color=color, alpha=0.4)

ax.axhline(zs_mean, color=COLORS["zs"], lw=1.5, linestyle="--",
           label=f"Zero-shot  ({zs_mean:.1%})", zorder=0)

plot_band(ax, ce_x, ce_mu, ce_std, COLORS["ce"], "CE only",   ce_n)
plot_band(ax, co_x, co_mu, co_std, COLORS["co"], "CE + Emb",  co_n)
plot_band(ax, eu_x, eu_mu, eu_std, COLORS["eu"], "Emb only*", eu_n)

ax.set_xlabel("Training step", fontsize=10)
ax.set_ylabel("Val accuracy", fontsize=10)
ax.set_title("Validation accuracy during training\nBERT-base · T-REx", fontsize=10, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(alpha=0.2)
ax.tick_params(labelsize=8.5)

fig.text(0.5, -0.03,
         "* Emb only uses Embedding-Similarity + Uniformity loss; no Cross-Entropy",
         ha="center", fontsize=7.5, color="#888", style="italic")

plt.tight_layout()
out = f"{LAMA_DIR}/training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
