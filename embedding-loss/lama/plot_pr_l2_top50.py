#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

N_BINS   = 25
TOP_K    = 50
OUT_DIR  = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama"

models = [
    ("GPT-2-medium",         f"{OUT_DIR}/gpt2_embed_corr_data.pkl",        "gpt2"),
    ("parameter-golf (λ=0)", f"{OUT_DIR}/pg_baseline_embed_corr_data.pkl", "pg_baseline"),
    ("bert-base-uncased",    f"{OUT_DIR}/bert_embed_corr_data.pkl",         "bert"),
]

def plot_one(title, data, slug):
    mask = data["rank"] <= TOP_K
    x    = data["lpr"][mask]
    y    = data["l2"][mask]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")

    hb = ax.hexbin(x, y, gridsize=60, cmap="Blues", bins="log",
                   mincnt=1, linewidths=0.15, alpha=0.9)
    plt.colorbar(hb, ax=ax, label="pair count (log₁₀)", shrink=0.85)

    edges = np.percentile(x, np.linspace(0, 100, N_BINS + 1))
    edges = np.unique(edges)
    ctrs, mus, sigs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < 30: continue
        ctrs.append((lo + hi) / 2); mus.append(y[m].mean()); sigs.append(y[m].std())
    ctrs = np.array(ctrs); mus = np.array(mus); sigs = np.array(sigs)
    ax.plot(ctrs, mus, color="crimson", lw=2.0, zorder=3, label="bin mean")
    ax.fill_between(ctrs, mus - sigs, mus + sigs,
                    color="crimson", alpha=0.20, zorder=2, label="±1 std")
    ax.legend(fontsize=9, loc="lower left")

    rp, _ = stats.pearsonr(x, y); rs, _ = stats.spearmanr(x, y)
    n_pairs = mask.sum()
    ax.set_title(f"{title}  ·  top-{TOP_K} tokens  ·  {n_pairs:,} pairs",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("log p_i − log p_top  (log-prob ratio)", fontsize=10)
    ax.set_ylabel("L2 distance to top token", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.text(0.97, 0.97, f"Pearson r = {rp:.3f}\nSpearman ρ = {rs:.3f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    plt.tight_layout()
    out = f"{OUT_DIR}/{slug}_lpr_l2_top{TOP_K}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")

for title, path, slug in models:
    plot_one(title, pickle.load(open(path, "rb")), slug)
