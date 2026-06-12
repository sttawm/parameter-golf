#!/usr/bin/env python3
"""
Do tokens with higher predicted probability sit closer to the top-predicted
token in GPT-2's embedding space?

Per context (FineWeb), we anchor on the argmax token at each position and
compare the remaining top-K tokens by (log-)probability ratio vs embedding
distance (cosine and L2).

Outputs
-------
embedding_prob_data.csv  — raw per-pair data for downstream analysis
embedding_prob_plot.png  — 2×2 binned plot with correlation statistics
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "gpt2-medium"
TOP_K       = 100      # tokens to compare per position (anchor + 99 others)
N_CONTEXTS  = 500      # FineWeb documents to process
CONTEXT_LEN = 128      # max tokens per context
N_BINS      = 25       # bins for the mean ± std overlay
SEED        = 42
OUT_CSV     = "embedding_prob_data.csv"
OUT_PLOT    = "embedding_prob_plot.png"

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} on {DEVICE}…")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
model     = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# GPT-2 uses tied embeddings: wte == lm_head.weight
E      = model.transformer.wte.weight.detach().float().to(DEVICE)  # (V, D)
E_norm = E / E.norm(dim=-1, keepdim=True)                          # pre-normalised

# ── Collect ───────────────────────────────────────────────────────────────────
log_prob_ratios, prob_ratios   = [], []
l2s, cos_dists                 = [], []
anchor_ids_all, other_ids_all  = [], []
ranks_all                      = []

print("Streaming FineWeb sample-10BT…")
ds = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    streaming=True,
).shuffle(seed=SEED, buffer_size=10_000)

done = 0
for ex in ds:
    if done >= N_CONTEXTS:
        break

    enc = tokenizer(
        ex["text"],
        return_tensors="pt",
        truncation=True,
        max_length=CONTEXT_LEN + 1,
    )
    ids = enc["input_ids"].to(DEVICE)
    if ids.shape[1] < 16:
        continue

    with torch.no_grad():
        logits = model(ids[:, :-1]).logits[0]       # (L, V)

    log_p = torch.log_softmax(logits, dim=-1)       # (L, V)
    probs  = log_p.exp()

    top_p, top_i = probs.topk(TOP_K, dim=-1)        # (L, K)
    top_lp       = log_p.gather(1, top_i)           # (L, K)

    Ek      = E[top_i]           # (L, K, D)
    Ek_norm = E_norm[top_i]      # (L, K, D)

    anchor      = Ek[:, 0:1, :]
    anchor_norm = Ek_norm[:, 0:1, :]

    other      = Ek[:, 1:, :]
    other_norm = Ek_norm[:, 1:, :]

    l2       = (other - anchor).norm(dim=-1)                   # (L, K-1)
    cos_dist = 1 - (other_norm * anchor_norm).sum(dim=-1)      # (L, K-1)

    lpr = (top_lp[:, 1:] - top_lp[:, 0:1]).cpu().numpy()      # ≤ 0
    pr  = (top_p[:, 1:]  / top_p[:, 0:1]).cpu().numpy()       # ≤ 1

    L = logits.shape[0]
    log_prob_ratios.append(lpr.ravel())
    prob_ratios.append(pr.ravel())
    l2s.append(l2.cpu().numpy().ravel())
    cos_dists.append(cos_dist.cpu().numpy().ravel())
    anchor_ids_all.append(
        top_i[:, 0:1].expand(-1, TOP_K - 1).cpu().numpy().ravel()
    )
    other_ids_all.append(top_i[:, 1:].cpu().numpy().ravel())
    ranks_all.append(np.tile(np.arange(1, TOP_K), L))

    done += 1
    if done % 50 == 0:
        print(f"  {done}/{N_CONTEXTS} contexts")

# ── DataFrame ─────────────────────────────────────────────────────────────────
print("Building dataframe…")
anchor_ids = np.concatenate(anchor_ids_all)
other_ids  = np.concatenate(other_ids_all)

df = pd.DataFrame({
    "log_prob_ratio": np.concatenate(log_prob_ratios),
    "prob_ratio":     np.concatenate(prob_ratios),
    "l2_dist":        np.concatenate(l2s),
    "cos_dist":       np.concatenate(cos_dists),
    "rank":           np.concatenate(ranks_all),
    "anchor_id":      anchor_ids,
    "other_id":       other_ids,
    "anchor_token":   [tokenizer.decode([int(i)]) for i in anchor_ids],
    "other_token":    [tokenizer.decode([int(i)]) for i in other_ids],
})
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df):,} rows → {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
combos = [
    ("log_prob_ratio", "cos_dist",
     "log p_i − log p_top  (log-prob ratio)", "Cosine distance to top token"),
    ("log_prob_ratio", "l2_dist",
     "log p_i − log p_top  (log-prob ratio)", "L2 distance to top token"),
    ("prob_ratio",     "cos_dist",
     "p_i / p_top  (probability ratio)",      "Cosine distance to top token"),
    ("prob_ratio",     "l2_dist",
     "p_i / p_top  (probability ratio)",      "L2 distance to top token"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    f"{MODEL_NAME}  ·  top-{TOP_K} tokens  ·  {N_CONTEXTS} FineWeb contexts  "
    f"·  {len(df):,} pairs\n(tied embeddings: wte = lm_head)",
    fontsize=12,
)

for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flat, combos):
    x = df[xcol].values
    y = df[ycol].values

    hb = ax.hexbin(x, y, gridsize=70, cmap="Blues", bins="log",
                   mincnt=1, linewidths=0.2, alpha=0.85)
    plt.colorbar(hb, ax=ax, label="pair count (log₁₀)")

    edges = np.percentile(x, np.linspace(0, 100, N_BINS + 1))
    edges = np.unique(edges)
    ctrs, mus, sigs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < 30:
            continue
        ctrs.append((lo + hi) / 2)
        mus.append(y[m].mean())
        sigs.append(y[m].std())

    ctrs = np.array(ctrs)
    mus  = np.array(mus)
    sigs = np.array(sigs)
    ax.plot(ctrs, mus, color="crimson", lw=2.0, zorder=3, label="bin mean")
    ax.fill_between(ctrs, mus - sigs, mus + sigs,
                    color="crimson", alpha=0.22, zorder=2, label="±1 std")

    rp, _ = stats.pearsonr(x, y)
    rs, _ = stats.spearmanr(x, y)
    ax.set_title(f"Pearson r = {rp:.3f}   Spearman ρ = {rs:.3f}", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"Saved plot → {OUT_PLOT}")
plt.show()
