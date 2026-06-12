#!/usr/bin/env python3
"""
Probability vs embedding distance for Pythia-410M (untied embeddings).

For each context (FineWeb), anchor on the argmax token and compare the
remaining top-K tokens by log-probability ratio vs distance in BOTH the
input embedding matrix (embed_in) and the output/unembedding matrix
(embed_out).

The theory predicts embed_out should show a stronger correlation since
probabilities are softmax(h · embed_out^T) — embed_out is directly
optimized to separate tokens in prediction space.

Outputs
-------
embedding_prob_pythia_data.csv   — raw per-pair data
embedding_prob_pythia_plot.png   — 2×2 plot (embed_in/out × cosine/L2)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "EleutherAI/pythia-410m"
TOP_K       = 100
N_CONTEXTS  = 500
CONTEXT_LEN = 128
N_BINS      = 25
SEED        = 42
OUT_CSV     = "embedding_prob_pythia_data.csv"
OUT_PLOT    = "embedding_prob_pythia_plot.png"

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} on {DEVICE}…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Pythia has untied embeddings — two distinct matrices
E_in      = model.gpt_neox.embed_in.weight.detach().float().to(DEVICE)   # (V, D)
E_out     = model.embed_out.weight.detach().float().to(DEVICE)            # (V, D)
E_in_norm  = E_in  / E_in.norm(dim=-1, keepdim=True)
E_out_norm = E_out / E_out.norm(dim=-1, keepdim=True)

# ── Collect ───────────────────────────────────────────────────────────────────
lists = {k: [] for k in [
    "log_prob_ratio",
    "l2_in", "cos_in",
    "l2_out", "cos_out",
    "anchor_id", "other_id", "rank",
]}

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

    log_p = torch.log_softmax(logits, dim=-1)
    probs  = log_p.exp()
    top_p, top_i = probs.topk(TOP_K, dim=-1)        # (L, K)
    top_lp = log_p.gather(1, top_i)                 # (L, K)

    def distances(E, E_norm):
        Ek      = E[top_i]           # (L, K, D)
        Ek_norm = E_norm[top_i]
        anchor      = Ek[:, 0:1, :]
        anchor_norm = Ek_norm[:, 0:1, :]
        other       = Ek[:, 1:, :]
        other_norm  = Ek_norm[:, 1:, :]
        l2  = (other - anchor).norm(dim=-1)
        cos = 1 - (other_norm * anchor_norm).sum(dim=-1)
        return l2.cpu().numpy().ravel(), cos.cpu().numpy().ravel()

    l2_in,  cos_in  = distances(E_in,  E_in_norm)
    l2_out, cos_out = distances(E_out, E_out_norm)

    lpr = (top_lp[:, 1:] - top_lp[:, 0:1]).cpu().numpy().ravel()
    L   = logits.shape[0]

    lists["log_prob_ratio"].append(lpr)
    lists["l2_in"].append(l2_in)
    lists["cos_in"].append(cos_in)
    lists["l2_out"].append(l2_out)
    lists["cos_out"].append(cos_out)
    lists["anchor_id"].append(
        top_i[:, 0:1].expand(-1, TOP_K - 1).cpu().numpy().ravel()
    )
    lists["other_id"].append(top_i[:, 1:].cpu().numpy().ravel())
    lists["rank"].append(np.tile(np.arange(1, TOP_K), L))

    done += 1
    if done % 50 == 0:
        print(f"  {done}/{N_CONTEXTS} contexts")

# ── DataFrame ─────────────────────────────────────────────────────────────────
print("Building dataframe…")
anchor_ids = np.concatenate(lists["anchor_id"])
other_ids  = np.concatenate(lists["other_id"])

df = pd.DataFrame({
    "log_prob_ratio": np.concatenate(lists["log_prob_ratio"]),
    "l2_in":          np.concatenate(lists["l2_in"]),
    "cos_in":         np.concatenate(lists["cos_in"]),
    "l2_out":         np.concatenate(lists["l2_out"]),
    "cos_out":        np.concatenate(lists["cos_out"]),
    "rank":           np.concatenate(lists["rank"]),
    "anchor_id":      anchor_ids,
    "other_id":       other_ids,
    "anchor_token":   [tokenizer.decode([int(i)]) for i in anchor_ids],
    "other_token":    [tokenizer.decode([int(i)]) for i in other_ids],
})
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df):,} rows → {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
combos = [
    ("cos_in",  "embed_in  — cosine distance"),
    ("l2_in",   "embed_in  — L2 distance"),
    ("cos_out", "embed_out — cosine distance"),
    ("l2_out",  "embed_out — L2 distance"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    f"{MODEL_NAME}  ·  top-{TOP_K} tokens  ·  {N_CONTEXTS} FineWeb contexts  "
    f"·  {len(df):,} pairs\n(untied embeddings: embed_in ≠ embed_out)",
    fontsize=12,
)

x = df["log_prob_ratio"].values

for ax, (ycol, ylabel) in zip(axes.flat, combos):
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
    ax.set_xlabel("log p_i − log p_top  (log-prob ratio)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"Saved plot → {OUT_PLOT}")
plt.show()
