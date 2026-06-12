#!/usr/bin/env python3
"""
Probability vs embedding distance for the Parameter Golf baseline model.

Same analysis as embedding_prob_pythia.py but using our own trained model.
The baseline uses tied embeddings: tok_emb.weight is the single matrix for
both input lookup and output projection (logits = softcap(x @ tok_emb.weight.T)).

Reads pre-tokenized val sequences directly from the on-disk binary shards
(no network required).

Outputs
-------
embedding_prob_baseline_plot.png  — 2×2 binned plot (cos & L2 × log-prob & prob)
embedding_prob_baseline_data.csv  — raw per-pair data
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from train_gpt_mlx import GPT, Hyperparameters, load_data_shard

import mlx.core as mx
from mlx.utils import tree_unflatten

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT   = "logs/baseline_330_mlx_model.npz"
VAL_SHARD    = "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"
N_CONTEXTS   = 500
CONTEXT_LEN  = 128       # tokens per context (model supports up to 1024)
TOP_K        = 50        # vocab is only 1024, so 50 is already a large fraction
N_BINS       = 25
OUT_PLOT     = "embedding_prob_baseline_plot.png"
OUT_CSV      = "embedding_prob_baseline_data.csv"

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading {CHECKPOINT}…")
args = Hyperparameters()
model = GPT(
    vocab_size         = args.vocab_size,
    num_layers         = args.num_layers,
    dim                = args.model_dim,
    num_heads          = args.num_heads,
    num_kv_heads       = args.num_kv_heads,
    mlp_mult           = args.mlp_mult,
    logit_chunk_tokens = args.logit_chunk_tokens,
    logit_softcap      = args.logit_softcap,
    rope_base          = args.rope_base,
    tied_embed_init_std= args.tied_embed_init_std,
    qk_gain_init       = args.qk_gain_init,
)
weights = mx.load(CHECKPOINT)   # mx.load handles bfloat16 npz natively
model.update(tree_unflatten(list(weights.items())))
mx.eval(model.parameters())
print(f"  vocab={args.vocab_size}  dim={args.model_dim}  layers={args.num_layers}")

# ── Data ──────────────────────────────────────────────────────────────────────
print(f"Reading {VAL_SHARD}…")
tokens = load_data_shard(Path(VAL_SHARD))  # shape (N_tokens,), int32
print(f"  {len(tokens):,} tokens available")

# ── Collect ───────────────────────────────────────────────────────────────────
log_prob_ratios, prob_ratios = [], []
l2s, cos_dists               = [], []
anchor_ids_all, other_ids_all = [], []
ranks_all                    = []

# Embedding matrix: tied, so same for input and output
E_np    = np.array(model.tok_emb.weight.astype(mx.float32))  # (V, D)
E_norm  = E_np / (np.linalg.norm(E_np, axis=-1, keepdims=True) + 1e-8)

step = 0
for ctx_idx in range(N_CONTEXTS):
    start = ctx_idx * CONTEXT_LEN
    end   = start + CONTEXT_LEN + 1
    if end > len(tokens):
        break

    ids = mx.array(tokens[start:end][None], dtype=mx.int32)   # (1, L+1)

    # Forward pass → hidden states → logits
    hidden  = model(ids[:, :-1])                                   # (1, L, D)
    hidden  = hidden.reshape(-1, hidden.shape[-1])                 # (L, D)
    E_mx    = model.tok_emb.weight.astype(hidden.dtype)
    logits  = model.softcap(hidden @ E_mx.T).astype(mx.float32)    # (L, V)
    mx.eval(logits)

    logits_np = np.array(logits)                                   # (L, V)
    L = logits_np.shape[0]

    # Softmax → log-probs
    log_p = logits_np - np.log(np.sum(np.exp(logits_np - logits_np.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)) - logits_np.max(axis=-1, keepdims=True)
    # Numerically stable: subtract max first
    shifted = logits_np - logits_np.max(axis=-1, keepdims=True)
    log_p   = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    probs   = np.exp(log_p)

    # Top-K per position
    top_idx = np.argsort(probs, axis=-1)[:, -TOP_K:][:, ::-1]     # (L, K)

    top_p   = probs[np.arange(L)[:, None], top_idx]               # (L, K)
    top_lp  = log_p[np.arange(L)[:, None], top_idx]               # (L, K)

    # Embeddings for top-K tokens at each position
    Ek      = E_np[top_idx]       # (L, K, D)
    Ek_norm = E_norm[top_idx]     # (L, K, D)

    anchor      = Ek[:, 0:1, :]
    anchor_norm = Ek_norm[:, 0:1, :]

    other      = Ek[:, 1:, :]
    other_norm = Ek_norm[:, 1:, :]

    l2       = np.linalg.norm(other - anchor, axis=-1)                # (L, K-1)
    cos_dist = 1 - np.sum(other_norm * anchor_norm, axis=-1)          # (L, K-1)

    lpr = (top_lp[:, 1:] - top_lp[:, 0:1]).ravel()
    pr  = (top_p[:, 1:]  / (top_p[:, 0:1] + 1e-12)).ravel()

    log_prob_ratios.append(lpr)
    prob_ratios.append(pr)
    l2s.append(l2.ravel())
    cos_dists.append(cos_dist.ravel())
    anchor_ids_all.append(np.tile(top_idx[:, 0], TOP_K - 1).reshape(TOP_K - 1, L).T.ravel())
    other_ids_all.append(top_idx[:, 1:].ravel())
    ranks_all.append(np.tile(np.arange(1, TOP_K), L))

    step += 1
    if step % 100 == 0:
        print(f"  {step}/{N_CONTEXTS} contexts")

# ── DataFrame ─────────────────────────────────────────────────────────────────
print("Building dataframe…")
anchor_ids = np.concatenate(anchor_ids_all)
other_ids  = np.concatenate(other_ids_all)

# Decode token IDs using the vocab file
vocab_path = "data/tokenizers/fineweb_1024_bpe.vocab"
id2tok = {}
try:
    with open(vocab_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                id2tok[int(parts[1]) if parts[1].isdigit() else len(id2tok)] = parts[0]
except Exception:
    pass
def tok_str(i): return id2tok.get(int(i), str(i))

df = pd.DataFrame({
    "log_prob_ratio": np.concatenate(log_prob_ratios),
    "prob_ratio":     np.concatenate(prob_ratios),
    "l2_dist":        np.concatenate(l2s),
    "cos_dist":       np.concatenate(cos_dists),
    "rank":           np.concatenate(ranks_all),
    "anchor_id":      anchor_ids,
    "other_id":       other_ids,
})
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df):,} rows → {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
combos = [
    ("log_prob_ratio", "cos_dist",
     "log p_i − log p_top  (log-prob ratio)", "Cosine distance (tied emb)"),
    ("log_prob_ratio", "l2_dist",
     "log p_i − log p_top  (log-prob ratio)", "L2 distance (tied emb)"),
    ("prob_ratio",     "cos_dist",
     "p_i / p_top  (probability ratio)",      "Cosine distance (tied emb)"),
    ("prob_ratio",     "l2_dist",
     "p_i / p_top  (probability ratio)",      "L2 distance (tied emb)"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    f"Parameter Golf baseline  ·  top-{TOP_K} tokens  ·  {N_CONTEXTS} val contexts  "
    f"·  {len(df):,} pairs\n(tied embeddings · vocab={args.vocab_size} · dim={args.model_dim})",
    fontsize=12,
)

for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flat, combos):
    x = df[xcol].values
    y = df[ycol].values

    hb = ax.hexbin(x, y, gridsize=60, cmap="Blues", bins="log",
                   mincnt=1, linewidths=0.2, alpha=0.85)
    plt.colorbar(hb, ax=ax, label="pair count (log₁₀)")

    edges = np.percentile(x, np.linspace(0, 100, N_BINS + 1))
    edges = np.unique(edges)
    ctrs, mus, sigs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < 10:
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
