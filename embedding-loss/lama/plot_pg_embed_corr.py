#!/usr/bin/env python3
"""Parameter-golf baseline probability vs embedding distance — pod version."""
import sys, io, zlib, pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from datasets import load_dataset
import sentencepiece as spm

sys.path.insert(0, "/Users/sttawm/dev/parameter-golf")
from train_gpt import GPT, dequantize_state_dict_int8, Hyperparameters

TOP_K       = 50
N_CONTEXTS  = 500
CONTEXT_LEN = 128
N_BINS      = 25
SEED        = 42

SP_PATH  = "/Users/sttawm/dev/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
PG_CKPT  = "/Users/sttawm/dev/parameter-golf/logs/baseline_330_mlx_model.int8.ptz"
OUT_DATA = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/pg_embed_corr_data.pkl"
OUT_PNG  = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/pg_embed_corr_preview.png"

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
torch.manual_seed(SEED); np.random.seed(SEED)

# Load model
print("Loading parameter-golf baseline…")
with open(PG_CKPT, "rb") as f:
    quant_blob = f.read()
quant_state = pickle.loads(zlib.decompress(quant_blob))
# MLX checkpoint stores numpy arrays; dequantize_state_dict_int8 expects tensors
import numpy as _np
def _np_to_tensor(obj):
    if isinstance(obj, dict):
        return {k: _np_to_tensor(v) for k, v in obj.items()}
    if isinstance(obj, _np.ndarray):
        return torch.from_numpy(obj)
    return obj
for key in ("quantized", "scales", "passthrough"):
    if key in quant_state:
        quant_state[key] = _np_to_tensor(quant_state[key])
state_dict  = dequantize_state_dict_int8(quant_state)

hp = Hyperparameters()
model = GPT(
    vocab_size=hp.vocab_size, num_layers=hp.num_layers,
    model_dim=hp.model_dim, num_heads=hp.num_heads,
    num_kv_heads=hp.num_kv_heads, mlp_mult=hp.mlp_mult,
    tie_embeddings=hp.tie_embeddings, tied_embed_init_std=hp.tied_embed_init_std,
    logit_softcap=hp.logit_softcap, rope_base=hp.rope_base,
    qk_gain_init=hp.qk_gain_init,
).to(DEVICE).eval()
model.load_state_dict(state_dict, strict=True)

E      = model.tok_emb.weight.detach().float().to(DEVICE)
E_norm = F.normalize(E, dim=-1)

sp = spm.SentencePieceProcessor(model_file=SP_PATH)

# Stream contexts
print("Streaming FineWeb…")
raw_texts = []
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                  split="train", streaming=True).shuffle(seed=SEED, buffer_size=10_000)
for ex in ds:
    if len(raw_texts) >= N_CONTEXTS: break
    raw_texts.append(ex["text"])
print(f"  {len(raw_texts)} contexts")

# Collect pairs
lpr_all, pr_all, l2_all, cos_all = [], [], [], []
for i, text in enumerate(raw_texts):
    ids_list = sp.encode(text, out_type=int)[: CONTEXT_LEN + 1]
    if len(ids_list) < 16:
        continue
    ids = torch.tensor([ids_list[:-1]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        _, logits = model._get_logits(ids)
        logits = logits.reshape(ids.shape[1], -1)  # (L, V)

    log_p = torch.log_softmax(logits.float(), dim=-1)
    probs  = log_p.exp()
    top_p, top_i = probs.topk(TOP_K, dim=-1)
    top_lp        = log_p.gather(1, top_i)

    Ek      = E[top_i]; Ek_norm = E_norm[top_i]
    l2       = (Ek[:, 1:] - Ek[:, 0:1]).norm(dim=-1)
    cos_dist = 1 - (Ek_norm[:, 1:] * Ek_norm[:, 0:1]).sum(dim=-1)

    lpr_all.append((top_lp[:, 1:] - top_lp[:, 0:1]).cpu().numpy().ravel())
    pr_all.append((top_p[:, 1:] / top_p[:, 0:1]).cpu().numpy().ravel())
    l2_all.append(l2.cpu().numpy().ravel())
    cos_all.append(cos_dist.cpu().numpy().ravel())
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(raw_texts)}")

data = dict(lpr=np.concatenate(lpr_all), pr=np.concatenate(pr_all),
            l2=np.concatenate(l2_all),   cos=np.concatenate(cos_all))
with open(OUT_DATA, "wb") as f:
    pickle.dump(data, f)
print(f"  {len(data['lpr']):,} pairs saved → {OUT_DATA}")

# Plot
x, y = data["lpr"], data["cos"]
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
ax.plot(ctrs, mus, color="crimson", lw=2.0, zorder=3)
ax.fill_between(ctrs, mus - sigs, mus + sigs, color="crimson", alpha=0.20, zorder=2)

rp, _ = stats.pearsonr(x, y); rs, _ = stats.spearmanr(x, y)
ax.set_title("parameter-golf baseline (λ=0)", fontsize=13, fontweight="bold", pad=8)
ax.set_xlabel("log p_i − log p_top  (log-prob ratio)", fontsize=10)
ax.set_ylabel("Cosine distance to top token", fontsize=10)
ax.text(0.97, 0.97, f"Pearson r = {rp:.3f}\nSpearman ρ = {rs:.3f}",
        transform=ax.transAxes, fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
print(f"Pearson r = {rp:.3f}  Spearman ρ = {rs:.3f}")
