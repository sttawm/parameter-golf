#!/usr/bin/env python3
"""
Probability vs embedding-distance correlation for three models:
  1. bert-base-uncased  (MLM, pre-trained, no fine-tuning)
  2. parameter-golf baseline  (causal GPT, trained without lambda term)
  3. gpt2-medium  (causal GPT)

Saves per-model pickle with columns: lpr, pr, l2, cos, rank
so any top-K subset can be plotted without re-running.
"""
import sys, io, zlib, os, pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    BertForMaskedLM, BertTokenizerFast,
)

sys.path.insert(0, "/Users/sttawm/dev/parameter-golf")
import sentencepiece as spm
from train_gpt import GPT, dequantize_state_dict_int8, Hyperparameters

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K       = 100
N_CONTEXTS  = 500
CONTEXT_LEN = 128
N_BINS      = 25
SEED        = 42
MASK_BATCH  = 64

BASE    = "/Users/sttawm/dev/parameter-golf"
SP_PATH = f"{BASE}/data/tokenizers/fineweb_1024_bpe.model"
PG_CKPT = f"{BASE}/logs/baseline_330_mlx_model.int8.ptz"
OUT_DIR = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama"

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
print(f"Device: {DEVICE}")
torch.manual_seed(SEED); np.random.seed(SEED)


# ── 1. Stream raw text contexts ───────────────────────────────────────────────
print("Streaming FineWeb…")
raw_texts: list[str] = []
ds = load_dataset(
    "HuggingFaceFW/fineweb", name="sample-10BT",
    split="train", streaming=True,
).shuffle(seed=SEED, buffer_size=10_000)
for ex in ds:
    if len(raw_texts) >= N_CONTEXTS: break
    raw_texts.append(ex["text"])
print(f"  {len(raw_texts)} contexts")


# ── 2. Core helpers ───────────────────────────────────────────────────────────
def _pairs_from_logits(logits_nv, E, E_norm):
    """Returns (lpr, pr, l2, cos, rank) all shape (N*(K-1),) float32 numpy."""
    log_p = torch.log_softmax(logits_nv.float(), dim=-1)
    probs  = log_p.exp()
    top_p, top_i = probs.topk(TOP_K, dim=-1)      # (N, K)
    top_lp        = log_p.gather(1, top_i)

    Ek      = E[top_i]
    Ek_norm = E_norm[top_i]
    l2       = (Ek[:, 1:] - Ek[:, 0:1]).norm(dim=-1)
    cos_dist = 1 - (Ek_norm[:, 1:] * Ek_norm[:, 0:1]).sum(dim=-1)

    N = logits_nv.shape[0]
    ranks = np.tile(np.arange(2, TOP_K + 1), N)   # rank 2..K repeated N times

    lpr = (top_lp[:, 1:] - top_lp[:, 0:1]).cpu().numpy().ravel()
    pr  = (top_p[:, 1:]  / top_p[:, 0:1]).cpu().numpy().ravel()
    return lpr, pr, l2.cpu().numpy().ravel(), cos_dist.cpu().numpy().ravel(), ranks


def run_causal(model_fwd, E, E_norm, tokenizer, texts, encode_fn=None):
    lpr_all, pr_all, l2_all, cos_all, rank_all = [], [], [], [], []
    for i, text in enumerate(texts):
        if encode_fn:
            ids_list = encode_fn(text)[: CONTEXT_LEN + 1]
            if len(ids_list) < 16: continue
            ids = torch.tensor([ids_list[:-1]], dtype=torch.long, device=DEVICE)
        else:
            enc = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=CONTEXT_LEN + 1)
            if enc["input_ids"].shape[1] < 16: continue
            ids = enc["input_ids"][:, :-1].to(DEVICE)

        with torch.no_grad():
            logits = model_fwd(ids)

        lpr, pr, l2, cos, rank = _pairs_from_logits(logits, E, E_norm)
        lpr_all.append(lpr); pr_all.append(pr)
        l2_all.append(l2);   cos_all.append(cos); rank_all.append(rank)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(texts)}")

    return (np.concatenate(lpr_all), np.concatenate(pr_all),
            np.concatenate(l2_all),  np.concatenate(cos_all),
            np.concatenate(rank_all))


def run_bert(model, E, E_norm, tokenizer, texts):
    mask_id  = tokenizer.mask_token_id
    skip_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    lpr_all, pr_all, l2_all, cos_all, rank_all = [], [], [], [], []

    for i, text in enumerate(texts):
        enc  = tokenizer(text, return_tensors="pt",
                         truncation=True, max_length=CONTEXT_LEN)
        ids  = enc["input_ids"][0]
        amsk = enc["attention_mask"][0]
        L    = ids.shape[0]
        valid_pos = [p for p in range(L) if int(ids[p]) not in skip_ids]
        if len(valid_pos) < 4: continue

        all_logits = []
        for b0 in range(0, len(valid_pos), MASK_BATCH):
            b_pos  = valid_pos[b0: b0 + MASK_BATCH]
            B      = len(b_pos)
            b_ids  = ids.unsqueeze(0).expand(B, -1).clone().to(DEVICE)
            b_amsk = amsk.unsqueeze(0).expand(B, -1).to(DEVICE)
            for j, pos in enumerate(b_pos):
                b_ids[j, pos] = mask_id
            with torch.no_grad():
                logits_b = model(input_ids=b_ids, attention_mask=b_amsk).logits
            for j, pos in enumerate(b_pos):
                all_logits.append(logits_b[j, pos])

        if not all_logits: continue
        logits_nv = torch.stack(all_logits)
        top1 = logits_nv.argmax(dim=-1)
        keep = [j for j in range(len(top1)) if int(top1[j]) not in skip_ids]
        if not keep: continue
        logits_nv = logits_nv[keep]

        lpr, pr, l2, cos, rank = _pairs_from_logits(logits_nv, E, E_norm)
        lpr_all.append(lpr); pr_all.append(pr)
        l2_all.append(l2);   cos_all.append(cos); rank_all.append(rank)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(texts)}")

    return (np.concatenate(lpr_all), np.concatenate(pr_all),
            np.concatenate(l2_all),  np.concatenate(cos_all),
            np.concatenate(rank_all))


def save_model_data(name_slug, lpr, pr, l2, cos, rank):
    data = dict(lpr=lpr, pr=pr, l2=l2, cos=cos, rank=rank)
    path = f"{OUT_DIR}/{name_slug}_embed_corr_data.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  saved {len(lpr):,} pairs → {path}")
    return data


# ── 3. Load models and collect data ──────────────────────────────────────────
results: dict[str, dict] = {}

# ── 3a. GPT-2-medium ─────────────────────────────────────────────────────────
print("\n=== GPT-2-medium ===")
gpt2_tok   = GPT2TokenizerFast.from_pretrained("gpt2-medium")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(DEVICE).eval()
E_gpt2     = gpt2_model.transformer.wte.weight.detach().float().to(DEVICE)
E_gpt2_n   = F.normalize(E_gpt2, dim=-1)

def gpt2_fwd(ids):
    return gpt2_model(ids).logits[0]

lpr, pr, l2, cos, rank = run_causal(gpt2_fwd, E_gpt2, E_gpt2_n, gpt2_tok, raw_texts)
results["GPT-2-medium"] = save_model_data("gpt2", lpr, pr, l2, cos, rank)

del gpt2_model
if DEVICE == "mps": torch.mps.empty_cache()

# ── 3b. Parameter-golf baseline ──────────────────────────────────────────────
print("\n=== Parameter-golf baseline (λ=0) ===")
with open(PG_CKPT, "rb") as f:
    quant_blob = f.read()
quant_state = pickle.loads(zlib.decompress(quant_blob))
# MLX checkpoint uses numpy arrays; dequantize_state_dict_int8 expects tensors
def _np_to_tensor(obj):
    if isinstance(obj, dict):   return {k: _np_to_tensor(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray): return torch.from_numpy(obj)
    return obj
for key in ("quantized", "scales", "passthrough"):
    if key in quant_state:
        quant_state[key] = _np_to_tensor(quant_state[key])
state_dict = dequantize_state_dict_int8(quant_state)

hp = Hyperparameters()
pg_model = GPT(
    vocab_size=hp.vocab_size, num_layers=hp.num_layers,
    model_dim=hp.model_dim, num_heads=hp.num_heads,
    num_kv_heads=hp.num_kv_heads, mlp_mult=hp.mlp_mult,
    tie_embeddings=hp.tie_embeddings, tied_embed_init_std=hp.tied_embed_init_std,
    logit_softcap=hp.logit_softcap, rope_base=hp.rope_base,
    qk_gain_init=hp.qk_gain_init,
).to(DEVICE).eval()
pg_model.load_state_dict(state_dict, strict=True)

E_pg   = pg_model.tok_emb.weight.detach().float().to(DEVICE)
E_pg_n = F.normalize(E_pg, dim=-1)

sp        = spm.SentencePieceProcessor(model_file=SP_PATH)
encode_sp = lambda text: sp.encode(text, out_type=int)

def pg_fwd(ids):
    _, logits = pg_model._get_logits(ids)
    return logits.reshape(ids.shape[1], -1)

lpr, pr, l2, cos, rank = run_causal(pg_fwd, E_pg, E_pg_n,
                                     tokenizer=None, texts=raw_texts,
                                     encode_fn=encode_sp)
results["parameter-golf\n(baseline, λ=0)"] = save_model_data("pg_baseline", lpr, pr, l2, cos, rank)

del pg_model
if DEVICE == "mps": torch.mps.empty_cache()


# ── 4. Plot ───────────────────────────────────────────────────────────────────
def plot_2x2(data, title, out_path, top_k_filter=None):
    combos = [
        ("lpr", "cos", "log p_i − log p_top  (log-prob ratio)", "Cosine distance to top token"),
        ("lpr", "l2",  "log p_i − log p_top  (log-prob ratio)", "L2 distance to top token"),
        ("pr",  "cos", "p_i / p_top  (probability ratio)",      "Cosine distance to top token"),
        ("pr",  "l2",  "p_i / p_top  (probability ratio)",      "L2 distance to top token"),
    ]
    mask = (data["rank"] <= top_k_filter) if top_k_filter else np.ones(len(data["rank"]), dtype=bool)
    n_pairs = mask.sum()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("white")
    k_label = f"top-{top_k_filter}" if top_k_filter else f"top-{TOP_K}"
    fig.suptitle(f"{title}  ·  {k_label} tokens  ·  {N_CONTEXTS} FineWeb contexts  ·  {n_pairs:,} pairs",
                 fontsize=12, y=1.01)

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flat, combos):
        x = data[xcol][mask]; y = data[ycol][mask]
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
        ax.fill_between(ctrs, mus - sigs, mus + sigs, color="crimson", alpha=0.20, zorder=2, label="±1 std")
        ax.legend(fontsize=8, loc="upper left")

        rp, _ = stats.pearsonr(x, y); rs, _ = stats.spearmanr(x, y)
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.text(0.97, 0.97, f"Pearson r = {rp:.3f}\nSpearman ρ = {rs:.3f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


MODELS = list(results.keys())
MODEL_SLUGS = ["gpt2", "pg_baseline", "bert"]

for name, slug in zip(MODELS, MODEL_SLUGS):
    label = name.replace("\n", " ")
    for k in [50, 100]:
        out = f"{OUT_DIR}/{slug}_embed_corr_top{k}.png"
        plot_2x2(results[name], label, out, top_k_filter=k)
