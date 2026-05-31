#!/usr/bin/env python3
"""Run BERT collection only, then plot all three models from saved pickles."""
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from datasets import load_dataset
from transformers import BertForMaskedLM, BertTokenizerFast

TOP_K       = 100
N_CONTEXTS  = 500
CONTEXT_LEN = 128
MASK_BATCH  = 64
N_BINS      = 25
SEED        = 42
OUT_DIR     = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")
torch.manual_seed(SEED); np.random.seed(SEED)

# ── Stream contexts ───────────────────────────────────────────────────────────
print("Streaming FineWeb…")
raw_texts = []
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                  split="train", streaming=True).shuffle(seed=SEED, buffer_size=10_000)
for ex in ds:
    if len(raw_texts) >= N_CONTEXTS: break
    raw_texts.append(ex["text"])
print(f"  {len(raw_texts)} contexts")

# ── Run BERT ──────────────────────────────────────────────────────────────────
print("\n=== BERT (bert-base-uncased) ===")
bert_tok   = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(DEVICE).eval()
E      = bert_model.bert.embeddings.word_embeddings.weight.detach().float().to(DEVICE)
E_norm = F.normalize(E, dim=-1)

mask_id  = bert_tok.mask_token_id
skip_ids = {bert_tok.cls_token_id, bert_tok.sep_token_id, bert_tok.pad_token_id}

lpr_all, pr_all, l2_all, cos_all, rank_all = [], [], [], [], []
for i, text in enumerate(raw_texts):
    enc  = bert_tok(text, return_tensors="pt", truncation=True, max_length=CONTEXT_LEN)
    ids  = enc["input_ids"][0]
    amsk = enc["attention_mask"][0]
    valid_pos = [p for p in range(ids.shape[0]) if int(ids[p]) not in skip_ids]
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
            logits_b = bert_model(input_ids=b_ids, attention_mask=b_amsk).logits
        for j, pos in enumerate(b_pos):
            all_logits.append(logits_b[j, pos])

    if not all_logits: continue
    logits_nv = torch.stack(all_logits)
    keep = [j for j in range(logits_nv.shape[0])
            if int(logits_nv[j].argmax()) not in skip_ids]
    if not keep: continue
    logits_nv = logits_nv[keep]

    log_p = torch.log_softmax(logits_nv.float(), dim=-1)
    probs  = log_p.exp()
    top_p, top_i = probs.topk(TOP_K, dim=-1)
    top_lp        = log_p.gather(1, top_i)

    Ek      = E[top_i]; Ek_norm = E_norm[top_i]
    l2       = (Ek[:, 1:] - Ek[:, 0:1]).norm(dim=-1)
    cos_dist = 1 - (Ek_norm[:, 1:] * Ek_norm[:, 0:1]).sum(dim=-1)
    N = logits_nv.shape[0]

    lpr_all.append((top_lp[:, 1:] - top_lp[:, 0:1]).cpu().numpy().ravel())
    pr_all.append((top_p[:, 1:] / top_p[:, 0:1]).cpu().numpy().ravel())
    l2_all.append(l2.cpu().numpy().ravel())
    cos_all.append(cos_dist.cpu().numpy().ravel())
    rank_all.append(np.tile(np.arange(2, TOP_K + 1), N))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(raw_texts)}")

bert_data = dict(lpr=np.concatenate(lpr_all), pr=np.concatenate(pr_all),
                 l2=np.concatenate(l2_all),   cos=np.concatenate(cos_all),
                 rank=np.concatenate(rank_all))
with open(f"{OUT_DIR}/bert_embed_corr_data.pkl", "wb") as f:
    pickle.dump(bert_data, f)
print(f"  saved {len(bert_data['lpr']):,} pairs")

del bert_model
torch.mps.empty_cache()

# ── Load all three and plot ───────────────────────────────────────────────────
datasets = {
    "bert-base-uncased":          (bert_data,                                       "bert"),
    "parameter-golf (baseline, λ=0)": (pickle.load(open(f"{OUT_DIR}/pg_baseline_embed_corr_data.pkl","rb")), "pg_baseline"),
    "GPT-2-medium":               (pickle.load(open(f"{OUT_DIR}/gpt2_embed_corr_data.pkl","rb")),      "gpt2"),
}

combos = [
    ("lpr", "cos", "log p_i − log p_top  (log-prob ratio)", "Cosine distance to top token"),
    ("lpr", "l2",  "log p_i − log p_top  (log-prob ratio)", "L2 distance to top token"),
    ("pr",  "cos", "p_i / p_top  (probability ratio)",      "Cosine distance to top token"),
    ("pr",  "l2",  "p_i / p_top  (probability ratio)",      "L2 distance to top token"),
]

for top_k_filter in [50, 100]:
    for title, (data, slug) in datasets.items():
        mask = data["rank"] <= top_k_filter
        n_pairs = mask.sum()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor("white")
        fig.suptitle(f"{title}  ·  top-{top_k_filter} tokens  ·  {N_CONTEXTS} FineWeb contexts  ·  {n_pairs:,} pairs",
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
            ax.fill_between(ctrs, mus - sigs, mus + sigs,
                            color="crimson", alpha=0.20, zorder=2, label="±1 std")
            ax.legend(fontsize=8, loc="upper left")

            rp, _ = stats.pearsonr(x, y); rs, _ = stats.spearmanr(x, y)
            ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.text(0.97, 0.97, f"Pearson r = {rp:.3f}\nSpearman ρ = {rs:.3f}",
                    transform=ax.transAxes, fontsize=8, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

        plt.tight_layout()
        out = f"{OUT_DIR}/{slug}_embed_corr_top{top_k_filter}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out}")
