#!/usr/bin/env python3
"""GPT-2-medium probability vs embedding distance — runs locally in parallel with pod."""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pickle

TOP_K       = 100
N_CONTEXTS  = 500
CONTEXT_LEN = 128
N_BINS      = 25
SEED        = 42
OUT_DATA    = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/gpt2_embed_corr_data.pkl"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")
torch.manual_seed(SEED); np.random.seed(SEED)

print("Loading GPT-2-medium…")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
model     = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(DEVICE).eval()
E      = model.transformer.wte.weight.detach().float().to(DEVICE)
E_norm = F.normalize(E, dim=-1)

print("Streaming FineWeb…")
raw_texts = []
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                  split="train", streaming=True).shuffle(seed=SEED, buffer_size=10_000)
for ex in ds:
    if len(raw_texts) >= N_CONTEXTS: break
    raw_texts.append(ex["text"])
print(f"  {len(raw_texts)} contexts")

lpr_all, pr_all, l2_all, cos_all = [], [], [], []
for i, text in enumerate(raw_texts):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONTEXT_LEN + 1)
    if enc["input_ids"].shape[1] < 16:
        continue
    ids = enc["input_ids"][:, :-1].to(DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0]

    log_p = torch.log_softmax(logits.float(), dim=-1)
    probs  = log_p.exp()
    top_p, top_i = probs.topk(TOP_K, dim=-1)
    top_lp        = log_p.gather(1, top_i)

    Ek      = E[top_i]
    Ek_norm = E_norm[top_i]
    l2       = (Ek[:, 1:] - Ek[:, 0:1]).norm(dim=-1)
    cos_dist = 1 - (Ek_norm[:, 1:] * Ek_norm[:, 0:1]).sum(dim=-1)

    lpr_all.append((top_lp[:, 1:] - top_lp[:, 0:1]).cpu().numpy().ravel())
    pr_all.append((top_p[:, 1:] / top_p[:, 0:1]).cpu().numpy().ravel())
    l2_all.append(l2.cpu().numpy().ravel())
    cos_all.append(cos_dist.cpu().numpy().ravel())
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(raw_texts)}")

data = dict(
    lpr=np.concatenate(lpr_all), pr=np.concatenate(pr_all),
    l2=np.concatenate(l2_all),   cos=np.concatenate(cos_all),
)
with open(OUT_DATA, "wb") as f:
    pickle.dump(data, f)
print(f"Saved data → {OUT_DATA}  ({len(data['lpr']):,} pairs)")
