#!/usr/bin/env python3
"""
Ablation: replace cosine embedding loss with Kronecker-delta distance.

With Kronecker delta, distance(i, ans) = 1 if i != ans, 0 otherwise.
The probability-weighted distance reduces to: sum_i p_i * 1(i != ans) = 1 - p[ans]

So the embedding loss term becomes:
  L_emb = mean(1 - softmax(logits)[ans])

This tests whether the geometric structure of the embedding space matters,
or if simply pushing probability mass toward the correct token is sufficient.
Compare against lama_results_eu.csv (cosine, gamma=8.0, lambda=1.0).

L = lambda_emb * (1 - p[ans])  +  gamma_unif * L_unif
  =       1.0  * (1 - p[ans])  +          8.0 * L_unif

Same seeds/splits/hyperparams as EU runs for a direct comparison.
Outputs: lama_results_kronecker.csv
"""

import copy
import glob
import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForMaskedLM
from tqdm import tqdm

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(SCRIPT_DIR, "data")
OUT_CSV        = os.path.join(SCRIPT_DIR, "lama_results_kronecker.csv")
MODEL_NAME     = "bert-base-uncased"
VAL_FRAC       = 0.1
TEST_FRAC      = 0.1
BATCH_SIZE     = 32
EVAL_EVERY     = 200
PATIENCE_EVALS = 6
LR             = 2e-5
MAX_LEN        = 128
GRAD_CLIP      = 1.0
UNIF_SUBSAMPLE = 2000
LAMBDA_EMB     = 1.0
GAMMA_UNIF     = 8.0
DEFAULT_SEEDS  = [42, 123, 456]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
args = parser.parse_args()
SEEDS = args.seeds

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
print(f"Device: {DEVICE}")
print(f"lambda_emb={LAMBDA_EMB}  gamma_unif={GAMMA_UNIF}  seeds={SEEDS}")

# ── Resume support ─────────────────────────────────────────────────────────────
if os.path.exists(OUT_CSV):
    existing   = pd.read_csv(OUT_CSV)
    results    = existing.to_dict("records")
    done_seeds = set(
        existing[existing["epoch"] == "final"]["seed"]
        .astype(int).tolist()
    )
else:
    results    = []
    done_seeds = set()
print(f"Already done seeds: {done_seeds}")

# ── Load data ──────────────────────────────────────────────────────────────────
print("\nLoading T-REx data...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

all_examples = []
for path in glob.glob(f"{DATA_DIR}/*.jsonl"):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("evidences"): continue
            sentence  = d["evidences"][0]["masked_sentence"]
            obj_label = d["obj_label"].strip()
            if "[MASK]" not in sentence: continue
            toks = tokenizer.tokenize(obj_label)
            if len(toks) != 1: continue
            answer_id = tokenizer.convert_tokens_to_ids(toks[0])
            all_examples.append({"sentence": sentence, "answer": obj_label,
                                  "answer_id": answer_id})
print(f"  {len(all_examples):,} single-token examples")

# ── Dataset ────────────────────────────────────────────────────────────────────
class LamaDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex  = self.examples[idx]
        enc = tokenizer(ex["sentence"], max_length=MAX_LEN,
                        padding="max_length", truncation=True, return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        am  = enc["attention_mask"].squeeze(0)
        mp  = (ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        mp  = mp[0] if len(mp) > 0 else torch.tensor(0)
        return {"input_ids": ids, "attention_mask": am, "mask_pos": mp,
                "answer_id": torch.tensor(ex["answer_id"], dtype=torch.long)}

def uniformity_loss(E, n=UNIF_SUBSAMPLE):
    idx     = torch.randperm(E.shape[0], device=E.device)[:n]
    e       = F.normalize(E[idx].float(), dim=-1)   # (n, d)
    sq_dist = 2.0 - 2.0 * (e @ e.T)                # (n, n)
    kernel  = sq_dist.mul(-2.0).exp()               # (n, n)
    off_diag = (kernel.sum() - n) / (n * (n - 1))
    return off_diag.log()

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    correct, total, ce_total = 0, 0, 0.0
    for batch in dl:
        ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
        mp  = batch["mask_pos"].to(DEVICE);  ans = batch["answer_id"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=am)
        B   = out.logits.shape[0]
        lat = out.logits[torch.arange(B), mp]
        ce_total += F.cross_entropy(lat, ans).item() * B
        correct  += (lat.argmax(-1) == ans).sum().item(); total += B
    return correct / total, ce_total / total

# ── Main loop ──────────────────────────────────────────────────────────────────
for seed in SEEDS:
    if seed in done_seeds:
        print(f"\nSkipping seed={seed} (already done)")
        continue

    print(f"\n{'='*60}\nSEED = {seed}\n{'='*60}")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    examples = all_examples.copy()
    random.shuffle(examples)
    n_test   = int(len(examples) * TEST_FRAC)
    n_val    = int(len(examples) * VAL_FRAC)
    test_ex  = examples[:n_test]
    val_ex   = examples[n_test:n_test + n_val]
    train_ex = examples[n_test + n_val:]
    print(f"  train={len(train_ex):,}  val={len(val_ex):,}  test={len(test_ex):,}")

    train_dl = DataLoader(LamaDataset(train_ex), batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(LamaDataset(val_ex),   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(LamaDataset(test_ex),  batch_size=BATCH_SIZE)

    torch.manual_seed(seed + hash(f"{LAMBDA_EMB}_{GAMMA_UNIF}") % 10000)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    E     = model.bert.embeddings.word_embeddings.weight

    best_acc, best_step, best_state = 0.0, 0, None
    patience_count = 0
    global_step    = 0
    stopped        = False
    sum_el, sum_ul, steps_since_eval = 0.0, 0.0, 0

    for epoch in range(1, 21):
        model.train()
        for batch in tqdm(train_dl, desc=f"  seed={seed} ep={epoch}", leave=False):
            ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            mp  = batch["mask_pos"].to(DEVICE);  ans = batch["answer_id"].to(DEVICE)
            out = model(input_ids=ids, attention_mask=am)
            B   = out.logits.shape[0]
            lat = out.logits[torch.arange(B), mp]

            p  = torch.softmax(lat.float(), dim=-1)
            el = (1 - p[torch.arange(B), ans]).mean()   # Kronecker delta distance
            ul = uniformity_loss(E)
            loss = LAMBDA_EMB * el + GAMMA_UNIF * ul

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); opt.zero_grad()

            global_step += 1; steps_since_eval += 1
            sum_el += el.item(); sum_ul += ul.item()

            if global_step % EVAL_EVERY == 0:
                avg_el = sum_el / steps_since_eval
                avg_ul = sum_ul / steps_since_eval
                sum_el = 0.0; sum_ul = 0.0; steps_since_eval = 0

                val_acc, val_ce = evaluate(model, val_dl)
                print(f"  step={global_step} ep={round(global_step/len(train_dl),2)} "
                      f"val={val_acc:.4f}  el={avg_el:.4f}  ul={avg_ul:.4f}")
                results.append({"seed": seed, "lambda_emb": LAMBDA_EMB,
                                "gamma_unif": GAMMA_UNIF, "step": global_step,
                                "epoch": round(global_step / len(train_dl), 2),
                                "val_acc": val_acc, "val_ce": val_ce,
                                "train_embed_loss": avg_el, "train_unif_loss": avg_ul,
                                "test_acc": None, "test_ce": None})
                pd.DataFrame(results).to_csv(OUT_CSV, index=False)

                if val_acc > best_acc:
                    best_acc = val_acc; best_step = global_step
                    best_state = copy.deepcopy(model.state_dict())
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= PATIENCE_EVALS:
                        print(f"  early stop at step={global_step}, best={best_step}")
                        stopped = True; break
                model.train()
        if stopped: break

    model.load_state_dict(best_state)
    test_acc, test_ce = evaluate(model, test_dl)
    print(f"\n  FINAL seed={seed}  test_acc={test_acc:.4f}")
    results.append({"seed": seed, "lambda_emb": LAMBDA_EMB,
                    "gamma_unif": GAMMA_UNIF, "step": best_step, "epoch": "final",
                    "val_acc": best_acc, "val_ce": None,
                    "train_embed_loss": None, "train_unif_loss": None,
                    "test_acc": test_acc, "test_ce": test_ce})
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    done_seeds.add(seed)

print(f"\nAll done → {OUT_CSV}")
