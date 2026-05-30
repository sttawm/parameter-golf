#!/usr/bin/env python3
"""
Same as lama_finetune.py but uses cosine embedding loss instead of L2.

L_embed_cos = mean(1 - cosine(e_hat, e_gt))

where e_hat = softmax(logits) @ E  (expected embedding)
      e_gt  = E[answer_id]          (ground truth embedding)

Runs the same lambda sweep: [4.0, 2.0, 1.0, 0.5, 0.1, 0.0]
Outputs: lama_results_cosine.csv
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

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(SCRIPT_DIR, "data")
MODEL_NAME       = "bert-base-uncased"
VAL_FRAC         = 0.1
TEST_FRAC        = 0.1
BATCH_SIZE       = 32
MAX_EPOCHS       = 20
EVAL_EVERY_STEPS = 200
PATIENCE_EVALS   = 6
LR               = 2e-5
MAX_LEN          = 128
LAMBDAS          = [4.0, 2.0, 1.0, 0.5, 0.1, 0.0]
SEED             = 42
OUT_CSV          = "lama_results_cosine.csv"

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading T-REx data...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

examples = []
for path in glob.glob(f"{DATA_DIR}/*.jsonl"):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("evidences"):
                continue
            sentence  = d["evidences"][0]["masked_sentence"]
            obj_label = d["obj_label"].strip()
            if "[MASK]" not in sentence:
                continue
            toks = tokenizer.tokenize(obj_label)
            if len(toks) != 1:
                continue
            answer_id = tokenizer.convert_tokens_to_ids(toks[0])
            examples.append({"sentence": sentence, "answer": obj_label,
                              "answer_id": answer_id})

print(f"  {len(examples):,} single-token examples")

random.shuffle(examples)
n_test   = int(len(examples) * TEST_FRAC)
n_val    = int(len(examples) * VAL_FRAC)
test_ex  = examples[:n_test]
val_ex   = examples[n_test:n_test + n_val]
train_ex = examples[n_test + n_val:]
print(f"  train={len(train_ex):,}  val={len(val_ex):,}  test={len(test_ex):,}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class LamaDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex  = self.examples[idx]
        enc = self.tokenizer(
            ex["sentence"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        mask_pos       = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        mask_pos       = mask_pos[0] if len(mask_pos) > 0 else torch.tensor(0)
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "mask_pos":       mask_pos,
            "answer_id":      torch.tensor(ex["answer_id"], dtype=torch.long),
        }

train_ds = LamaDataset(train_ex, tokenizer, MAX_LEN)
val_ds   = LamaDataset(val_ex,   tokenizer, MAX_LEN)
test_ds  = LamaDataset(test_ex,  tokenizer, MAX_LEN)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ── Embedding losses ───────────────────────────────────────────────────────────
def embed_loss_cosine(logits_at_mask, answer_ids, E):
    """
    logits_at_mask : (B, V)
    answer_ids     : (B,)
    E              : (V, D)
    """
    p      = torch.softmax(logits_at_mask.float(), dim=-1)   # (B, V)
    e_hat  = p @ E                                           # (B, D)
    e_gt   = E[answer_ids]                                   # (B, D)
    return (1 - F.cosine_similarity(e_hat, e_gt, dim=-1)).mean()

# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    correct, total, ce_total = 0, 0, 0.0
    for batch in dl:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        mask_pos       = batch["mask_pos"].to(DEVICE)
        answer_ids     = batch["answer_id"].to(DEVICE)

        out    = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        B      = logits.shape[0]
        logits_at_mask = logits[torch.arange(B), mask_pos]

        ce = F.cross_entropy(logits_at_mask, answer_ids, reduction="mean")
        ce_total += ce.item() * B
        preds    = logits_at_mask.argmax(dim=-1)
        correct += (preds == answer_ids).sum().item()
        total   += B

    return correct / total, ce_total / total

# ── Resume from existing results ──────────────────────────────────────────────
if os.path.exists(OUT_CSV):
    existing     = pd.read_csv(OUT_CSV)
    results      = existing.to_dict("records")
    done_lambdas = set(existing[existing["epoch"] == "final"]["lambda"].astype(str).tolist())
    print(f"Resuming — found results for: {done_lambdas or 'none'}")
else:
    results      = []
    done_lambdas = set()

# ── Zero-shot ─────────────────────────────────────────────────────────────────
print("\nLoading BERT...")

if "zero-shot" not in done_lambdas:
    model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    acc, ce = evaluate(model, test_dl)
    print(f"Zero-shot  →  test acc={acc:.4f}  test_ce={ce:.4f}")
    results.append({"lambda": "zero-shot", "epoch": 0, "val_acc": None, "val_ce": None,
                    "train_ce": None, "train_embed_loss": None, "embed_frac": None,
                    "test_acc": acc, "test_ce": ce, "step": None})
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
else:
    print("Zero-shot already done, skipping.")

# ── Fine-tuning ───────────────────────────────────────────────────────────────
for lam in LAMBDAS:
    if str(lam) in done_lambdas:
        print(f"\nSkipping λ={lam} (already complete)")
        continue

    print(f"\nFine-tuning λ={lam}  [cosine loss]...")
    model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    E     = model.bert.embeddings.word_embeddings.weight

    best_acc       = 0.0
    best_step      = 0
    best_state     = None
    patience_count = 0
    global_step    = 0
    stopped        = False
    sum_ce         = 0.0
    sum_el         = 0.0
    steps_since_eval = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in tqdm(train_dl, desc=f"  λ={lam} epoch={epoch}", leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            mask_pos       = batch["mask_pos"].to(DEVICE)
            answer_ids     = batch["answer_id"].to(DEVICE)

            out    = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            B      = logits.shape[0]
            logits_at_mask = logits[torch.arange(B), mask_pos]

            ce   = F.cross_entropy(logits_at_mask, answer_ids)
            el   = embed_loss_cosine(logits_at_mask, answer_ids, E) if lam > 0 else torch.tensor(0.0)
            loss = ce + lam * el

            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step      += 1
            steps_since_eval += 1
            sum_ce           += ce.item()
            sum_el           += el.item()

            if global_step % EVAL_EVERY_STEPS == 0:
                avg_ce = sum_ce / steps_since_eval
                avg_el = sum_el / steps_since_eval
                sum_ce = 0.0; sum_el = 0.0; steps_since_eval = 0

                val_acc, val_ce = evaluate(model, val_dl)
                frac = (lam * avg_el) / (avg_ce + lam * avg_el) if (avg_ce + lam * avg_el) > 0 else 0
                print(f"  λ={lam}  step={global_step}  epoch={round(global_step/len(train_dl),2)}"
                      f"  val acc={val_acc:.4f}  train_ce={avg_ce:.4f}"
                      f"  train_el={avg_el:.4f}  el_frac={frac:.2%}")
                results.append({"lambda": lam, "step": global_step,
                                "epoch": round(global_step / len(train_dl), 2),
                                "val_acc": val_acc, "val_ce": val_ce,
                                "train_ce": avg_ce, "train_embed_loss": avg_el,
                                "embed_frac": round(frac, 4),
                                "test_acc": None, "test_ce": None})
                pd.DataFrame(results).to_csv(OUT_CSV, index=False)

                if val_acc > best_acc:
                    best_acc   = val_acc
                    best_step  = global_step
                    best_state = copy.deepcopy(model.state_dict())
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= PATIENCE_EVALS:
                        print(f"  early stop at step {global_step} (best step={best_step})")
                        stopped = True
                        break
                model.train()

        if stopped:
            break

    model.load_state_dict(best_state)
    test_acc, test_ce = evaluate(model, test_dl)
    print(f"  λ={lam}  FINAL (best step={best_step})  test acc={test_acc:.4f}  test_ce={test_ce:.4f}")
    results.append({"lambda": lam, "step": best_step, "epoch": "final",
                    "val_acc": best_acc, "val_ce": None,
                    "train_ce": None, "train_embed_loss": None, "embed_frac": None,
                    "test_acc": test_acc, "test_ce": test_ce})
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)

# ── Done ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(OUT_CSV)
print(f"\nFinal results (cosine) → {OUT_CSV}")
finals = df[df["epoch"] == "final"][["lambda", "test_acc"]].sort_values("lambda")
print(finals.to_string(index=False))
