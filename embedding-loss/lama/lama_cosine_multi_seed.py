#!/usr/bin/env python3
"""
Cosine embedding loss sweep over multiple seeds.

Seeds 123 and 456 only -- seed 42 already exists in lama_results_cosine.csv
and is merged in at startup. Each seed gets its own train/val/test split
(shuffled with that seed) so results are truly independent.

Outputs: lama_results_cosine_multi.csv  (seed column added)
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
NEW_SEEDS        = [123, 456, 789, 1011, 2024]   # seed 42 already done
SEED_42_CSV      = "lama_results_cosine.csv"
OUT_CSV          = "lama_results_cosine_multi.csv"

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

# ── Merge seed=42 results into multi CSV if not already done ──────────────────
if not os.path.exists(OUT_CSV) and os.path.exists(SEED_42_CSV):
    df42 = pd.read_csv(SEED_42_CSV)
    df42["seed"] = 42
    df42.to_csv(OUT_CSV, index=False)
    print(f"Initialized {OUT_CSV} from seed=42 results ({len(df42)} rows)")
elif not os.path.exists(OUT_CSV):
    pd.DataFrame().to_csv(OUT_CSV, index=False)

# ── Load existing results and find done (seed, lambda) pairs ──────────────────
existing     = pd.read_csv(OUT_CSV)
results      = existing.to_dict("records")
done_pairs   = set(
    existing[existing["epoch"] == "final"]
    .apply(lambda r: (int(r["seed"]), str(r["lambda"])), axis=1)
    .tolist()
)
print(f"Already done: {done_pairs}")

# ── Load raw examples once (order doesn't matter, shuffle is per-seed) ────────
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

# ── Dataset / helpers ─────────────────────────────────────────────────────────
class LamaDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.examples = examples; self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex  = self.examples[idx]
        enc = self.tokenizer(ex["sentence"], max_length=self.max_len,
                             padding="max_length", truncation=True, return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        am  = enc["attention_mask"].squeeze(0)
        mp  = (ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        mp  = mp[0] if len(mp) > 0 else torch.tensor(0)
        return {"input_ids": ids, "attention_mask": am, "mask_pos": mp,
                "answer_id": torch.tensor(ex["answer_id"], dtype=torch.long)}

def embed_loss_cosine(logits_at_mask, answer_ids, E):
    p     = torch.softmax(logits_at_mask.float(), dim=-1)
    e_hat = p @ E
    e_gt  = E[answer_ids]
    return (1 - F.cosine_similarity(e_hat, e_gt, dim=-1)).mean()

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    correct, total, ce_total = 0, 0, 0.0
    for batch in dl:
        ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
        mp  = batch["mask_pos"].to(DEVICE);  ans = batch["answer_id"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=am)
        B   = out.logits.shape[0]
        lam = out.logits[torch.arange(B), mp]
        ce_total += F.cross_entropy(lam, ans).item() * B
        correct  += (lam.argmax(-1) == ans).sum().item(); total += B
    return correct / total, ce_total / total

# ── Main loop: seed → lambda ──────────────────────────────────────────────────
for seed in NEW_SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED = {seed}")
    print(f"{'='*60}")

    # Seed everything and create this seed's splits
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    examples = all_examples.copy()
    random.shuffle(examples)
    n_test   = int(len(examples) * TEST_FRAC)
    n_val    = int(len(examples) * VAL_FRAC)
    test_ex  = examples[:n_test]
    val_ex   = examples[n_test:n_test + n_val]
    train_ex = examples[n_test + n_val:]
    print(f"  train={len(train_ex):,}  val={len(val_ex):,}  test={len(test_ex):,}")

    train_dl = DataLoader(LamaDataset(train_ex, tokenizer, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(LamaDataset(val_ex,  tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
    test_dl  = DataLoader(LamaDataset(test_ex, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

    # Zero-shot for this seed
    if (seed, "zero-shot") not in done_pairs:
        model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
        acc, ce = evaluate(model, test_dl)
        print(f"  Zero-shot  →  test acc={acc:.4f}  test_ce={ce:.4f}")
        results.append({"seed": seed, "lambda": "zero-shot", "epoch": 0,
                        "val_acc": None, "val_ce": None,
                        "train_ce": None, "train_embed_loss": None, "embed_frac": None,
                        "test_acc": acc, "test_ce": ce, "step": None})
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)
        done_pairs.add((seed, "zero-shot"))

    for lam in LAMBDAS:
        if (seed, str(lam)) in done_pairs:
            print(f"\n  Skipping λ={lam} seed={seed} (already done)")
            continue

        print(f"\n  Fine-tuning λ={lam}  seed={seed}  [cosine]...")

        # Re-seed model init for reproducibility within this (seed, lam) run
        torch.manual_seed(seed + hash(lam) % 10000)

        model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=LR)
        E     = model.bert.embeddings.word_embeddings.weight

        best_acc, best_step, best_state = 0.0, 0, None
        patience_count = 0
        global_step    = 0
        stopped        = False
        sum_ce, sum_el, steps_since_eval = 0.0, 0.0, 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for batch in tqdm(train_dl, desc=f"    λ={lam} s={seed} ep={epoch}", leave=False):
                ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
                mp  = batch["mask_pos"].to(DEVICE);  ans = batch["answer_id"].to(DEVICE)
                out = model(input_ids=ids, attention_mask=am)
                B   = out.logits.shape[0]
                lam_l = out.logits[torch.arange(B), mp]
                ce    = F.cross_entropy(lam_l, ans)
                el    = embed_loss_cosine(lam_l, ans, E) if lam > 0 else torch.tensor(0.0)
                loss  = ce + lam * el

                opt.zero_grad(); loss.backward(); opt.step()
                global_step += 1; steps_since_eval += 1
                sum_ce += ce.item(); sum_el += el.item()

                if global_step % EVAL_EVERY_STEPS == 0:
                    avg_ce = sum_ce / steps_since_eval
                    avg_el = sum_el / steps_since_eval
                    sum_ce = 0.0; sum_el = 0.0; steps_since_eval = 0

                    val_acc, val_ce = evaluate(model, val_dl)
                    frac = (lam * avg_el) / (avg_ce + lam * avg_el) if (avg_ce + lam * avg_el) > 0 else 0
                    print(f"    λ={lam} s={seed} step={global_step} "
                          f"epoch={round(global_step/len(train_dl),2)} "
                          f"val={val_acc:.4f} ce={avg_ce:.4f} el={avg_el:.4f} frac={frac:.2%}")
                    results.append({"seed": seed, "lambda": lam, "step": global_step,
                                    "epoch": round(global_step / len(train_dl), 2),
                                    "val_acc": val_acc, "val_ce": val_ce,
                                    "train_ce": avg_ce, "train_embed_loss": avg_el,
                                    "embed_frac": round(frac, 4),
                                    "test_acc": None, "test_ce": None})
                    pd.DataFrame(results).to_csv(OUT_CSV, index=False)

                    if val_acc > best_acc:
                        best_acc = val_acc; best_step = global_step
                        best_state = copy.deepcopy(model.state_dict())
                        patience_count = 0
                    else:
                        patience_count += 1
                        if patience_count >= PATIENCE_EVALS:
                            print(f"    early stop step={global_step} best={best_step}")
                            stopped = True; break
                    model.train()
            if stopped: break

        model.load_state_dict(best_state)
        test_acc, test_ce = evaluate(model, test_dl)
        print(f"    λ={lam} s={seed} FINAL test acc={test_acc:.4f}")
        results.append({"seed": seed, "lambda": lam, "step": best_step, "epoch": "final",
                        "val_acc": best_acc, "val_ce": None,
                        "train_ce": None, "train_embed_loss": None, "embed_frac": None,
                        "test_acc": test_acc, "test_ce": test_ce})
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)
        done_pairs.add((seed, str(lam)))

print(f"\nAll done → {OUT_CSV}")
