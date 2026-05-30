#!/usr/bin/env python3
"""
Top-up sweep: 10 seeds per lambda, cosine and L2 back-to-back.

Cosine: appends to lama_results_cosine_multi.csv (matching existing schema)
        Only runs missing (seed, lambda) pairs — λ=0.1/0.5/2.0/4.0 need 4 more seeds
L2:     writes to lama_results_l2_multi.csv (fresh file, same schema)
        Runs all 10 seeds × 6 lambdas

Resume-safe for both: skips any (seed, lambda) already in the target CSV.
"""

import copy, glob, json, os, random
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
MAX_STEPS        = 6000
EVAL_EVERY_STEPS = 200
LR               = 2e-5
MAX_LEN          = 128

LAMBDAS        = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0]
DEFAULT_SEEDS  = [42, 123, 456, 789, 1011, 2024, 3000, 4000, 5000, 6000]

COS_CSV = os.path.join(SCRIPT_DIR, "lama_results_cosine_multi.csv")
L2_CSV  = os.path.join(SCRIPT_DIR, "lama_results_l2_multi.csv")

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
args = parser.parse_args()
ALL_SEEDS = args.seeds

print(f"Device: {DEVICE}")
print(f"Seeds:  {ALL_SEEDS}")

# ── Loss functions ────────────────────────────────────────────────────────────
def embed_loss_cosine(lat, ans, E):
    p = torch.softmax(lat.float(), dim=-1)
    return (1 - F.cosine_similarity(p @ E, E[ans], dim=-1)).mean()

def embed_loss_l2(lat, ans, E):
    p = torch.softmax(lat.float(), dim=-1)
    return ((p @ E - E[ans]) ** 2).mean()

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading T-REx data...")
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


class LamaDataset(Dataset):
    def __init__(self, ex, tok, ml): self.ex=ex; self.tok=tok; self.ml=ml
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        e   = self.ex[i]
        enc = self.tok(e["sentence"], max_length=self.ml, padding="max_length",
                       truncation=True, return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        am  = enc["attention_mask"].squeeze(0)
        mp  = (ids == self.tok.mask_token_id).nonzero(as_tuple=True)[0]
        mp  = mp[0] if len(mp) > 0 else torch.tensor(0)
        return {"input_ids": ids, "attention_mask": am, "mask_pos": mp,
                "answer_id": torch.tensor(e["answer_id"], dtype=torch.long)}


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


def load_csv(path):
    if not os.path.exists(path):
        return [], set()
    df = pd.read_csv(path)
    records = df.to_dict("records")
    done = set(
        df[df["epoch"] == "final"]
        .apply(lambda r: (int(r["seed"]), str(r["lambda"])), axis=1)
        .tolist()
    )
    return records, done


def run_sweep(loss_fn, csv_path, label, lambdas=None):
    if lambdas is None:
        lambdas = LAMBDAS
    results, done_pairs = load_csv(csv_path)
    total_todo = sum(1 for s in ALL_SEEDS for lam in lambdas
                     if (s, str(lam)) not in done_pairs)
    print(f"\n{'='*60}\n{label}  ({total_todo} runs remaining)  →  {csv_path}\n{'='*60}")

    for seed in ALL_SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        examples = all_examples.copy(); random.shuffle(examples)
        n_test   = int(len(examples) * TEST_FRAC)
        n_val    = int(len(examples) * VAL_FRAC)
        test_ex  = examples[:n_test]
        val_ex   = examples[n_test:n_test + n_val]
        train_ex = examples[n_test + n_val:]

        train_dl = DataLoader(LamaDataset(train_ex, tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=True)
        val_dl   = DataLoader(LamaDataset(val_ex,   tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
        test_dl  = DataLoader(LamaDataset(test_ex,  tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

        for lam in lambdas:
            if (seed, str(lam)) in done_pairs:
                continue
            print(f"\n  λ={lam}  seed={seed}  [{label}]")
            torch.manual_seed(seed + hash(str(lam)) % 10000)

            model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
            opt   = torch.optim.AdamW(model.parameters(), lr=LR)
            E     = model.bert.embeddings.word_embeddings.weight

            best_acc, best_step, best_state = 0.0, 0, None
            global_step = 0; finished = False
            sum_ce = 0.0; sum_el = 0.0; steps_since_eval = 0

            while not finished:
                model.train()
                for batch in tqdm(train_dl, desc=f"    λ={lam} s={seed}", leave=False):
                    ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
                    mp  = batch["mask_pos"].to(DEVICE);  ans = batch["answer_id"].to(DEVICE)
                    out = model(input_ids=ids, attention_mask=am)
                    B   = out.logits.shape[0]
                    lat = out.logits[torch.arange(B), mp]
                    ce  = F.cross_entropy(lat, ans)
                    el  = loss_fn(lat, ans, E) if lam > 0 else torch.tensor(0.0, device=DEVICE)
                    (ce + lam * el).backward()
                    opt.step(); opt.zero_grad()

                    global_step += 1; steps_since_eval += 1
                    sum_ce += ce.item(); sum_el += el.item()

                    if global_step % EVAL_EVERY_STEPS == 0:
                        avg_ce = sum_ce / steps_since_eval
                        avg_el = sum_el / steps_since_eval
                        sum_ce = 0.0; sum_el = 0.0; steps_since_eval = 0
                        val_acc, val_ce = evaluate(model, val_dl)
                        frac = (lam * avg_el) / (avg_ce + lam * avg_el + 1e-9)
                        print(f"    step={global_step} val={val_acc:.4f} "
                              f"ce={avg_ce:.4f} el={avg_el:.4f} frac={frac:.2%}")
                        results.append({
                            "seed": seed, "lambda": lam, "step": global_step,
                            "epoch": round(global_step / len(train_dl), 2),
                            "val_acc": val_acc, "val_ce": val_ce,
                            "train_ce": avg_ce, "train_embed_loss": avg_el,
                            "embed_frac": round(frac, 4),
                            "test_acc": None, "test_ce": None,
                        })
                        pd.DataFrame(results).to_csv(csv_path, index=False)
                        if val_acc > best_acc:
                            best_acc = val_acc; best_step = global_step
                            best_state = copy.deepcopy(model.state_dict())
                        model.train()

                    if global_step >= MAX_STEPS:
                        finished = True; break

            model.load_state_dict(best_state)
            test_acc, test_ce = evaluate(model, test_dl)
            print(f"    FINAL  test_acc={test_acc:.4f}  best_step={best_step}")
            results.append({
                "seed": seed, "lambda": lam, "step": best_step, "epoch": "final",
                "val_acc": best_acc, "val_ce": None,
                "train_ce": None, "train_embed_loss": None, "embed_frac": None,
                "test_acc": test_acc, "test_ce": test_ce,
            })
            pd.DataFrame(results).to_csv(csv_path, index=False)
            done_pairs.add((seed, str(lam)))

    print(f"Done → {csv_path}")


run_sweep(embed_loss_cosine, COS_CSV, "cosine")

# λ=0.0 is identical regardless of loss type (term is multiplied by 0), skip for L2
L2_LAMBDAS = [l for l in LAMBDAS if l != 0.0] + [8.0]
run_sweep(embed_loss_l2, L2_CSV, "l2", lambdas=L2_LAMBDAS)
print("\nAll sweeps complete.")
