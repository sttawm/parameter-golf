#!/usr/bin/env python3
"""
High-lambda cosine embedding loss sweep: λ=20 (with gradient clipping).

Motivation: the embedding loss gradient is ~5% the size of CE at λ=1,
so λ≈20 is needed to equalize.  Previous sweep topped out at λ=4.

Writes to lama_results_cosine_lam20.csv (same schema as lama_results_cosine_multi.csv).
Resume-safe: skips any (seed, lambda) already present.
"""

import copy, glob, json, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForMaskedLM
from tqdm import tqdm
import argparse

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
GRAD_CLIP        = 1.0   # standard BERT fine-tuning clip

LAMBDAS       = [20.0]
DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 2024, 3000, 4000, 5000, 6000]
CSV_PATH      = os.path.join(SCRIPT_DIR, "lama_results_cosine_lam20.csv")

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
args = parser.parse_args()
ALL_SEEDS = args.seeds

print(f"Device: {DEVICE}")
print(f"Seeds:  {ALL_SEEDS}")
print(f"Lambdas: {LAMBDAS}")


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


def load_csv():
    if not os.path.exists(CSV_PATH):
        return [], set()
    df = pd.read_csv(CSV_PATH)
    records = df.to_dict("records")
    done = set(
        df[df["epoch"] == "final"]
        .apply(lambda r: (int(r["seed"]), str(r["lambda"])), axis=1)
        .tolist()
    )
    return records, done


print("Loading tokenizer and data...")
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

results, done_pairs = load_csv()
total_todo = sum(1 for s in ALL_SEEDS for lam in LAMBDAS
                 if (s, str(lam)) not in done_pairs)
print(f"\n{total_todo} runs remaining → {CSV_PATH}\n")

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

    for lam in LAMBDAS:
        if (seed, str(lam)) in done_pairs:
            print(f"  Skipping λ={lam} seed={seed} (already done)")
            continue
        print(f"\n  λ={lam}  seed={seed}")
        torch.manual_seed(seed + hash(str(lam)) % 10000)

        model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=LR)
        E     = model.bert.embeddings.word_embeddings.weight

        best_acc, best_step, best_state = 0.0, 0, None
        global_step = 0; finished = False
        sum_ce = 0.0; sum_el = 0.0; steps_since_eval = 0

        while not finished:
            model.train()
            for batch in tqdm(train_dl, desc=f"  λ={lam} s={seed}", leave=False):
                ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
                mp  = batch["mask_pos"].to(DEVICE);  ans = batch["answer_id"].to(DEVICE)
                out = model(input_ids=ids, attention_mask=am)
                B   = out.logits.shape[0]
                lat = out.logits[torch.arange(B), mp]
                ce  = F.cross_entropy(lat, ans)
                p   = torch.softmax(lat.float(), dim=-1)
                el  = (1 - F.cosine_similarity(p @ E, E[ans], dim=-1)).mean()
                loss = ce + lam * el
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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
                    pd.DataFrame(results).to_csv(CSV_PATH, index=False)
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
        pd.DataFrame(results).to_csv(CSV_PATH, index=False)
        done_pairs.add((seed, str(lam)))

print(f"\nDone → {CSV_PATH}")
