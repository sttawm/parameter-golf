#!/usr/bin/env python3
"""
Cosine Embedding Loss + Uniformity Loss on LAMA T-REx (no CE).

  L = lambda_emb * L_emb  +  gamma_unif * L_unif

L_emb:  1 - cos(sum_v p_v * E_v, E_answer)       (per-example, batched)
L_unif: log mean exp(-2 * ||e_i - e_j||^2)        (Wang & Isola 2020)
        computed on a random subsample of vocab each step (full V×V too large)

Defaults match the ceeu result: lambda_emb=1.0, gamma_unif=2.0.
Writes to lama_results_ceeu.csv.
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
GRAD_CLIP        = 1.0
UNIF_SUBSAMPLE   = 2000   # vocab tokens sampled per step for uniformity loss

DEFAULT_SEEDS      = [42, 123, 456]
DEFAULT_LAMBDA_EMB = 1.0
DEFAULT_GAMMA_UNIF = 2.0
CSV_PATH           = os.path.join(SCRIPT_DIR, "lama_results_eu.csv")

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

parser = argparse.ArgumentParser()
parser.add_argument("--seeds",      type=int,   nargs="+", default=DEFAULT_SEEDS)
parser.add_argument("--lambda-emb", type=float, default=DEFAULT_LAMBDA_EMB)
parser.add_argument("--gamma-unif", type=float, default=DEFAULT_GAMMA_UNIF)
args = parser.parse_args()
ALL_SEEDS  = args.seeds
LAMBDA_EMB = args.lambda_emb
GAMMA_UNIF = args.gamma_unif

print(f"Device:      {DEVICE}")
print(f"Seeds:       {ALL_SEEDS}")
print(f"lambda_emb:  {LAMBDA_EMB}")
print(f"gamma_unif:  {GAMMA_UNIF}")


def uniformity_loss(E_weight, n=UNIF_SUBSAMPLE):
    """Wang & Isola (2020) uniformity loss on a random subsample of vocab."""
    idx = torch.randperm(E_weight.shape[0], device=E_weight.device)[:n]
    e   = F.normalize(E_weight[idx].float(), dim=-1)   # (n, d)
    sq_dist = 2.0 - 2.0 * (e @ e.T)                   # (n, n)
    kernel  = sq_dist.mul(-2.0).exp()                  # (n, n)
    off_diag = (kernel.sum() - n) / (n * (n - 1))
    return off_diag.log()


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
        .apply(lambda r: (int(r["seed"]), float(r["lambda_emb"]), float(r["gamma_unif"])), axis=1)
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
total_todo = sum(1 for s in ALL_SEEDS
                 if (s, LAMBDA_EMB, GAMMA_UNIF) not in done_pairs)
print(f"\n{total_todo} runs remaining → {CSV_PATH}\n")

for seed in ALL_SEEDS:
    if (seed, LAMBDA_EMB, GAMMA_UNIF) in done_pairs:
        print(f"  Skipping seed={seed} (already done)")
        continue

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

    print(f"\n  λ_emb={LAMBDA_EMB}  γ_unif={GAMMA_UNIF}  seed={seed}")
    torch.manual_seed(seed + hash(f"{LAMBDA_EMB}_{GAMMA_UNIF}") % 10000)

    model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    E     = model.bert.embeddings.word_embeddings.weight

    best_acc, best_step, best_state = 0.0, 0, None
    global_step = 0; finished = False
    sum_ce = 0.0; sum_el = 0.0; sum_ul = 0.0; steps_since_eval = 0

    while not finished:
        model.train()
        for batch in tqdm(train_dl, desc=f"  seed={seed}", leave=False):
            ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            mp  = batch["mask_pos"].to(DEVICE);  ans = batch["answer_id"].to(DEVICE)
            out = model(input_ids=ids, attention_mask=am)
            B   = out.logits.shape[0]
            lat = out.logits[torch.arange(B), mp]

            p  = torch.softmax(lat.float(), dim=-1)
            el = (1 - F.cosine_similarity(p @ E, E[ans], dim=-1)).mean()
            ul = uniformity_loss(E)

            loss = LAMBDA_EMB * el + GAMMA_UNIF * ul
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); opt.zero_grad()

            global_step += 1; steps_since_eval += 1
            sum_ce += F.cross_entropy(lat.detach(), ans).item()  # track CE for monitoring only
            sum_el += el.item(); sum_ul += ul.item()

            if global_step % EVAL_EVERY_STEPS == 0:
                avg_ce = sum_ce / steps_since_eval
                avg_el = sum_el / steps_since_eval
                avg_ul = sum_ul / steps_since_eval
                sum_ce = sum_el = sum_ul = 0.0; steps_since_eval = 0
                val_acc, val_ce = evaluate(model, val_dl)
                print(f"    step={global_step} val={val_acc:.4f} "
                      f"ce={avg_ce:.4f} el={avg_el:.4f} ul={avg_ul:.4f}")
                results.append({
                    "seed": seed, "lambda_emb": LAMBDA_EMB, "gamma_unif": GAMMA_UNIF,
                    "step": global_step,
                    "epoch": round(global_step / len(train_dl), 2),
                    "val_acc": val_acc, "val_ce": val_ce,
                    "train_ce": avg_ce, "train_embed_loss": avg_el, "train_unif_loss": avg_ul,
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
        "seed": seed, "lambda_emb": LAMBDA_EMB, "gamma_unif": GAMMA_UNIF,
        "step": best_step, "epoch": "final",
        "val_acc": best_acc, "val_ce": None,
        "train_ce": None, "train_embed_loss": None, "train_unif_loss": None,
        "test_acc": test_acc, "test_ce": test_ce,
    })
    pd.DataFrame(results).to_csv(CSV_PATH, index=False)
    done_pairs.add((seed, LAMBDA_EMB, GAMMA_UNIF))

print(f"\nDone → {CSV_PATH}")
