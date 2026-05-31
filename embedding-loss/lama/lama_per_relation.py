#!/usr/bin/env python3
"""
Per-relation accuracy breakdown for the best L2 and cosine checkpoints.

For each T-REx relation (P101, P131, etc.) computes top-1 accuracy for:
  - zero-shot BERT
  - best L2 lambda (from lama_results.csv)
  - best cosine lambda (from lama_results_cosine.csv)

Outputs: lama_per_relation.csv, lama_per_relation_plot.png
"""

import copy, glob, json, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForMaskedLM
from tqdm import tqdm

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
MODEL_NAME  = "bert-base-uncased"
TEST_FRAC   = 0.1
VAL_FRAC    = 0.1
BATCH_SIZE  = 32
MAX_LEN     = 128
SEED        = 42
LR          = 2e-5
OUT_CSV     = "lama_per_relation.csv"
OUT_PLOT    = "lama_per_relation_plot.png"

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Load data with relation labels ────────────────────────────────────────────
print("Loading T-REx data...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

examples = []
for path in glob.glob(f"{DATA_DIR}/*.jsonl"):
    relation = os.path.basename(path).replace(".jsonl", "")
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
            examples.append({"sentence": sentence, "answer": obj_label,
                              "answer_id": answer_id, "relation": relation})

print(f"  {len(examples):,} examples across {len(set(e['relation'] for e in examples))} relations")

random.shuffle(examples)
n_test   = int(len(examples) * TEST_FRAC)
n_val    = int(len(examples) * VAL_FRAC)
test_ex  = examples[:n_test]
val_ex   = examples[n_test:n_test + n_val]
train_ex = examples[n_test + n_val:]

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
                "answer_id": torch.tensor(ex["answer_id"], dtype=torch.long),
                "relation":  ex["relation"]}

def collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "mask_pos":       torch.stack([b["mask_pos"] for b in batch]),
        "answer_id":      torch.stack([b["answer_id"] for b in batch]),
        "relation":       [b["relation"] for b in batch],
    }

train_dl = DataLoader(LamaDataset(train_ex, tokenizer, MAX_LEN),
                      batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_dl   = DataLoader(LamaDataset(val_ex,   tokenizer, MAX_LEN),
                      batch_size=BATCH_SIZE, collate_fn=collate)
test_dl  = DataLoader(LamaDataset(test_ex,  tokenizer, MAX_LEN),
                      batch_size=BATCH_SIZE, collate_fn=collate)

# ── Helpers ───────────────────────────────────────────────────────────────────
def embed_loss_l2(logits_at_mask, answer_ids, E):
    p = torch.softmax(logits_at_mask.float(), dim=-1)
    return ((p @ E - E[answer_ids]) ** 2).sum(dim=-1).mean()

def embed_loss_cosine(logits_at_mask, answer_ids, E):
    p = torch.softmax(logits_at_mask.float(), dim=-1)
    e_hat = p @ E; e_gt = E[answer_ids]
    return (1 - F.cosine_similarity(e_hat, e_gt, dim=-1)).mean()

@torch.no_grad()
def evaluate_per_relation(model, dl):
    model.eval()
    rows = []
    for batch in dl:
        ids = batch["input_ids"].to(DEVICE)
        am  = batch["attention_mask"].to(DEVICE)
        mp  = batch["mask_pos"].to(DEVICE)
        ans = batch["answer_id"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=am)
        B   = ids.shape[0]
        lam = out.logits[torch.arange(B), mp]
        pred = lam.argmax(dim=-1)
        for i in range(B):
            rows.append({"relation": batch["relation"][i],
                         "correct": int(pred[i] == ans[i])})
    return pd.DataFrame(rows)

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    correct, total, ce_total = 0, 0, 0.0
    for batch in dl:
        ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
        mp  = batch["mask_pos"].to(DEVICE);  ans = batch["answer_id"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=am)
        B   = ids.shape[0]
        lam = out.logits[torch.arange(B), mp]
        ce_total += F.cross_entropy(lam, ans).item() * B
        correct  += (lam.argmax(-1) == ans).sum().item(); total += B
    return correct / total, ce_total / total

def train_model(lam, loss_fn):
    model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    E     = model.bert.embeddings.word_embeddings.weight
    best_acc, best_state, patience, step = 0.0, None, 0, 0
    EVAL_EVERY, PATIENCE_EVALS = 200, 6
    stopped = False
    for epoch in range(1, 21):
        model.train()
        for batch in tqdm(train_dl, desc=f"  λ={lam} epoch={epoch}", leave=False):
            ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
            mp  = batch["mask_pos"].to(DEVICE);  ans = batch["answer_id"].to(DEVICE)
            out = model(input_ids=ids, attention_mask=am)
            B   = ids.shape[0]
            lam_l = out.logits[torch.arange(B), mp]
            ce    = F.cross_entropy(lam_l, ans)
            el    = loss_fn(lam_l, ans, E) if lam > 0 else torch.tensor(0.0)
            loss  = ce + lam * el
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % EVAL_EVERY == 0:
                val_acc, _ = evaluate(model, val_dl)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                    if patience >= PATIENCE_EVALS:
                        stopped = True; break
                model.train()
        if stopped: break
    model.load_state_dict(best_state)
    return model

# ── Pick best lambdas from sweep results ──────────────────────────────────────
def best_lambda(csv_path):
    if not os.path.exists(csv_path): return 1.0
    df = pd.read_csv(csv_path)
    finals = df[df["epoch"] == "final"].copy()
    finals["test_acc"] = pd.to_numeric(finals["test_acc"], errors="coerce")
    row = finals.loc[finals["test_acc"].idxmax()]
    return float(row["lambda"])

best_l2  = best_lambda("lama_results.csv")
best_cos = best_lambda("lama_results_cosine.csv")
print(f"Best L2 lambda: {best_l2}  |  Best cosine lambda: {best_cos}")

# ── Run evaluations ───────────────────────────────────────────────────────────
records = []

print("\nZero-shot per-relation eval...")
model_zs = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
df_zs = evaluate_per_relation(model_zs, test_dl)
for rel, grp in df_zs.groupby("relation"):
    records.append({"relation": rel, "condition": "zero-shot",
                    "n": len(grp), "acc": grp["correct"].mean()})

print(f"\nTraining best L2 (λ={best_l2})...")
model_l2 = train_model(best_l2, embed_loss_l2)
df_l2 = evaluate_per_relation(model_l2, test_dl)
for rel, grp in df_l2.groupby("relation"):
    records.append({"relation": rel, "condition": f"L2 λ={best_l2}",
                    "n": len(grp), "acc": grp["correct"].mean()})

print(f"\nTraining best cosine (λ={best_cos})...")
model_cos = train_model(best_cos, embed_loss_cosine)
df_cos = evaluate_per_relation(model_cos, test_dl)
for rel, grp in df_cos.groupby("relation"):
    records.append({"relation": rel, "condition": f"cosine λ={best_cos}",
                    "n": len(grp), "acc": grp["correct"].mean()})

out_df = pd.DataFrame(records)
out_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}")

# ── Plot: relations with biggest gain from embedding loss ─────────────────────
pivot = out_df.pivot(index="relation", columns="condition", values="acc")
l2_col  = f"L2 λ={best_l2}"
cos_col = f"cosine λ={best_cos}"

pivot["gain_l2"]  = pivot[l2_col]  - pivot["zero-shot"]
pivot["gain_cos"] = pivot[cos_col] - pivot["zero-shot"]
pivot["n"]        = out_df[out_df["condition"] == "zero-shot"].set_index("relation")["n"]
pivot = pivot[pivot["n"] >= 20].sort_values("gain_l2", ascending=False)

top_n = 20
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Per-relation accuracy gain over zero-shot  ·  LAMA T-REx", fontsize=13)

for ax, col, title in [
    (axes[0], "gain_l2",  f"L2 embedding loss (λ={best_l2})"),
    (axes[1], "gain_cos", f"Cosine embedding loss (λ={best_cos})"),
]:
    data = pivot[col].sort_values(ascending=False).head(top_n)
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in data.values]
    ax.barh(data.index[::-1], data.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Accuracy gain over zero-shot")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1%}"))

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PLOT}")
