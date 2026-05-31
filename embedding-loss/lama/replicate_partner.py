#!/usr/bin/env python3
"""
Replicates partner's setup: all evidence sentences, random 90/10 split.
Hypothesis: same facts in train and eval -> ~99% accuracy.
Control:    one sentence per fact, sequential split -> should give ~67%.
"""
import json, glob, random, math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForMaskedLM

DATA_DIR   = "/root/data"
MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 128
BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42

random.seed(SEED)
torch.manual_seed(SEED)

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

# ── Load all facts ──────────────────────────────────────────────────────────
PARTNER_LIMIT = 34000

all_facts = []   # {fact_key, sentence, answer_id}
done = False
for path in sorted(glob.glob(f"{DATA_DIR}/*.jsonl")):
    if done: break
    with open(path) as f:
        for line in f:
            if len(all_facts) >= PARTNER_LIMIT:
                done = True; break
            d = json.loads(line)
            if not d.get("evidences"): continue
            obj = d["obj_label"].strip()
            toks = tokenizer.tokenize(obj)
            if len(toks) != 1: continue
            answer_id = tokenizer.convert_tokens_to_ids(toks[0])
            fact_key  = (d.get("sub_uri",""), d.get("obj_uri",""), d.get("predicate_id",""))
            for ev in d["evidences"]:
                sent = ev["masked_sentence"]
                if "[MASK]" not in sent: continue
                all_facts.append({"fact": fact_key, "sentence": sent, "answer_id": answer_id})
                if len(all_facts) >= PARTNER_LIMIT:
                    break

print(f"Total sentences: {len(all_facts):,}")
print(f"Unique facts:    {len(set(e['fact'] for e in all_facts)):,}")

# ── Dataset ─────────────────────────────────────────────────────────────────
class LamaDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        ex  = self.examples[i]
        enc = tokenizer(ex["sentence"], max_length=MAX_LEN,
                        padding="max_length", truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = torch.full_like(input_ids, -100)
        mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions):
            labels[mask_positions[0]] = ex["answer_id"]
        return input_ids, attention_mask, labels

def evaluate(model, dl):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ids, amask, labels in dl:
            ids, amask, labels = ids.to(DEVICE), amask.to(DEVICE), labels.to(DEVICE)
            logits = model(input_ids=ids, attention_mask=amask).logits
            preds  = logits.argmax(-1)
            mask   = labels != -100
            correct += (preds[mask] == labels[mask]).sum().item()
            total   += mask.sum().item()
    return correct / total if total else 0.0

def run_experiment(name, train_ex, eval_ex, n_epochs=10):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  train={len(train_ex):,}  eval={len(eval_ex):,}")
    train_facts = set(e["fact"] for e in train_ex)
    eval_facts  = set(e["fact"] for e in eval_ex)
    overlap = train_facts & eval_facts
    print(f"  unique train facts={len(train_facts):,}  eval facts={len(eval_facts):,}")
    print(f"  eval facts seen in train: {len(overlap)}/{len(eval_facts)} ({len(overlap)/len(eval_facts):.1%})")

    train_dl = DataLoader(LamaDataset(train_ex), batch_size=BATCH_SIZE, shuffle=True)
    eval_dl  = DataLoader(LamaDataset(eval_ex),  batch_size=BATCH_SIZE)

    model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for ids, amask, labels in train_dl:
            ids, amask, labels = ids.to(DEVICE), amask.to(DEVICE), labels.to(DEVICE)
            loss = model(input_ids=ids, attention_mask=amask, labels=labels).loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        acc = evaluate(model, eval_dl)
        print(f"  epoch {epoch}  loss={total_loss/len(train_dl):.4f}  eval_acc={acc:.2%}")

# ── Experiment A: partner setup (all sentences, random split) ───────────────
subset = all_facts[:34000]
random.shuffle(subset)
n_eval = int(len(subset) * 0.1)
run_experiment(
    "PARTNER SETUP: all sentences, random split",
    train_ex=subset[n_eval:],
    eval_ex=subset[:n_eval],
)

# ── Experiment B: our setup (one sentence per fact, sequential split) ────────
seen = set()
one_per_fact = []
for e in all_facts:
    if e["fact"] not in seen:
        seen.add(e["fact"])
        one_per_fact.append(e)

n_test = int(len(one_per_fact) * 0.1)
run_experiment(
    "OUR SETUP: one sentence/fact, sequential split",
    train_ex=one_per_fact[n_test:],
    eval_ex=one_per_fact[:n_test],
)
