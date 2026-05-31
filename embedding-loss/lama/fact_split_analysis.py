#!/usr/bin/env python3
"""
Analyzes fact leakage in the random train/test split and evaluates
zero-shot BERT accuracy on shared-fact vs. unseen-fact test examples.
"""
import json, glob, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import BertTokenizerFast, BertForMaskedLM
from torch.utils.data import Dataset, DataLoader

LAMA_DIR  = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama"
DATA_DIR  = f"{LAMA_DIR}/data"
MAX_LEN   = 128
BATCH     = 64
DEVICE    = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

# ── Load all examples with fact key ──────────────────────────────────────────
print("Loading data...")
all_examples = []
for path in glob.glob(f"{DATA_DIR}/*.jsonl"):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("evidences"): continue
            obj_label = d["obj_label"].strip()
            toks = tok.tokenize(obj_label)
            if len(toks) != 1: continue
            answer_id = tok.convert_tokens_to_ids(toks[0])
            key = (d["sub_uri"], d["predicate_id"], d["obj_uri"])
            sub, pred, obj = d["sub_label"], d["predicate_id"], d["obj_label"]
            for ev in d["evidences"]:
                s = ev.get("masked_sentence", "")
                if "[MASK]" in s:
                    all_examples.append({"sentence": s, "answer_id": answer_id,
                                         "answer": obj, "fact_key": key,
                                         "sub": sub, "pred": pred, "obj": obj})

print(f"  {len(all_examples):,} total examples, {len(set(e['fact_key'] for e in all_examples)):,} unique facts")

# ── Reproduce the random split (seed=42) ─────────────────────────────────────
random.seed(42); np.random.seed(42)
examples = all_examples.copy(); random.shuffle(examples)
n = int(len(examples) * 0.1)
test_ex  = examples[:n]
train_ex = examples[2*n:]
train_facts = set(e["fact_key"] for e in train_ex)

shared = [e for e in test_ex if e["fact_key"] in     train_facts]
unseen = [e for e in test_ex if e["fact_key"] not in train_facts]
print(f"\nTest split:  {len(test_ex):,} examples")
print(f"  shared fact (triple in train): {len(shared):,}  ({len(shared)/len(test_ex):.1%})")
print(f"  unseen fact:                   {len(unseen):,}  ({len(unseen)/len(test_ex):.1%})")

# ── Print example sentences for the same fact ─────────────────────────────────
print("\n── Example: same fact, different sentences ──────────────────────────────")
SHOW_PREDS = {"P36", "P17", "P131", "P19", "P495"}
shown = 0
fact_to_sents = defaultdict(list)
for e in all_examples:
    if e["pred"] in SHOW_PREDS:
        fact_to_sents[e["fact_key"]].append(e)

for key, exs in fact_to_sents.items():
    uniq = list({e["sentence"]: e for e in exs}.values())
    if len(uniq) >= 3 and shown < 4:
        e0 = uniq[0]
        print(f"\n  [{e0['pred']}]  {e0['sub']}  →  {e0['obj']}")
        for ex in uniq[:3]:
            print(f"    {ex['sentence'][:110]}")
        shown += 1

# ── Zero-shot BERT eval ───────────────────────────────────────────────────────
class LamaDL(Dataset):
    def __init__(self, ex):
        self.ex = ex
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        e   = self.ex[i]
        enc = tok(e["sentence"], max_length=MAX_LEN, padding="max_length",
                  truncation=True, return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        am  = enc["attention_mask"].squeeze(0)
        mp  = (ids == tok.mask_token_id).nonzero(as_tuple=True)[0]
        mp  = mp[0] if len(mp) > 0 else torch.tensor(0)
        return {"input_ids": ids, "attention_mask": am, "mask_pos": mp,
                "answer_id": torch.tensor(e["answer_id"], dtype=torch.long)}

@torch.no_grad()
def eval_acc(model, data):
    model.eval()
    dl = DataLoader(LamaDL(data), batch_size=BATCH, num_workers=0)
    correct, total = 0, 0
    for batch in dl:
        ids = batch["input_ids"].to(DEVICE)
        am  = batch["attention_mask"].to(DEVICE)
        mp  = batch["mask_pos"].to(DEVICE)
        ans = batch["answer_id"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=am)
        B   = ids.shape[0]
        lat = out.logits[torch.arange(B), mp]
        correct += (lat.argmax(-1) == ans).sum().item()
        total   += B
    return correct / total

print("\nLoading BERT (zero-shot)...")
model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(DEVICE)

# Sample to keep it fast
rng = random.Random(0)
shared_sample = rng.sample(shared, min(5000, len(shared)))
unseen_sample = unseen  # only 987, use all

print(f"Evaluating on {len(shared_sample):,} shared + {len(unseen_sample):,} unseen...")
acc_shared = eval_acc(model, shared_sample)
acc_unseen = eval_acc(model, unseen_sample)
print(f"  Zero-shot  shared: {acc_shared:.1%}")
print(f"  Zero-shot  unseen: {acc_unseen:.1%}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
fig.patch.set_facecolor("white")

# Left: fact overlap pie
ax = axes[0]
ax.pie([len(shared), len(unseen)],
       labels=[f"Shared fact\n{len(shared)/len(test_ex):.1%}", f"Unseen fact\n{len(unseen)/len(test_ex):.1%}"],
       colors=["#f8d7da", "#e0e0e0"], startangle=90,
       wedgeprops=dict(edgecolor="white", linewidth=2),
       textprops=dict(fontsize=10))
ax.set_title("Test-set fact overlap\n(random split, seed=42)", fontsize=10, fontweight="bold")

# Right: zero-shot accuracy by split type
ax = axes[1]
bars = ax.bar([0, 1], [acc_shared, acc_unseen],
              color=["#f8d7da", "#e0e0e0"], edgecolor="#888",
              linewidth=0.6, width=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Shared fact\n(triple in train)", "Unseen fact\n(triple not in train)"], fontsize=10)
ax.set_ylabel("Zero-shot accuracy", fontsize=10)
ax.set_title("Zero-shot BERT accuracy\nby fact overlap type", fontsize=10, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
for i, v in enumerate([acc_shared, acc_unseen]):
    ax.text(i, v + 0.003, f"{v:.1%}", ha="center", fontsize=10)
ax.set_ylim(0, max(acc_shared, acc_unseen) * 1.2)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
out = f"{LAMA_DIR}/fact_split_analysis.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
