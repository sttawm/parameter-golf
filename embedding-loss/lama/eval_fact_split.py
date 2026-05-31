#!/usr/bin/env python3
"""
Train 1 seed CE-only, save best checkpoint, then evaluate on
shared-fact vs unseen-fact test subsets. Writes results to
fact_split_results.json for use by plot_fact_split_bar.py.
"""
import copy, glob, json, os, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForMaskedLM
from tqdm import tqdm

LAMA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(LAMA_DIR, "data")
DEVICE   = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN, BATCH, LR, MAX_STEPS, GRAD_CLIP = 128, 32, 2e-5, 6000, 1.0
SEED = 42
print(f"Device: {DEVICE}")

tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

# ── Load all examples with fact key ──────────────────────────────────────────
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
            for ev in d["evidences"]:
                s = ev.get("masked_sentence", "")
                if "[MASK]" in s:
                    all_examples.append({"sentence": s, "answer_id": answer_id, "fact_key": key})

random.seed(SEED); np.random.seed(SEED)
examples = all_examples.copy(); random.shuffle(examples)
n = int(len(examples) * 0.1)
test_ex  = examples[:n]
val_ex   = examples[n:2*n]
train_ex = examples[2*n:]

train_facts = set(e["fact_key"] for e in train_ex)
shared = [e for e in test_ex if e["fact_key"] in     train_facts]
unseen = [e for e in test_ex if e["fact_key"] not in train_facts]
print(f"Train: {len(train_ex):,}  Val: {len(val_ex):,}")
print(f"Test shared: {len(shared):,}   unseen: {len(unseen):,}")

class LamaDS(Dataset):
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
def eval_acc(model, data, batch=64):
    model.eval()
    dl = DataLoader(LamaDS(data), batch_size=batch, num_workers=0)
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

# ── Zero-shot baseline ────────────────────────────────────────────────────────
model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(DEVICE)
rng   = random.Random(1)
shared_sample = rng.sample(shared, min(3000, len(shared)))

print("Zero-shot eval...")
zs_shared = eval_acc(model, shared_sample)
zs_unseen = eval_acc(model, unseen)
print(f"  ZS shared={zs_shared:.4f}  unseen={zs_unseen:.4f}")

# ── Fine-tune CE-only ─────────────────────────────────────────────────────────
train_dl = DataLoader(LamaDS(train_ex), batch_size=BATCH, shuffle=True, num_workers=0)
val_dl   = DataLoader(LamaDS(val_ex),   batch_size=64,    num_workers=0)

torch.manual_seed(SEED)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
best_val, best_state, global_step, finished = 0.0, None, 0, False

while not finished:
    model.train()
    for batch in tqdm(train_dl, desc="train", leave=False):
        ids = batch["input_ids"].to(DEVICE)
        am  = batch["attention_mask"].to(DEVICE)
        mp  = batch["mask_pos"].to(DEVICE)
        ans = batch["answer_id"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=am)
        B   = ids.shape[0]
        lat = out.logits[torch.arange(B), mp]
        loss = F.cross_entropy(lat, ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step(); opt.zero_grad()
        global_step += 1
        if global_step % 500 == 0:
            val_acc = eval_acc(model, list(val_ex)[:2000])
            print(f"  step={global_step}  val={val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                best_state = copy.deepcopy(model.state_dict())
            model.train()
        if global_step >= MAX_STEPS:
            finished = True; break

model.load_state_dict(best_state)
print(f"\nFine-tune done. Best val={best_val:.4f}")
print("Evaluating on shared vs unseen test sets...")
ft_shared = eval_acc(model, shared_sample)
ft_unseen = eval_acc(model, unseen)
print(f"  FT shared={ft_shared:.4f}  unseen={ft_unseen:.4f}")

results = {
    "zs_shared": zs_shared, "zs_unseen": zs_unseen,
    "ft_shared": ft_shared, "ft_unseen": ft_unseen,
    "n_shared": len(shared), "n_unseen": len(unseen),
    "n_shared_sample": len(shared_sample),
}
out_json = os.path.join(LAMA_DIR, "fact_split_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved → {out_json}")
