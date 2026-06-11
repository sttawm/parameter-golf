#!/usr/bin/env python3
"""
Retrain CE-only (lambda=0) and CE+Emb (lambda=1.0) for seeds 42, 123, 456,
save best checkpoint, then evaluate top-1/3/5/10/20 accuracy on the test set.
Outputs: topk_results.csv
"""

import copy, glob, json, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForMaskedLM
from tqdm import tqdm

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(SCRIPT_DIR, "data")
CKPT_DIR       = os.path.join(SCRIPT_DIR, "checkpoints")
OUT_CSV        = os.path.join(SCRIPT_DIR, "topk_results.csv")
MODEL_NAME     = "bert-base-uncased"
VAL_FRAC       = 0.1
TEST_FRAC      = 0.1
BATCH_SIZE     = 32
EVAL_EVERY     = 200
PATIENCE_EVALS = 6
LR             = 2e-5
MAX_LEN        = 128
GRAD_CLIP      = 1.0
LAMBDAS        = [0.0, 1.0]
SEEDS          = [42, 123, 456, 789, 1011, 2024, 3000, 4000, 5000, 6000,
                  7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000,
                  16000, 17000, 18000, 19000, 20000, 21000]
TOPKS          = [1, 3, 5, 10, 20]
TRAIN_EVAL_N   = 3403   # same size as test set (10% of ~34k)

os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
print(f"Device: {DEVICE}")

# ── Resume support ─────────────────────────────────────────────────────────────
if os.path.exists(OUT_CSV):
    existing   = pd.read_csv(OUT_CSV)
    results    = existing.to_dict("records")
    done_pairs = set(zip(existing["seed"].astype(int), existing["lambda"].astype(float)))
else:
    results    = []
    done_pairs = set()

# ── Load data ──────────────────────────────────────────────────────────────────
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
            all_examples.append({"sentence": sentence, "answer_id": answer_id})
print(f"  {len(all_examples):,} examples")

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

def embed_loss_cosine(lat, ans, E):
    p = torch.softmax(lat.float(), dim=-1)
    return (1 - F.cosine_similarity(p @ E, E[ans], dim=-1)).mean()

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    correct, total = 0, 0
    for batch in dl:
        ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
        mp  = batch["mask_pos"].to(DEVICE);  ans = batch["answer_id"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=am)
        B   = out.logits.shape[0]
        lat = out.logits[torch.arange(B), mp]
        correct += (lat.argmax(-1) == ans).sum().item(); total += B
    return correct / total

@torch.no_grad()
def evaluate_topk(model, dl, ks):
    model.eval()
    counts = {k: 0 for k in ks}
    total  = 0
    for batch in dl:
        ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
        mp  = batch["mask_pos"].to(DEVICE);  ans = batch["answer_id"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=am)
        B   = out.logits.shape[0]
        lat = out.logits[torch.arange(B), mp]
        max_k = max(ks)
        topk_ids = lat.topk(max_k, dim=-1).indices  # (B, max_k)
        for k in ks:
            counts[k] += (topk_ids[:, :k] == ans.unsqueeze(1)).any(dim=1).sum().item()
        total += B
    return {k: counts[k] / total for k in ks}

# ── Main loop ──────────────────────────────────────────────────────────────────
for seed in SEEDS:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    examples = all_examples.copy(); random.shuffle(examples)
    n_test   = int(len(examples) * TEST_FRAC)
    n_val    = int(len(examples) * VAL_FRAC)
    test_ex  = examples[:n_test]
    val_ex   = examples[n_test:n_test + n_val]
    train_ex = examples[n_test + n_val:]

    # Fixed training eval subset — first TRAIN_EVAL_N examples of train_ex
    # (order is determined by the seed-based shuffle, so always the same set)
    train_eval_ex = train_ex[:TRAIN_EVAL_N]

    train_dl      = DataLoader(LamaDataset(train_ex),       batch_size=BATCH_SIZE, shuffle=True)
    train_eval_dl = DataLoader(LamaDataset(train_eval_ex),  batch_size=BATCH_SIZE)
    val_dl        = DataLoader(LamaDataset(val_ex),         batch_size=BATCH_SIZE)
    test_dl       = DataLoader(LamaDataset(test_ex),        batch_size=BATCH_SIZE)

    for lam in LAMBDAS:
        if (seed, lam) in done_pairs:
            print(f"Skipping seed={seed} λ={lam} (done)")
            continue

        ckpt_path = os.path.join(CKPT_DIR, f"seed{seed}_lam{lam}.pt")

        if os.path.exists(ckpt_path):
            print(f"\nLoading checkpoint: {ckpt_path}")
            model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        else:
            print(f"\n{'='*60}\nTraining seed={seed} λ={lam}\n{'='*60}")
            torch.manual_seed(seed + hash(lam) % 10000)
            model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
            opt   = torch.optim.AdamW(model.parameters(), lr=LR)
            E     = model.bert.embeddings.word_embeddings.weight

            best_acc, best_step, best_state = 0.0, 0, None
            patience_count = 0; global_step = 0; stopped = False

            for epoch in range(1, 21):
                model.train()
                for batch in tqdm(train_dl, desc=f"  s={seed} λ={lam} ep={epoch}", leave=False):
                    ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
                    mp  = batch["mask_pos"].to(DEVICE);  ans = batch["answer_id"].to(DEVICE)
                    out = model(input_ids=ids, attention_mask=am)
                    B   = out.logits.shape[0]
                    lat = out.logits[torch.arange(B), mp]
                    ce  = F.cross_entropy(lat, ans)
                    el  = embed_loss_cosine(lat, ans, E) if lam > 0 else torch.tensor(0.0)
                    loss = ce + lam * el
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step(); opt.zero_grad()
                    global_step += 1

                    if global_step % EVAL_EVERY == 0:
                        val_acc = evaluate(model, val_dl)
                        print(f"  step={global_step} val={val_acc:.4f}")
                        if val_acc > best_acc:
                            best_acc = val_acc; best_step = global_step
                            best_state = copy.deepcopy(model.state_dict())
                            patience_count = 0
                        else:
                            patience_count += 1
                            if patience_count >= PATIENCE_EVALS:
                                print(f"  early stop at step={global_step}")
                                stopped = True; break
                        model.train()
                if stopped: break

            model.load_state_dict(best_state)
            torch.save(best_state, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

        print(f"  Evaluating top-k + train/val/test accuracy (seed={seed} λ={lam})...")
        train_acc = evaluate(model, train_eval_dl)
        val_acc   = evaluate(model, val_dl)
        topk      = evaluate_topk(model, test_dl, TOPKS)
        print(f"    train_acc (n={TRAIN_EVAL_N}): {train_acc:.4f}  val_acc: {val_acc:.4f}  gap: {train_acc - val_acc:+.4f}")
        row = {"seed": seed, "lambda": lam, "train_acc": train_acc, "val_acc": val_acc,
               "gap": train_acc - val_acc}
        for k, acc in topk.items():
            row[f"top{k}"] = acc
            print(f"    top-{k:2d}: {acc:.4f} ({acc:.1%})")
        results.append(row)
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)
        done_pairs.add((seed, lam))

print(f"\nAll done → {OUT_CSV}")

# ── Summary table ──────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
for lam in LAMBDAS:
    sub = df[df["lambda"] == lam]
    label = "CE only (λ=0)" if lam == 0 else f"CE + Emb (λ={lam})"
    print(f"\n{label}  (n={len(sub)}):")
    print(f"  train_acc: {sub['train_acc'].mean():.1%} ± {sub['train_acc'].std(ddof=1):.1%}")
    print(f"  val_acc:   {sub['val_acc'].mean():.1%} ± {sub['val_acc'].std(ddof=1):.1%}")
    print(f"  gap:       {sub['gap'].mean():+.1%} ± {sub['gap'].std(ddof=1):.1%}")
    for k in TOPKS:
        col = f"top{k}"
        print(f"  top-{k:2d}:    {sub[col].mean():.1%} ± {sub[col].std(ddof=1):.1%}")
