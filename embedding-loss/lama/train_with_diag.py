#!/usr/bin/env python3
"""
Train BERT on LAMA T-REx and, at every eval step, compute the gradient diagnostic:
  rho_emb = Spearman( cos_sim(E[v], E[answer]),  Δp from embedding-loss gradient )
  rho_ce  = Spearman( cos_sim(E[v], E[answer]),  Δp from CE gradient )

Supports multiple lambdas in one run (--lams 0.0 4.0) with a comparison overlay plot.

Key question: does CE already create the embedding-similarity correlation (rho_CE high)?
  - If rho_CE ≈ rho_Emb for λ=0 run → CE mimics embedding loss → redundancy explains null result
  - If λ=4.0 raises rho_Emb but not val_acc → embedding loss adds gradient signal that doesn't help recall

Outputs (per lambda):
  diag_results_<tag>.csv    — per-step rho_emb, rho_ce, val_acc
  diag_plot_<tag>.png       — single-run training curve with rho overlay
  diag_compare_<seed>.png   — overlay of all lambdas (when multiple --lams given)
"""

import argparse
import copy
import glob
import json
import os
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizerFast

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(SCRIPT_DIR, "data")
MODEL_NAME       = "bert-base-uncased"
BATCH_SIZE       = 32
MAX_STEPS        = 6000
EVAL_EVERY_STEPS = 200
LR               = 2e-5
MAX_LEN          = 128
DIAG_N           = 200   # examples for inline diagnostic (keep fast)
DIAG_TOP_K       = 100   # restrict diagnostic to top-K tokens by p_before per example


# ── Data ──────────────────────────────────────────────────────────────────────
def load_all_examples(tokenizer):
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
    return examples


class LamaDataset(Dataset):
    def __init__(self, ex, tok):
        self.ex = ex; self.tok = tok
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        e   = self.ex[i]
        enc = self.tok(e["sentence"], max_length=MAX_LEN, padding="max_length",
                       truncation=True, return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        am  = enc["attention_mask"].squeeze(0)
        mp  = (ids == self.tok.mask_token_id).nonzero(as_tuple=True)[0]
        mp  = mp[0] if len(mp) > 0 else torch.tensor(0)
        return {"input_ids": ids, "attention_mask": am, "mask_pos": mp,
                "answer_id": torch.tensor(e["answer_id"], dtype=torch.long)}


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    correct, total = 0, 0
    for batch in dl:
        ids = batch["input_ids"].to(device)
        am  = batch["attention_mask"].to(device)
        mp  = batch["mask_pos"].to(device)
        ans = batch["answer_id"].to(device)
        out = model(input_ids=ids, attention_mask=am)
        B   = out.logits.shape[0]
        lat = out.logits[torch.arange(B), mp]
        correct += (lat.argmax(-1) == ans).sum().item()
        total   += B
    return correct / total


# ── Inline gradient diagnostic ────────────────────────────────────────────────
def compute_diag_rho(model, diag_examples, tokenizer, device, top_k=DIAG_TOP_K):
    """
    Compute Spearman rho for embedding loss and CE loss on diag_examples.

    For each example, restrict to the top_k tokens by p_before — these are the
    tokens where gradient signal actually matters.  All vocab tokens would be
    dominated by the near-zero-prob mass and obscure the relevant signal.

    Returns (rho_emb, rho_ce, n_points) aggregated across all examples.
    """
    E      = model.bert.embeddings.word_embeddings.weight.detach()
    E_norm = F.normalize(E, dim=-1)

    all_cos    = []
    all_dp_emb = []
    all_dp_ce  = []

    model.eval()
    for ex in diag_examples:
        inputs = tokenizer(
            ex["sentence"], return_tensors="pt",
            max_length=MAX_LEN, padding="max_length", truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        mask_pos = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_pos) == 0:
            continue
        mp        = mask_pos[0].item()
        answer_id = ex["answer_id"]

        with torch.no_grad():
            raw = model(**inputs).logits[0, mp].detach()

        # ── embedding loss gradient w.r.t. logits ────────────────────────────
        logits1 = raw.requires_grad_(True)
        probs1  = torch.softmax(logits1, dim=-1)
        wemb    = probs1 @ E
        emb_loss = 1.0 - F.cosine_similarity(wemb.unsqueeze(0), E[answer_id].unsqueeze(0))
        g_emb   = torch.autograd.grad(emb_loss, logits1)[0].detach()

        # ── CE gradient w.r.t. logits ─────────────────────────────────────────
        logits2 = raw.requires_grad_(True)
        ce_loss = F.cross_entropy(logits2.unsqueeze(0),
                                  torch.tensor([answer_id], device=device))
        g_ce    = torch.autograd.grad(ce_loss, logits2)[0].detach()

        # virtual step Δp (lr=1 — doesn't affect rank correlation)
        p0     = torch.softmax(raw, dim=-1).cpu().numpy()
        dp_emb = torch.softmax(raw - g_emb, dim=-1).cpu().numpy() - p0
        dp_ce  = torch.softmax(raw - g_ce,  dim=-1).cpu().numpy() - p0

        E_answer_norm = E_norm[answer_id].detach()
        cos_sim = (E_norm @ E_answer_norm).detach().cpu().numpy()

        # Restrict to top-K tokens by p_before
        topk_idx = np.argpartition(p0, -top_k)[-top_k:]
        all_cos.append(cos_sim[topk_idx])
        all_dp_emb.append(dp_emb[topk_idx])
        all_dp_ce.append(dp_ce[topk_idx])

    if not all_cos:
        return float("nan"), float("nan"), 0

    all_cos    = np.concatenate(all_cos)
    all_dp_emb = np.concatenate(all_dp_emb)
    all_dp_ce  = np.concatenate(all_dp_ce)

    rho_emb, _ = spearmanr(all_cos, all_dp_emb)
    rho_ce,  _ = spearmanr(all_cos, all_dp_ce)
    return float(rho_emb), float(rho_ce), len(all_cos)


# ── Plotting ──────────────────────────────────────────────────────────────────
def make_plot(log: list[dict], tag: str, lam: float, out_dir: str):
    df = pd.DataFrame(log)
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 6))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(df["step"], df["val_acc"], color="#2ecc71", linewidth=2, label="Val accuracy")
    ax1.set_ylabel("Val accuracy", fontsize=10)
    ax1.set_title(
        f"Training dynamics — {tag}  (λ={lam})\n"
        "If rho_CE ≈ rho_Emb, CE already captures embedding-loss signal",
        fontsize=10
    )
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)

    ax2.plot(df["step"], df["rho_ce"],  color="#61afef", linewidth=2, label="rho_CE")
    ax2.plot(df["step"], df["rho_emb"], color="#e06c75", linewidth=2, linestyle="--",
             label="rho_Emb")
    ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax2.set_ylabel("Spearman ρ (cos_sim vs Δp)", fontsize=10)
    ax2.set_xlabel("Training step", fontsize=10)
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(out_dir, f"diag_plot_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Comparison overlay plot ───────────────────────────────────────────────────
def make_compare_plot(all_logs: dict, seed: int, out_dir: str):
    """
    Overlay rho_CE and rho_Emb for multiple lambdas on shared axes.
    all_logs: {lam: list-of-dicts}
    """
    colors   = ["#61afef", "#e06c75", "#98c379", "#e5c07b", "#c678dd"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"Gradient diagnostic comparison (seed={seed})\n"
        "Top: val accuracy  |  Bottom: Spearman ρ(cos_sim, Δp)",
        fontsize=10
    )
    axes[0].set_ylabel("Val accuracy", fontsize=10)
    axes[1].set_ylabel("Spearman ρ", fontsize=10)
    axes[1].set_xlabel("Training step", fontsize=10)
    axes[1].axhline(0, color="black", linewidth=0.7, linestyle=":")

    for i, (lam, log) in enumerate(sorted(all_logs.items())):
        df  = pd.DataFrame(log)
        c   = colors[i % len(colors)]
        lbl = f"λ={lam}"
        axes[0].plot(df["step"], df["val_acc"],  color=c, linewidth=1.8, label=lbl)
        axes[1].plot(df["step"], df["rho_ce"],   color=c, linewidth=1.8, linestyle="-",
                     label=f"{lbl} rho_CE")
        axes[1].plot(df["step"], df["rho_emb"],  color=c, linewidth=1.8, linestyle="--",
                     label=f"{lbl} rho_Emb")

    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"diag_compare_seed{seed}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Single-lambda training run ────────────────────────────────────────────────
def run_one(lam, seed, train_dl, val_dl, diag_ex, tokenizer, device, max_steps, out_dir):
    tag = f"lam{lam}_seed{seed}"
    csv_path = os.path.join(out_dir, f"diag_results_{tag}.csv")

    # Resume from existing CSV if present
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        if len(existing) > 0:
            print(f"  [{tag}] Resuming from existing {len(existing)} rows in {csv_path}")
            return existing.to_dict("records")

    model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    torch.manual_seed(seed + int(lam * 100))
    opt  = torch.optim.AdamW(model.parameters(), lr=LR)
    E    = model.bert.embeddings.word_embeddings.weight

    log         = []
    global_step = 0
    finished    = False

    print(f"\n{'─'*55}\n  λ={lam}  seed={seed}\n{'─'*55}")
    while not finished:
        model.train()
        for batch in tqdm(train_dl, desc=f"λ={lam} step={global_step}", leave=False):
            ids = batch["input_ids"].to(device)
            am  = batch["attention_mask"].to(device)
            mp  = batch["mask_pos"].to(device)
            ans = batch["answer_id"].to(device)
            out = model(input_ids=ids, attention_mask=am)
            B   = out.logits.shape[0]
            lat = out.logits[torch.arange(B), mp]
            ce  = F.cross_entropy(lat, ans)
            if lam > 0:
                el   = (1 - F.cosine_similarity(
                    torch.softmax(lat.float(), dim=-1) @ E,
                    E[ans], dim=-1
                )).mean()
                loss = ce + lam * el
            else:
                loss = ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            global_step += 1

            if global_step % EVAL_EVERY_STEPS == 0:
                val_acc         = evaluate(model, val_dl, device)
                rho_emb, rho_ce, n_pts = compute_diag_rho(model, diag_ex, tokenizer, device)
                print(f"  step={global_step:5d}  val={val_acc:.4f}  "
                      f"rho_CE={rho_ce:+.3f}  rho_Emb={rho_emb:+.3f}  "
                      f"(top{DIAG_TOP_K}×{len(diag_ex)}={n_pts} pts)")
                log.append({"step": global_step, "val_acc": val_acc,
                             "rho_emb": rho_emb, "rho_ce": rho_ce})
                model.train()

            if global_step >= max_steps:
                finished = True; break

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(log).to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")
    make_plot(log, tag, lam, out_dir)
    return log


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lams", type=float, nargs="+", default=[0.0],
                        help="Lambda value(s) for cosine embedding loss (e.g. --lams 0.0 4.0)")
    parser.add_argument("--seed", type=int,   default=42)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--out-dir",   default=os.path.join(SCRIPT_DIR, "grad_diagnostic"))
    args = parser.parse_args()

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}  |  λ(s)={args.lams}  seed={args.seed}")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    print("Loading data...")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    all_ex = load_all_examples(tokenizer)
    random.shuffle(all_ex)
    n_test = int(len(all_ex) * 0.1)
    n_val  = int(len(all_ex) * 0.1)
    val_ex   = all_ex[n_test:n_test + n_val]
    train_ex = all_ex[n_test + n_val:]
    print(f"  train={len(train_ex)}  val={len(val_ex)}")

    # Fixed diagnostic subset (same examples for all lambda conditions)
    rng      = np.random.default_rng(0)
    diag_idx = rng.choice(len(val_ex), size=min(DIAG_N, len(val_ex)), replace=False)
    diag_ex  = [val_ex[i] for i in diag_idx]

    val_dl   = DataLoader(LamaDataset(val_ex,   tokenizer), batch_size=BATCH_SIZE)
    train_dl = DataLoader(LamaDataset(train_ex, tokenizer), batch_size=BATCH_SIZE, shuffle=True)

    all_logs = {}
    for lam in args.lams:
        all_logs[lam] = run_one(lam, args.seed, train_dl, val_dl, diag_ex,
                                tokenizer, device, args.max_steps, args.out_dir)

    if len(args.lams) > 1:
        make_compare_plot(all_logs, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
