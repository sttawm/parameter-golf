#!/usr/bin/env python3
"""
Gradient diagnostic for embedding loss — explains why adding embedding loss
doesn't improve LAMA factual recall over CE alone.

Hypothesis: CE loss already creates a positive correlation between
cos_sim(E[v], E[answer]) and Δp[v] — i.e., the model's output logits are
already embedding-aware, so the cosine embedding loss adds no new gradient
signal that CE isn't already providing.

For each LAMA example:
  1. Forward pass → logits at [MASK] → p_before
  2. Compute grad of cosine embedding loss w.r.t. logits
  3. Compute grad of CE loss w.r.t. logits
  4. Virtual single step: p_after = softmax(logits - lr * grad)
  5. delta_p = p_after - p_before
  6. cos_sim[v] = cosine(E[v], E[answer]) for every vocab token v
  7. Scatter delta_p vs cos_sim; report Spearman rho for both losses

Key comparison:
  - rho_ce high  → CE already pushes prob toward embedding-similar tokens
                   → embedding loss is redundant → explains null result
  - rho_ce low, rho_emb high → embedding loss adds distinct signal
                               → would need another explanation for null result

Usage:
  python grad_diagnostic.py --tag pretrained
  python grad_diagnostic.py --model /path/to/checkpoint --tag finetuned-ce
  python grad_diagnostic.py --model /path/to/checkpoint --tag finetuned-emb-1.0
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import BertForMaskedLM, BertTokenizerFast

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
MODEL_NAME = "bert-base-uncased"
TOP_K      = 100   # per-example: restrict to top-K tokens by p_before

# ── Data loading (same as lama_sweep_topup.py) ─────────────────────────────────
def load_examples(n: int, tokenizer):
    examples = []
    for path in sorted(glob.glob(f"{DATA_DIR}/*.jsonl")):
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
        if len(examples) >= n:
            break
    return examples[:n]


# ── Single-example gradient analysis ──────────────────────────────────────────
def analyze_example(model, tokenizer, device, example, E_norm, E, lr: float):
    """
    Returns dict with per-vocab arrays (cos_sim, delta_p_emb, delta_p_ce)
    and scalar summary stats.  Returns None if tokenization fails.
    """
    inputs = tokenizer(
        example["sentence"],
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return None
    mask_pos = mask_positions[0].item()
    answer_id = example["answer_id"]

    # Forward pass — detach logits so we can take grad w.r.t. them as inputs
    with torch.no_grad():
        outputs = model(**inputs)
        raw_logits = outputs.logits[0, mask_pos].detach()

    logits = raw_logits.requires_grad_(True)
    probs  = torch.softmax(logits, dim=-1)
    p_before = probs.detach().cpu().numpy()

    # ── Embedding loss gradient ───────────────────────────────────────────────
    # Loss = 1 - cosine(sum_v p_v * E_v, E_answer)
    probs_for_emb = torch.softmax(logits, dim=-1)
    weighted_emb  = probs_for_emb @ E                              # (768,)
    cos_emb_loss  = 1.0 - F.cosine_similarity(
        weighted_emb.unsqueeze(0),
        E[answer_id].unsqueeze(0)
    )
    grad_emb = torch.autograd.grad(cos_emb_loss, logits, retain_graph=False)[0].detach()

    # ── CE loss gradient ──────────────────────────────────────────────────────
    logits2  = raw_logits.requires_grad_(True)
    ce_loss  = F.cross_entropy(logits2.unsqueeze(0),
                               torch.tensor([answer_id], device=device))
    grad_ce  = torch.autograd.grad(ce_loss, logits2)[0].detach()

    # ── Virtual single gradient step ─────────────────────────────────────────
    p_after_emb = torch.softmax((raw_logits - lr * grad_emb), dim=-1).detach().cpu().numpy()
    p_after_ce  = torch.softmax((raw_logits - lr * grad_ce),  dim=-1).detach().cpu().numpy()

    delta_p_emb = p_after_emb - p_before
    delta_p_ce  = p_after_ce  - p_before

    # ── Cosine similarity of every vocab token to the answer token ────────────
    E_answer_norm = E_norm[answer_id].detach()
    cos_sim = (E_norm @ E_answer_norm).detach().cpu().numpy()      # (V,)

    # ── Restrict to top-K tokens by p_before ─────────────────────────────────
    # All-vocab correlation is dominated by the near-zero-prob mass; top-K
    # focuses on tokens where gradient signal actually matters.
    topk_idx    = np.argpartition(p_before, -TOP_K)[-TOP_K:]
    cos_topk    = cos_sim[topk_idx]
    dp_emb_topk = delta_p_emb[topk_idx]
    dp_ce_topk  = delta_p_ce[topk_idx]

    # ── Correlations ─────────────────────────────────────────────────────────
    rho_emb, _ = spearmanr(cos_topk, dp_emb_topk)
    rho_ce,  _ = spearmanr(cos_topk, dp_ce_topk)

    return {
        "answer":              example["answer"],
        "answer_id":           answer_id,
        "p_before_answer":     float(p_before[answer_id]),
        "p_after_emb_answer":  float(p_after_emb[answer_id]),
        "p_after_ce_answer":   float(p_after_ce[answer_id]),
        "delta_p_emb_answer":  float(delta_p_emb[answer_id]),
        "delta_p_ce_answer":   float(delta_p_ce[answer_id]),
        "cos_sim_answer":      float(cos_sim[answer_id]),
        "rho_emb":             rho_emb,
        "rho_ce":              rho_ce,
        # top-K arrays for aggregate scatter
        "_cos_sim":            cos_topk,
        "_delta_p_emb":        dp_emb_topk,
        "_delta_p_ce":         dp_ce_topk,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def make_plots(records: list[dict], out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in records])

    # ── 1. Scatter: cos_sim vs delta_p (aggregate, sampled) ──────────────────
    all_cos      = np.concatenate([r["_cos_sim"]      for r in records])
    all_dp_emb   = np.concatenate([r["_delta_p_emb"]  for r in records])
    all_dp_ce    = np.concatenate([r["_delta_p_ce"]   for r in records])

    # Subsample for scatter rendering only (Spearman computed on full set)
    rho_emb_agg, _ = spearmanr(all_cos, all_dp_emb)
    rho_ce_agg,  _ = spearmanr(all_cos, all_dp_ce)
    rng  = np.random.default_rng(42)
    plot_idx = rng.choice(len(all_cos), size=min(50_000, len(all_cos)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Gradient diagnostic — {tag}  (top-{TOP_K} tokens × {len(records):,} examples = {len(all_cos):,} pts)\n"
        "If rho_CE ≈ rho_Emb → CE already captures embedding-loss signal → redundancy explains null result",
        fontsize=10
    )
    for ax, dp, rho, label, color in [
        (axes[0], all_dp_emb[plot_idx], rho_emb_agg, "Cosine embedding loss", "#e06c75"),
        (axes[1], all_dp_ce[plot_idx],  rho_ce_agg,  "CE loss",               "#61afef"),
    ]:
        ax.scatter(all_cos[plot_idx], dp, alpha=0.03, s=1, color=color, rasterized=True)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("cos_sim(E[v], E[answer])", fontsize=10)
        ax.set_ylabel("Δ probability after 1 virtual step", fontsize=10)
        ax.set_title(f"{label}\nSpearman ρ = {rho:.4f}  (n = {len(all_cos):,})", fontsize=10)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    scatter_path = os.path.join(out_dir, f"grad_scatter_{tag}.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {scatter_path}")

    # ── 2. Distribution of per-example Spearman rho ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["rho_emb"], bins=40, color="#e06c75", alpha=0.7, label="Embedding loss")
    ax.hist(df["rho_ce"],  bins=40, color="#61afef", alpha=0.7, label="CE loss")
    ax.axvline(df["rho_emb"].mean(), color="#c0392b", linewidth=1.5, linestyle="--",
               label=f"Emb mean = {df['rho_emb'].mean():.3f}")
    ax.axvline(df["rho_ce"].mean(),  color="#2980b9", linewidth=1.5, linestyle="--",
               label=f"CE  mean = {df['rho_ce'].mean():.3f}")
    ax.set_xlabel("Spearman ρ (cos_sim vs Δp per example)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Per-example correlation distribution — {tag}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    hist_path = os.path.join(out_dir, f"grad_rho_hist_{tag}.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {hist_path}")

    # ── 3. Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Tag: {tag}   (n_examples = {len(df)})")
    print(f"{'='*60}")
    print(f"Mean per-example Spearman ρ (cos_sim vs Δp):")
    print(f"  CE loss:         {df['rho_ce'].mean():.4f} ± {df['rho_ce'].std():.4f}")
    print(f"  Embedding loss:  {df['rho_emb'].mean():.4f} ± {df['rho_emb'].std():.4f}")
    print(f"\nAggregate Spearman ρ (all vocab tokens pooled):")
    print(f"  CE loss:         {rho_ce_agg:.4f}")
    print(f"  Embedding loss:  {rho_emb_agg:.4f}")
    if abs(rho_ce_agg) > 0.3 and abs(rho_ce_agg / max(abs(rho_emb_agg), 1e-9)) > 0.7:
        print(f"\n  → CE already captures most of the embedding-loss signal (redundancy).")
    print(f"\nMean Δp for correct answer token:")
    print(f"  CE loss:         {df['delta_p_ce_answer'].mean():.5f}")
    print(f"  Embedding loss:  {df['delta_p_emb_answer'].mean():.5f}")
    print(f"\nMean p_before for correct answer: {df['p_before_answer'].mean():.4f}")
    print(f"{'='*60}\n")

    df.to_csv(os.path.join(out_dir, f"grad_diagnostic_{tag}.csv"), index=False)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME,
                        help="HuggingFace model name or path to fine-tuned checkpoint")
    parser.add_argument("--tag",  default="pretrained",
                        help="Label for output filenames / plot titles")
    parser.add_argument("--n",   type=int, default=34000,
                        help="Number of LAMA examples to analyse")
    parser.add_argument("--lr",  type=float, default=1.0,
                        help="Virtual learning rate for single-step delta (doesn't affect rho)")
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "grad_diagnostic"))
    args = parser.parse_args()

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device:  {device}")
    print(f"Model:   {args.model}")
    print(f"Tag:     {args.tag}")
    print(f"N:       {args.n}")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    print("Loading examples...")
    examples = load_examples(args.n, tokenizer)
    print(f"  {len(examples)} single-token examples")

    print("Loading model...")
    model = BertForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()

    # Pre-compute normalised embedding matrix once
    E = model.bert.embeddings.word_embeddings.weight.detach()   # (V, 768)
    E_norm = F.normalize(E, dim=-1)                             # (V, 768)

    print("Running gradient analysis...")
    records = []
    for i, ex in enumerate(examples):
        rec = analyze_example(model, tokenizer, device, ex, E_norm, E, lr=args.lr)
        if rec is not None:
            records.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(examples)}  "
                  f"rho_emb={np.mean([r['rho_emb'] for r in records]):.3f}  "
                  f"rho_ce={np.mean([r['rho_ce'] for r in records]):.3f}")

    make_plots(records, args.out_dir, args.tag)


if __name__ == "__main__":
    main()
