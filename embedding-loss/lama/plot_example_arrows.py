#!/usr/bin/env python3
"""
Arrow plot for a single LAMA example.

For the best example in grad_diagnostic CSV (highest rho_emb), shows the
top-N tokens by probability with arrows indicating Δp after a virtual
embedding-loss gradient step — positioned on the x-axis by cos_sim to the answer.

Tokens similar to the answer sit on the right and have upward arrows;
dissimilar tokens sit on the left with downward arrows.
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertTokenizerFast

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
MODEL_NAME = "bert-base-uncased"
N_SHOW     = 30   # tokens to show in the arrow plot


def load_examples(tokenizer):
    examples = []
    for path in sorted(glob.glob(f"{DATA_DIR}/*.jsonl")):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if not d.get("evidences"): continue
                s = d["evidences"][0]["masked_sentence"]
                o = d["obj_label"].strip()
                if "[MASK]" not in s: continue
                tks = tokenizer.tokenize(o)
                if len(tks) != 1: continue
                examples.append({
                    "sentence":  s,
                    "answer":    o,
                    "answer_id": tokenizer.convert_tokens_to_ids(tks[0]),
                })
    return examples


def run_example(model, tokenizer, device, ex):
    """Full per-vocab diagnostic for one example (no top-K restriction)."""
    E     = model.bert.embeddings.word_embeddings.weight.detach()
    E_norm = F.normalize(E, dim=-1).detach()

    inp = tokenizer(ex["sentence"], return_tensors="pt",
                    max_length=128, padding="max_length", truncation=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    mp  = (inp["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mp) == 0:
        return None

    with torch.no_grad():
        raw = model(**inp).logits[0, mp[0].item()].detach()

    aid  = ex["answer_id"]

    l1   = raw.requires_grad_(True)
    el   = 1 - F.cosine_similarity(
        (torch.softmax(l1, dim=-1) @ E).unsqueeze(0), E[aid].unsqueeze(0)
    )
    g_emb = torch.autograd.grad(el, l1)[0].detach()

    l2   = raw.requires_grad_(True)
    ce   = F.cross_entropy(l2.unsqueeze(0), torch.tensor([aid], device=device))
    g_ce  = torch.autograd.grad(ce, l2)[0].detach()

    p0      = torch.softmax(raw, dim=-1).detach().cpu().numpy()
    dp_emb  = torch.softmax(raw - g_emb, dim=-1).detach().cpu().numpy() - p0
    dp_ce   = torch.softmax(raw - g_ce,  dim=-1).detach().cpu().numpy() - p0
    cos_sim = (E_norm @ E_norm[aid].detach()).detach().cpu().numpy()

    return p0, dp_emb, dp_ce, cos_sim


def make_arrow_plot(model, tokenizer, device, ex, out_path):
    result = run_example(model, tokenizer, device, ex)
    if result is None:
        print("Could not run example.")
        return

    p0, dp_emb, dp_ce, cos_sim = result
    aid = ex["answer_id"]
    vocab = tokenizer.convert_ids_to_tokens(list(range(len(p0))))

    # Select top-N tokens by p_before (the ones that actually matter)
    topk_idx = np.argsort(p0)[-N_SHOW:][::-1]

    # Sort those by cos_sim for the x-axis (left = dissimilar, right = similar)
    topk_idx = topk_idx[np.argsort(cos_sim[topk_idx])]

    x      = np.arange(len(topk_idx))
    labels = [vocab[i] for i in topk_idx]
    dp_e   = dp_emb[topk_idx]
    dp_c   = dp_ce[topk_idx]
    cs     = cos_sim[topk_idx]
    p_vals = p0[topk_idx]

    # Truncate sentence for title if very long
    sent_short = ex["sentence"].replace("[MASK]", "___")
    if len(sent_short) > 120:
        sent_short = sent_short[:117] + "..."

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             constrained_layout=True)
    fig.suptitle(
        f'"{sent_short}"  →  answer: "{ex["answer"]}"\n'
        f"Top-{N_SHOW} tokens by p_before, sorted left→right by cos_sim to answer",
        fontsize=9
    )

    for ax, dp, title in [
        (axes[0], dp_e, r"$\Delta p \;/\; \Delta \mathcal{L}_{\mathrm{emb}}$"),
        (axes[1], dp_c, r"$\Delta p \;/\; \Delta \mathcal{L}_{\mathrm{CE}}$"),
    ]:
        colors = ["#e06c75" if v < 0 else "#98c379" for v in dp]
        ax.bar(x, dp, color=colors, width=0.7, alpha=0.85)

        # Mark the correct answer token
        for j, idx in enumerate(topk_idx):
            if idx == aid:
                ax.bar(j, dp[j], color="#e5c07b", width=0.7, alpha=1.0,
                       edgecolor="black", linewidth=1.5)
                ax.text(j, dp[j] + np.sign(dp[j]) * 0.0003,
                        "★", ha="center", va="bottom" if dp[j] >= 0 else "top",
                        fontsize=12)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(r"$\Delta p$", fontsize=11, labelpad=4)
        ax.set_title(title, fontsize=10, pad=4)
        ax.grid(True, alpha=0.2, axis="y")

    # x-tick labels: token text + p_before (on bottom panel only)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"{lbl}\n({p:.3f})" for lbl, p in zip(labels, p_vals)],
        fontsize=7.5, rotation=45, ha="right"
    )
    axes[0].tick_params(labelbottom=False)

    # Cos-sim gradient colorbar as background shading on both axes
    for ax in axes:
        for j, c in enumerate(cs):
            ax.axvspan(j - 0.5, j + 0.5,
                       color=plt.cm.RdYlGn((c + 1) / 2),
                       alpha=0.08, zorder=0)

    # Legend
    green_patch = mpatches.Patch(color="#98c379", label="Probability ↑")
    red_patch   = mpatches.Patch(color="#e06c75", label="Probability ↓")
    gold_patch  = mpatches.Patch(color="#e5c07b", label=f"Correct answer ({ex['answer']})")
    axes[0].legend(handles=[green_patch, red_patch, gold_patch],
                   fontsize=8, loc="upper left")

    # Subtitle: left = low cos_sim, right = high cos_sim
    axes[1].set_xlabel(
        "← low cos_sim to answer                                        "
        "high cos_sim to answer →",
        fontsize=9
    )

    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default=os.path.join(SCRIPT_DIR, "grad_diagnostic", "grad_diagnostic_pretrained.csv"),
                        help="CSV from grad_diagnostic.py — used to pick the best example")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--idx",   type=int, default=None,
                        help="Override: use example at this index instead of best rho_emb")
    parser.add_argument("--answer", default=None,
                        help="Override: pick first example with this answer word")
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "grad_diagnostic"))
    args = parser.parse_args()

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = BertForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()

    all_ex = load_examples(tokenizer)
    ex_by_answer = {e["answer"].lower(): e for e in all_ex}

    if args.answer:
        ex = ex_by_answer.get(args.answer.lower())
        if ex is None:
            print(f"No example found with answer '{args.answer}'")
            return
        label = args.answer
    elif args.idx is not None:
        ex    = all_ex[args.idx]
        label = ex["answer"]
    elif os.path.exists(args.csv):
        df    = pd.read_csv(args.csv)
        best  = df.loc[df["rho_emb"].idxmax()]
        ex    = ex_by_answer.get(best["answer"].lower())
        label = best["answer"]
        if ex is None:
            print(f"Best example answer '{best['answer']}' not found in loaded data; using idx 0")
            ex = all_ex[0]; label = ex["answer"]
        print(f"Best rho_emb example: answer='{label}'  rho_emb={best['rho_emb']:.3f}  rho_ce={best['rho_ce']:.3f}")
    else:
        ex    = all_ex[0]
        label = ex["answer"]

    print(f"Sentence: {ex['sentence']}")
    print(f"Answer:   {ex['answer']}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"arrow_plot_{label.replace(' ', '_')}.png")
    make_arrow_plot(model, tokenizer, device, ex, out)


if __name__ == "__main__":
    main()
