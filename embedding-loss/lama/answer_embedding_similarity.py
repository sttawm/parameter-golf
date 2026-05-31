#!/usr/bin/env python3
"""
For every relation in the per-relation analysis, compute the mean pairwise
cosine similarity of correct answer embeddings (using BERT's word embedding
matrix), then scatter-plot gain vs similarity to test the hypothesis.
"""
import glob, json, os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast, BertModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")

RELATION_LABELS = {
    "P17":   "country", "P19": "place of birth", "P20": "place of death",
    "P27":   "country of citizenship", "P30": "continent", "P31": "instance of",
    "P36":   "capital", "P37": "official language", "P39": "position held",
    "P47":   "shares border with", "P101": "field of work", "P103": "native language",
    "P106":  "occupation", "P108": "employer", "P127": "owned by",
    "P131":  "located in", "P136": "genre", "P138": "named after",
    "P140":  "religion", "P159": "headquarters location", "P176": "manufacturer",
    "P178":  "developer", "P190": "sister city", "P264": "record label",
    "P276":  "location", "P279": "subclass of", "P361": "part of",
    "P364":  "original language", "P407": "language of work", "P413": "position played",
    "P449":  "original network", "P463": "member of", "P495": "country of origin",
    "P527":  "has part", "P530": "diplomatic relation", "P740": "location of formation",
    "P937":  "work location", "P1001": "applies to jurisdiction",
    "P1303": "instrument", "P1376": "capital of", "P1412": "languages spoken",
}

# Load per-relation gains
df = pd.read_csv(os.path.join(SCRIPT_DIR, "lama_per_relation.csv"))
conditions = df["condition"].unique()
l2_col  = [c for c in conditions if "L2" in c][0]
cos_col = [c for c in conditions if "cosine" in c][0]
pivot = df.pivot(index="relation", columns="condition", values="acc")
pivot["gain"] = pivot[cos_col] - pivot[l2_col]
pivot["n"]    = df[df["condition"] == "zero-shot"].set_index("relation")["n"]
pivot = pivot[pivot["n"] >= 20]
all_rels = list(pivot.index)

print("Loading tokenizer and BERT embeddings...")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased")
E = model.embeddings.word_embeddings.weight.detach()

print("Loading answers per relation...")
rel_answers = {r: set() for r in all_rels}
for path in glob.glob(f"{DATA_DIR}/*.jsonl"):
    relation = os.path.basename(path).replace(".jsonl", "")
    if relation not in all_rels:
        continue
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("evidences"): continue
            obj_label = d["obj_label"].strip()
            toks = tokenizer.tokenize(obj_label)
            if len(toks) != 1: continue
            answer_id = tokenizer.convert_tokens_to_ids(toks[0])
            rel_answers[relation].add(answer_id)

print("Computing mean pairwise cosine similarities...")
rows = []
for rel in all_rels:
    ids = list(rel_answers[rel])
    if len(ids) < 2:
        continue
    vecs = F.normalize(E[ids].float(), dim=-1)
    sim_matrix = vecs @ vecs.T
    mask = ~torch.eye(len(ids), dtype=torch.bool)
    mean_sim = sim_matrix[mask].mean().item()
    rows.append({
        "relation": rel,
        "label":    RELATION_LABELS.get(rel, rel),
        "gain":     pivot.loc[rel, "gain"],
        "n_test":   int(pivot.loc[rel, "n"]),
        "n_ans":    len(ids),
        "mean_cos": mean_sim,
    })

results = pd.DataFrame(rows).sort_values("gain", ascending=False)
print(results[["label", "gain", "n_ans", "mean_cos"]].to_string(index=False))

# Correlation
r = results[["gain", "mean_cos"]].corr().loc["gain", "mean_cos"]
print(f"\nPearson r(gain, mean_cos) = {r:.3f}")

# Scatter plot
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(results["mean_cos"], results["gain"],
                c=results["gain"], cmap="RdYlGn",
                s=results["n_test"] * 0.8, alpha=0.85,
                vmin=-0.08, vmax=0.12, edgecolors="white", linewidths=0.5)

# Label every point
for _, row in results.iterrows():
    ax.annotate(f"{row['label']}", (row["mean_cos"], row["gain"]),
                fontsize=7.5, ha="left", va="bottom",
                xytext=(4, 3), textcoords="offset points")

ax.axhline(0, color="black", lw=0.8, alpha=0.5)
ax.set_xlabel("Mean pairwise cosine similarity of answer embeddings", fontsize=11)
ax.set_ylabel("Accuracy gain: cosine λ=1.0 vs CE-only", fontsize=11)
ax.set_title(f"Answer embedding compactness vs accuracy gain  ·  LAMA T-REx\n"
             f"Pearson r = {r:.3f}  (dot size ∝ n test examples)", fontsize=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1%}"))
plt.colorbar(sc, ax=ax, label="accuracy gain")
plt.tight_layout()

out = os.path.join(SCRIPT_DIR, "lama_embedding_compactness.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
