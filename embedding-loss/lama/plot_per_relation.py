#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

RELATION_LABELS = {
    "P17":   "country",
    "P19":   "place of birth",
    "P20":   "place of death",
    "P27":   "country of citizenship",
    "P30":   "continent",
    "P31":   "instance of",
    "P36":   "capital",
    "P37":   "official language",
    "P39":   "position held",
    "P47":   "shares border with",
    "P101":  "field of work",
    "P103":  "native language",
    "P106":  "occupation",
    "P108":  "employer",
    "P127":  "owned by",
    "P131":  "located in",
    "P136":  "genre",
    "P138":  "named after",
    "P140":  "religion",
    "P159":  "headquarters location",
    "P176":  "manufacturer",
    "P178":  "developer",
    "P190":  "sister city",
    "P264":  "record label",
    "P276":  "location",
    "P279":  "subclass of",
    "P361":  "part of",
    "P364":  "original language",
    "P407":  "language of work",
    "P413":  "position played",
    "P449":  "original network",
    "P463":  "member of",
    "P495":  "country of origin",
    "P527":  "has part",
    "P530":  "diplomatic relation",
    "P740":  "location of formation",
    "P937":  "work location",
    "P1001": "applies to jurisdiction",
    "P1303": "instrument",
    "P1376": "capital of",
    "P1412": "languages spoken",
}

df = pd.read_csv("/Users/sttawm/dev/parameter-golf/embedding-loss/lama/lama_per_relation.csv")

conditions = df["condition"].unique()
print("Conditions:", conditions)

zs_col  = "zero-shot"
l2_col  = [c for c in conditions if "L2" in c][0]
cos_col = [c for c in conditions if "cosine" in c][0]

pivot = df.pivot(index="relation", columns="condition", values="acc")
baseline_col = l2_col  # CE-only fine-tuned (λ=0.0)
pivot["gain_cos"] = pivot[cos_col] - pivot[baseline_col]
pivot["n"]        = df[df["condition"] == zs_col].set_index("relation")["n"]
pivot = pivot[pivot["n"] >= 20].sort_values("gain_cos", ascending=False)

pivot.index = [f"{RELATION_LABELS.get(r, r)}  ({r}, n={int(pivot.loc[r,'n'])})" for r in pivot.index]

top_n = 15
sorted_gains = pivot["gain_cos"].sort_values(ascending=False)
top_data = sorted_gains.head(top_n)
bot_data = sorted_gains.tail(top_n).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle(f"Per-relation accuracy: {cos_col}  vs  CE-only  ·  LAMA T-REx", fontsize=13)

for ax, data, title in [
    (axes[0], top_data,  f"Top {top_n} gainers"),
    (axes[1], bot_data,  f"Top {top_n} losers"),
]:
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in data.values]
    ax.barh(data.index[::-1], data.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Accuracy gain over CE-only fine-tuning")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.1%}"))
    ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
out = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/lama_per_relation_vs_baseline.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
