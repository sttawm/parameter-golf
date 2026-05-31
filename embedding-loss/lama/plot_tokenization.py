#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

examples = [
    {
        "sentence": "The quarterback threw an interception in overtime.",
        "tokens":   ["the", "quarter", "##back", "threw", "an", "inter", "##ception", "in", "over", "##time", "."],
        "words":    ["The", "quarterback", "threw", "an", "interception", "in", "overtime", "."],
    },
    {
        "sentence": "The immunotherapy revolutionized oncological treatment.",
        "tokens":   ["the", "immuno", "##therapy", "revolution", "##ized", "onco", "##logical", "treatment", "."],
        "words":    ["The", "immunotherapy", "revolutionized", "oncological", "treatment", "."],
    },
]

alt_colors = ["#e9ecef", "#ffffff"]

try:
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    for ex in examples:
        ex["ids"] = tokenizer.convert_tokens_to_ids(ex["tokens"])
except Exception:
    for ex in examples:
        ex["ids"] = ["?"] * len(ex["tokens"])

n_examples = len(examples)
fig, axes = plt.subplots(n_examples * 2, 1, figsize=(14, n_examples * 2.8))
fig.suptitle("BERT WordPiece Tokenization  ·  bert-base-uncased", fontsize=12, fontweight="bold", y=1.01)

row_labels = ["Words", "Tokens"]

for ei, ex in enumerate(examples):
    tokens = ex["tokens"]
    words  = ex["words"]
    ids    = ex["ids"]
    n_cols = max(len(tokens), len(words))

    row_data = [words, tokens]
    row_n    = [len(words), len(tokens)]

    for ri, (label, data) in enumerate(zip(row_labels, row_data)):
        ax = axes[ei * 2 + ri]
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, 1)
        ax.axis("off")

        if ri == 0:
            ax.set_title(f'"{ex["sentence"]}"', fontsize=9, color="#444",
                         loc="left", pad=4, style="italic")

        ax.text(-0.005, 0.5, f"{label}:", va="center", ha="right", fontsize=9.5,
                color="#555", transform=ax.transAxes)

        for i, item in enumerate(data):
            col = alt_colors[i % 2]
            is_continuation = str(item).startswith("##")
            ec = "#888" if not is_continuation else "#c0392b"
            lw = 0.8 if not is_continuation else 1.8

            ax.add_patch(mpatches.FancyBboxPatch(
                (i + 0.04, 0.1), 0.88, 0.8,
                boxstyle="round,pad=0.04", linewidth=lw,
                edgecolor=ec, facecolor=col
            ))
            fs = 9
            display = item.lstrip("#") if label == "Tokens" else item
            ax.text(i + 0.48, 0.5, display,
                    ha="center", va="center", fontsize=fs,
                    fontstyle="italic" if is_continuation else "normal",
                    color="#c0392b" if is_continuation else "#222")

    # separator line between examples
    if ei < n_examples - 1:
        sep_ax = axes[ei * 2 + 1]
        sep_ax.axhline(y=-0.1, color="#ccc", lw=0.8, clip_on=False)

legend_patches = [
    mpatches.Patch(facecolor="#e9ecef", edgecolor="#888", label="Whole-word token"),
    mpatches.Patch(facecolor="#ffffff", edgecolor="#888", label="Whole-word token (alt)"),
    mpatches.Patch(facecolor="#ffffff", edgecolor="#c0392b", lw=2,
                   label="## continuation piece (sub-word)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.02), frameon=True)

plt.tight_layout(h_pad=0.6)
out = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/tokenization_diagram.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
