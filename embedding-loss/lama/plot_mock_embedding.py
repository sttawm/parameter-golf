#!/usr/bin/env python3
import matplotlib.pyplot as plt

BASE = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama"

points = {
    "difficult":   (0.70, 0.52),
    "challenging": (0.43, 0.79),
    "purple":      (-0.71, -0.76),
}
colors = {
    "difficult":   "#e15759",
    "challenging": "#f28e2b",
    "purple":      "#b07ecf",
}
offsets = {
    "difficult":   (0.05, -0.07),
    "challenging": (-0.08, 0.06),
    "purple":      (0.05, -0.08),
}

import math

cx, cy = points["challenging"]

# Euclidean distances from "challenging"
distances = {w: round(math.sqrt((x - cx)**2 + (y - cy)**2), 2)
             for w, (x, y) in points.items()}
distances["challenging"] = 0.00

# label offsets for distance text — placed along midpoint of each line, nudged perp.
dist_label_offsets = {
    "difficult":   ( 0.12,  0.12),
    "purple":      ( 0.18, -0.10),
}

versions = [
    ("mock_embedding_1.png", ["challenging"],                        ["challenging"]),
    ("mock_embedding_2.png", ["challenging", "difficult"],           ["challenging", "difficult"]),
    ("mock_embedding_3.png", ["challenging", "difficult", "purple"], ["challenging", "difficult", "purple"]),
]

for filename, words, dist_words in versions:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")

    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.grid(True, color="#eeeeee", linewidth=0.6, zorder=0)

    # draw lines from challenging to each visible word
    for word in dist_words:
        if word == "challenging": continue
        x2, y2 = points[word]
        ax.plot([cx, x2], [cy, y2], color="#bbbbbb", lw=1.2,
                linestyle="dashed", zorder=2)

    for word in words:
        x, y = points[word]
        ax.scatter(x, y, s=180, color=colors[word], zorder=5,
                   edgecolors="white", linewidths=1.5)
        dx, dy = offsets[word]
        ax.text(x + dx, y + dy, word, fontsize=13, fontweight="bold",
                color=colors[word], ha="left" if dx > 0 else "right", va="center",
                zorder=6)

    # distance labels at midpoint of each line (skip challenging — implicit 0)
    for word in dist_words:
        if word == "challenging": continue
        x2, y2 = points[word]
        lx, ly = (cx + x2) / 2, (cy + y2) / 2
        ddx, ddy = dist_label_offsets[word]
        d = distances[word]
        ax.text(lx + ddx, ly + ddy, f"d = {d:.2f}", fontsize=10,
                color=colors[word], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[word],
                          alpha=0.9, lw=0.8),
                zorder=7)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Mock Embedding Space", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")

    plt.tight_layout()
    out = f"{BASE}/{filename}"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")
