#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

colors = {
    "challenging": "#f28e2b",
    "difficult":   "#e15759",
    "purple":      "#b07ecf",
}

points = {
    "difficult":   (0.70, 0.52),
    "challenging": (0.43, 0.79),
    "purple":      (-0.71, -0.76),
}

cx, cy = points["challenging"]
distances = {w: round(math.sqrt((x-cx)**2+(y-cy)**2), 2) for w,(x,y) in points.items()}
distances["challenging"] = 0.00

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor("white")

# ── Left text panel ──────────────────────────────────────────────────────────
ax = fig.add_axes([0.01, 0.0, 0.47, 1.0])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.04, 0.91, "Overview of Method", fontsize=26,
        fontweight="normal", color="black", va="center")

rows = [
    ("Ground Truth", "challenging", "0", "+ 0.0"),
    ("Prediction A", "difficult",   "1", "+ 0.38"),
    ("Prediction B", "purple",      "1", "+ 1.92"),
]
y_pos = [0.72, 0.57, 0.42]

# column header
ax.text(0.76, 0.83, "CE", fontsize=13, color="#888888",
        va="center", ha="center", fontweight="bold")
ax.text(0.90, 0.83, "Similarity", fontsize=13, color="#888888",
        va="center", ha="center", fontweight="bold")
ax.axhline(0.795, xmin=0.72, xmax=0.99, color="#dddddd", lw=0.8)

for (label, word, ce, sim), y in zip(rows, y_pos):
    # role label
    ax.text(0.04, y, label, fontsize=15, color="#999999",
            va="center", style="italic")
    # "This class is "
    ax.text(0.26, y, "This class is ", fontsize=19,
            color="#333333", va="center")
    # colored word
    ax.text(0.575, y, word, fontsize=19, fontweight="bold",
            color=colors[word], va="center")
    # CE column
    ax.text(0.76, y, ce, fontsize=15, color="#555555",
            va="center", ha="center", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5",
                      ec="#dddddd", lw=0.8))
    # Similarity column
    ax.text(0.90, y, sim, fontsize=15, color="#555555",
            va="center", ha="center", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5",
                      ec="#dddddd", lw=0.8))

# divider line between rows
for y in [0.645, 0.495]:
    ax.axhline(y, xmin=0.03, xmax=0.97, color="#eeeeee", lw=0.8)

# bottom text
ax.text(0.04, 0.30, "Reward similarity:", fontsize=18, color="#333333")
ax.text(0.04, 0.19, "Loss = CE + ?", fontsize=18, color="#333333")
ax.text(0.04, 0.08, "Loss = CE + embedding_similarity(gt, pred)",
        fontsize=15, color="#333333", family="monospace")

# ── Embedding chart ──────────────────────────────────────────────────────────
ax2 = fig.add_axes([0.50, 0.06, 0.47, 0.87])
ax2.set_facecolor("#f9f9f9")
ax2.axhline(0, color="#cccccc", lw=0.8, zorder=1)
ax2.axvline(0, color="#cccccc", lw=0.8, zorder=1)
ax2.grid(True, color="#eeeeee", linewidth=0.6, zorder=0)

# lines from challenging
for word, (x2, y2) in points.items():
    if word == "challenging": continue
    ax2.plot([cx, x2], [cy, y2], color="#bbbbbb", lw=1.2,
             linestyle="dashed", zorder=2)

# distance labels
dist_label_offsets = {
    "difficult": (0.12, 0.12),
    "purple":    (0.18, -0.10),
}
for word, (x2, y2) in points.items():
    if word == "challenging": continue
    lx, ly = (cx+x2)/2, (cy+y2)/2
    ddx, ddy = dist_label_offsets[word]
    ax2.text(lx+ddx, ly+ddy, f"d = {distances[word]:.2f}", fontsize=9,
             color=colors[word], fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=colors[word], alpha=0.9, lw=0.8), zorder=7)

# dots and labels
offsets = {"difficult": (0.05, -0.07), "challenging": (-0.08, 0.06),
           "purple": (0.05, -0.08)}
for word, (x, y) in points.items():
    ax2.scatter(x, y, s=180, color=colors[word], zorder=5,
                edgecolors="white", linewidths=1.5)
    dx, dy = offsets[word]
    ax2.text(x+dx, y+dy, word, fontsize=12, fontweight="bold",
             color=colors[word], ha="left" if dx > 0 else "right",
             va="center", zorder=6)

ax2.set_xlim(-1.15, 1.15)
ax2.set_ylim(-1.15, 1.15)
ax2.set_xlabel("Dimension 1", fontsize=10, color="#666")
ax2.set_ylabel("Dimension 2", fontsize=10, color="#666")
ax2.set_title("Mock Embedding Space", fontsize=13, fontweight="bold", pad=10)
ax2.tick_params(colors="#aaa", labelsize=8)
for spine in ax2.spines.values():
    spine.set_edgecolor("#dddddd")

out = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/slide_overview.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
