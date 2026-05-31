#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SENTENCES = [
    ("It is located on the east coast of the Jutland peninsula,",
     "in the geographical centre of Denmark, 187 km northwest of [MASK]."),
    ("[MASK] is home to the University of Copenhagen, the Technical",
     "University of Denmark and Copenhagen Business School."),
    ("Copenhagen is home to the University of [MASK],",
     "the Technical University of Denmark and Copenhagen Business School."),
    ("The city is located in central Denmark, 187 kilometres",
     "northwest of [MASK], and 289 kilometres north of Hamburg, Germany."),
]

MASK_COLOR = "#c0392b"
FS   = 8.8
X0   = 0.108
fig, ax = plt.subplots(figsize=(8.5, 5.6))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

ax.text(0.5, 0.97, "Same fact, different sentences",
        ha="center", va="top", fontsize=13, fontweight="bold", color="#222")

# ── Fact badges ───────────────────────────────────────────────────────────────
for bx, title, val, fc in [
    (0.18, "Predicate", "P36  ·  capital of", "#dce8f5"),
    (0.52, "Subject",   "Denmark",             "#fff3cd"),
    (0.82, "Answer",    "Copenhagen",           "#d4edda"),
]:
    ax.add_patch(mpatches.FancyBboxPatch(
        (bx - 0.14, 0.833), 0.28, 0.082,
        boxstyle="round,pad=0.015", fc=fc, ec="#bbb", lw=0.8, transform=ax.transAxes, zorder=2))
    ax.text(bx, 0.882, title, ha="center", va="center", fontsize=7.5, color="#777", transform=ax.transAxes)
    ax.text(bx, 0.858, val,   ha="center", va="center", fontsize=10, fontweight="bold", color="#222", transform=ax.transAxes)

# ── Measure space width once ──────────────────────────────────────────────────
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
ax_bb    = ax.get_window_extent(renderer=renderer)

def _w(t):
    fig.canvas.draw()
    return t.get_window_extent(renderer=renderer).width / ax_bb.width

# Measure a space by comparing "| |" vs "||"
_t1 = ax.text(0, -9, "| |", fontsize=FS, transform=ax.transAxes)
_t2 = ax.text(0, -9, "||",  fontsize=FS, transform=ax.transAxes)
SPACE_W = (_w(_t1) - _w(_t2))
_t1.remove(); _t2.remove()

def render_with_mask(y, text):
    """Render one line of text with [MASK] in red, tracking widths correctly."""
    cx = X0
    parts = text.split("[MASK]")
    for j, part in enumerate(parts):
        # Count and strip leading/trailing spaces
        n_lead  = len(part) - len(part.lstrip(" "))
        n_trail = len(part) - len(part.rstrip(" "))
        core    = part.strip(" ")
        cx += n_lead * SPACE_W
        if core:
            t = ax.text(cx, y, core, ha="left", va="center",
                        fontsize=FS, color="#333", transform=ax.transAxes, zorder=3)
            cx += _w(t)
        cx += n_trail * SPACE_W
        if j < len(parts) - 1:
            t = ax.text(cx, y, "[MASK]", ha="left", va="center",
                        fontsize=FS, color=MASK_COLOR, fontweight="bold",
                        transform=ax.transAxes, zorder=3)
            cx += _w(t)

# ── Sentence rows ─────────────────────────────────────────────────────────────
row_ys = [0.745, 0.590, 0.435, 0.280]
ROW_H  = 0.128

for i, ((l1, l2), ry) in enumerate(zip(SENTENCES, row_ys)):
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.04, ry - ROW_H/2), 0.92, ROW_H,
        boxstyle="round,pad=0.008", fc="#f9f9f9", ec="#e0e0e0", lw=0.7,
        transform=ax.transAxes, zorder=1))
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.053, ry - 0.021), 0.030, 0.042,
        boxstyle="round,pad=0.004", fc="#e4e4e4", ec="none",
        transform=ax.transAxes, zorder=2))
    ax.text(0.068, ry, str(i + 1), ha="center", va="center",
            fontsize=8.5, color="#555", fontweight="bold",
            transform=ax.transAxes, zorder=3)
    render_with_mask(ry + 0.027, l1)
    render_with_mask(ry - 0.027, l2)

ax.text(0.5, 0.028,
        "All four sentences encode the same (Denmark, capital of, Copenhagen) triple.  "
        "With a random split, variants of the same fact appear in both training and test sets.",
        ha="center", va="bottom", fontsize=7.8, color="#999", style="italic",
        transform=ax.transAxes)

plt.tight_layout(pad=0.3)
out = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/fact_examples.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
