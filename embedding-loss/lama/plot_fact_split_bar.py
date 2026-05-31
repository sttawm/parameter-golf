#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LAMA_DIR = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama"

df = pd.read_csv(f"{LAMA_DIR}/lama_results_cosine_multi.csv")
df["lambda"]   = pd.to_numeric(df["lambda"],   errors="coerce")
df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")

# Fine-tuned overall (λ=0, random split — test+train share facts)
ft_vals   = df[(df["lambda"] == 0.0) & (df["epoch"] == "final")]["test_acc"].dropna()
ft_mean   = ft_vals.mean()
ft_se     = ft_vals.std(ddof=1) / np.sqrt(len(ft_vals))

# Zero-shot
zs_vals   = pd.to_numeric(
    df[pd.to_numeric(df["epoch"], errors="coerce") == 0]["test_acc"], errors="coerce"
).dropna()
zs_mean   = zs_vals.mean() if len(zs_vals) else 0.508

ft_shared = 0.956  # fine-tuned, evaluated on shared-fact test examples only

# ── Plot: 3 bars ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.8))
fig.patch.set_facecolor("white")

BAR_W = 0.42
xs      = [0.0,    0.65,    1.40]
heights = [zs_mean, ft_mean, ft_shared]
errors  = [0,       ft_se,   0]
colors  = ["#e0e0e0", "#a5d6a7", "#f8d7da"]
labels  = ["Zero-shot",
           "Fine-tuned\n(test & train\ndon't share facts)",
           "Fine-tuned\n(test & train\ndo share facts)"]

bar_labels = [f"{h:.0%}" for h in heights]
bar_labels[-1] = "> 95%"

for xi, h, e, c, lbl in zip(xs, heights, errors, colors, bar_labels):
    ax.bar(xi, h, width=BAR_W, color=c, edgecolor="#888", linewidth=0.6,
           yerr=e or None, capsize=4,
           error_kw=dict(elinewidth=1.3, ecolor="#555"), zorder=3)
    ax.text(xi, h + (e or 0) + 0.012, lbl,
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_xticks(xs)
ax.set_xticklabels([])   # draw manually for mixed bold
ax.tick_params(bottom=False)

# Tick labels with "do"/"don't" bolded via separate text calls
trans = ax.get_xaxis_transform()   # x=data, y=axes fraction

# Simple labels: just text
ax.text(xs[0], -0.06, "Zero-shot", transform=trans,
        ha="center", va="top", fontsize=9.5)

# For xs[1] and xs[2]: 3-line label where last line has a bold word at start
for xi, line3_bold, line3_rest in [
    (xs[1], "don't", " share facts"),
    (xs[2], "do",    " share facts"),
]:
    ax.text(xi, -0.06, "Fine-tuned",    transform=trans, ha="center", va="top",  fontsize=9.5)
    ax.text(xi, -0.13, "(test & train", transform=trans, ha="center", va="top",  fontsize=9.5)

    # Draw last line: measure bold word width then place normal text after it
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_w = ax.get_window_extent(renderer).width

    full_line = line3_bold + line3_rest
    # Place full line invisibly to find center offset
    t_full = ax.text(xi, -0.20, full_line, transform=trans,
                     ha="center", va="top", fontsize=9.5, alpha=0)
    fig.canvas.draw()
    bb_full = t_full.get_window_extent(renderer)
    x0_px = bb_full.x0   # pixel x where full string starts (left-aligned from center)

    # Measure bold word width
    t_bold = ax.text(0, -999, line3_bold, transform=trans,
                     ha="left", va="top", fontsize=9.5, fontweight="bold", alpha=0)
    fig.canvas.draw()
    bold_w_px = t_bold.get_window_extent(renderer).width

    # Convert x0_px back to data coords
    x0_data = ax.transData.inverted().transform(
        ax.transAxes.inverted().transform(
            [[x0_px / fig.get_dpi() / fig.get_size_inches()[0], 0]]
        )
    )
    # Simpler: work in display (pixel) coords, convert to data at the end
    ax_bbox = ax.get_window_extent(renderer)
    x0_frac = (x0_px - ax_bbox.x0) / ax_bbox.width   # fraction of axes width

    # Place bold word at x0_frac, normal text right after
    t_bold.set_alpha(1)
    t_bold.set_position((x0_frac, -0.20))
    t_bold.set_transform(ax.transAxes)
    t_bold.set_horizontalalignment("left")

    x1_frac = x0_frac + bold_w_px / ax_bbox.width
    ax.text(x1_frac, -0.20, line3_rest, transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5)

    t_full.set_visible(False)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylabel("Test accuracy (LAMA T-REx)", fontsize=10)
ax.set_title("BERT-base · T-REx  ·  accuracy by fact overlap",
             fontsize=11, fontweight="bold", pad=10)
ax.set_ylim(0.35, 1.06)
ax.grid(axis="y", alpha=0.2, zorder=0)

plt.tight_layout()
out = f"{LAMA_DIR}/fact_split_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
