#!/usr/bin/env python3
"""val_bpb vs wall-clock time for the 8×H100 baseline vs CE+Embed+Uniform run."""

import re
from pathlib import Path
import matplotlib.pyplot as plt

LOG_DIR = Path("logs")

val_re = re.compile(r"^step:(\d+)/\d+ val_loss:[\d.]+ val_bpb:([\d.]+) train_time:(\d+)ms")

def parse(path):
    times_s, bpb = [], []
    with open(path) as f:
        for line in f:
            m = val_re.match(line.strip())
            if m and int(m.group(1)) > 0:   # skip step 0
                bpb.append(float(m.group(2)))
                times_s.append(int(m.group(3)) / 1000.0)
    return times_s, bpb

baseline = LOG_DIR / "baseline_8gpu_20260430_233026_s1337.txt"
ceeu     = LOG_DIR / "ceeu_8gpu_20260430_234234_lam1_g2_s1337.txt"

bt, bv = parse(baseline)
ct, cv = parse(ceeu)

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("white")

ax.plot(bt, bv, color="tab:blue",   linewidth=2, marker="o", markersize=4,
        label="Baseline  (13,682 steps in 10 min)")
ax.plot(ct, cv, color="tab:orange", linewidth=2, marker="o", markersize=4,
        label="CE + Embed + Uniform  (7,288 steps in 10 min)")

ax.axvline(600, color="gray", linestyle="--", linewidth=1, label="10-min budget")

# Annotate final values
for times, vals, color, ha in [(bt, bv, "tab:blue", "right"), (ct, cv, "tab:orange", "left")]:
    t_end, v_end = times[-1], vals[-1]
    offset = -8 if ha == "right" else 8
    ax.text(t_end + offset, v_end + 0.004, f"{v_end:.4f}",
            fontsize=8, color=color, va="bottom", ha=ha)

ax.set_xlim(0, 660)
ax.set_ylim(1.18, 1.42)
ax.set_xlabel("Wall-clock time (s)", fontsize=10)
ax.set_ylabel("val_bpb", fontsize=10)
ax.set_title("Training curves vs wall-clock time",
             fontsize=11, fontweight="bold", pad=10)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.25)
ax.tick_params(labelsize=9)

plt.tight_layout()
out = Path("8gpu_wallclock_plot.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
