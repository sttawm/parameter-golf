#!/usr/bin/env python3
"""
Plot CE loss, embed loss, and embedding norm for embed-loss-only runs.

Usage:
    python plot_embed_only.py logs/eo_*.txt
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

train_re   = re.compile(r"step:(\d+)/\d+ train_loss:([\d.]+)")
detail_re  = re.compile(r"step:(\d+) lambda:([\d.]+) ce:([\d.]+) embed_only:True emb_norm:([\d.]+)")


def parse_log(path: Path):
    embed_loss, ce, norm = {}, {}, {}
    with open(path) as f:
        for line in f:
            m = train_re.search(line)
            if m:
                embed_loss[int(m.group(1))] = float(m.group(2))
                continue
            m = detail_re.search(line)
            if m:
                step = int(m.group(1))
                ce[step]   = float(m.group(3))
                norm[step] = float(m.group(4))
    return embed_loss, ce, norm


def main(paths):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Embed-loss-only training — UNTIED weights", fontsize=13, fontweight="bold")

    for path in paths:
        embed_loss, ce, norm = parse_log(path)

        steps_e = sorted(embed_loss)
        steps_c = sorted(ce)
        steps_n = sorted(norm)

        axes[0].plot(steps_c, [ce[s] for s in steps_c], color="steelblue", linewidth=1.5)
        axes[0].axhline(y=6.9315, color="gray", linestyle="--", linewidth=1, label="init ≈ log(1024)")

        axes[1].plot(steps_e, [embed_loss[s] for s in steps_e], color="firebrick", linewidth=1.5)

        axes[2].plot(steps_n, [norm[s] for s in steps_n], color="darkorange", linewidth=1.5)

    axes[0].set_title("Train CE loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("CE loss (nats)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Embed loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("1 − cos(ê, e_gt)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Embedding norm")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Mean embedding L2 norm")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path("embed_only_untied_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main([Path(p) for p in sys.argv[1:]])
