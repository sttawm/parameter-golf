#!/usr/bin/env python3
"""
Parse sweep log files and plot CE loss, embed loss, and ms/step.

Usage:
    python plot_sweep.py logs/sweep_*.txt
    python plot_sweep.py logs/sweep_*.txt logs/baseline_run.txt
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


# ── parser ────────────────────────────────────────────────────────────────────

def parse_log(path: Path) -> dict:
    """
    Returns dict with keys:
        label, lambda, steps (list), ce (list), embed (list), step_avg_ms (list)
    embed is None for baseline runs.
    """
    train_re   = re.compile(
        r"step:(\d+)/\d+ train_loss:([\d.]+).*step_avg:([\d.]+)ms"
    )
    comps_re   = re.compile(
        r"step:(\d+) lambda:([\d.]+) ce:([\d.]+) embed:([\d.]+)"
    )
    lambda_re  = re.compile(r"embed_loss_lambda:([\d.]+)")
    run_id_re  = re.compile(r"run_id:(\S+)")

    embed_lambda = 0.0
    run_id = path.stem

    # step → {train_loss, ce, embed, step_avg_ms}
    rows: dict[int, dict] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = lambda_re.search(line)
            if m:
                embed_lambda = float(m.group(1))
                continue
            m = run_id_re.search(line)
            if m:
                run_id = m.group(1)
                continue
            m = train_re.search(line)
            if m:
                step = int(m.group(1))
                rows.setdefault(step, {})
                rows[step]["train_loss"] = float(m.group(2))
                rows[step]["step_avg_ms"] = float(m.group(3))
                continue
            m = comps_re.search(line)
            if m:
                step = int(m.group(1))
                rows.setdefault(step, {})
                rows[step]["ce"] = float(m.group(3))
                rows[step]["embed"] = float(m.group(4))

    steps, ce_vals, embed_vals, timing_vals = [], [], [], []
    for step in sorted(rows):
        row = rows[step]
        if "step_avg_ms" not in row:
            continue
        steps.append(step)
        if embed_lambda > 0.0 and "ce" in row:
            ce_vals.append(row["ce"])
            embed_vals.append(row.get("embed"))
        else:
            ce_vals.append(row["train_loss"])
            embed_vals.append(None)
        timing_vals.append(row["step_avg_ms"])

    if embed_lambda == 0.0:
        label = "baseline"
    else:
        label = f"λ={embed_lambda}"

    return dict(
        label=label,
        lam=embed_lambda,
        steps=steps,
        ce=ce_vals,
        embed=embed_vals,
        step_avg_ms=timing_vals,
        run_id=run_id,
    )


# ── plotting ──────────────────────────────────────────────────────────────────

def main(paths: list[Path]) -> None:
    runs = [parse_log(p) for p in paths]
    # sort by lambda so legend is ordered
    runs.sort(key=lambda r: r["lam"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("Embedding-loss lambda sweep", fontsize=14, fontweight="bold")

    ax_ce, ax_emb, ax_ms = axes

    colors = plt.cm.tab10.colors

    for i, run in enumerate(runs):
        color = colors[i % len(colors)]
        label = run["label"]
        steps = run["steps"]

        ax_ce.plot(steps, run["ce"], label=label, color=color, linewidth=1.5)

        embed_vals = run["embed"]
        if any(v is not None for v in embed_vals):
            clean_steps = [s for s, v in zip(steps, embed_vals) if v is not None]
            clean_vals  = [v for v in embed_vals if v is not None]
            ax_emb.plot(clean_steps, clean_vals, label=label, color=color, linewidth=1.5)

        ax_ms.plot(steps, run["step_avg_ms"], label=label, color=color, linewidth=1.5)

    ax_ce.set_ylabel("CE loss")
    ax_ce.legend(loc="upper right")
    ax_ce.grid(True, alpha=0.3)

    ax_emb.set_ylabel("Embed loss (1 − cosine)")
    ax_emb.legend(loc="upper right")
    ax_emb.grid(True, alpha=0.3)

    ax_ms.set_ylabel("ms / step (cumulative avg)")
    ax_ms.set_xlabel("Step")
    ax_ms.legend(loc="upper right")
    ax_ms.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path("sweep_plot.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main([Path(p) for p in sys.argv[1:]])
