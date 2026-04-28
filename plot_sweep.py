#!/usr/bin/env python3
"""
Parse sweep log files and plot CE loss, embed loss, ms/step, and val_bpb.
Shows each metric twice: vs steps (left) and vs wall-clock seconds (right).

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
    train_re  = re.compile(
        r"step:(\d+)/\d+ train_loss:([\d.]+).*train_time:(\d+)ms.*step_avg:([\d.]+)ms"
    )
    val_re    = re.compile(
        r"^step:(\d+)/\d+ val_loss:[\d.]+ val_bpb:([\d.]+).*train_time:(\d+)ms"
    )
    comps_re  = re.compile(
        r"step:(\d+) lambda:([\d.]+) ce:([\d.]+) embed:([\d.]+)"
    )
    lambda_re = re.compile(r"embed_loss_lambda:([\d.]+)")
    l2_re     = re.compile(r"embed_loss_l2:(\d)")
    topk_re   = re.compile(r"embed_loss_topk:(\d+)")
    tied_re   = re.compile(r"tie_embeddings:(True|False|0|1)")
    run_id_re = re.compile(r"run_id:(\S+)")

    embed_lambda = 0.0
    embed_l2 = False
    embed_topk = 0
    tied = True
    run_id = path.stem

    # step → {train_loss, train_time_ms, ce, embed, step_avg_ms, val_bpb, val_time_ms}
    rows: dict[int, dict] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Config fields may co-appear on the same line — parse without continue.
            is_config = False
            m = lambda_re.search(line)
            if m:
                embed_lambda = float(m.group(1))
                is_config = True
            m = l2_re.search(line)
            if m:
                embed_l2 = m.group(1) in ("1", "True")
                is_config = True
            m = topk_re.search(line)
            if m:
                embed_topk = int(m.group(1))
                is_config = True
            m = tied_re.search(line)
            if m:
                tied = m.group(1) in ("1", "True")
                is_config = True
            m = run_id_re.search(line)
            if m:
                run_id = m.group(1)
                is_config = True
            if is_config:
                continue
            m = val_re.match(line)
            if m:
                step = int(m.group(1))
                rows.setdefault(step, {})
                rows[step]["val_bpb"] = float(m.group(2))
                rows[step]["val_time_ms"] = int(m.group(3))
                continue
            m = train_re.search(line)
            if m:
                step = int(m.group(1))
                rows.setdefault(step, {})
                rows[step]["train_loss"] = float(m.group(2))
                rows[step]["train_time_ms"] = int(m.group(3))
                rows[step]["step_avg_ms"] = float(m.group(4))
                continue
            m = comps_re.search(line)
            if m:
                step = int(m.group(1))
                rows.setdefault(step, {})
                rows[step]["ce"] = float(m.group(3))
                rows[step]["embed"] = float(m.group(4))

    steps, times_s, ce_vals, embed_vals, timing_vals = [], [], [], [], []
    val_steps, val_times_s, val_bpb_vals = [], [], []

    for step in sorted(rows):
        row = rows[step]
        if "val_bpb" in row:
            val_steps.append(step)
            val_times_s.append(row.get("val_time_ms", 0) / 1000.0)
            val_bpb_vals.append(row["val_bpb"])
        if "step_avg_ms" not in row:
            continue
        steps.append(step)
        times_s.append(row["train_time_ms"] / 1000.0)
        if embed_lambda > 0.0 and "ce" in row:
            ce_vals.append(row["ce"])
            embed_vals.append(row.get("embed"))
        else:
            ce_vals.append(row["train_loss"])
            embed_vals.append(None)
        timing_vals.append(row["step_avg_ms"])

    # Build a descriptive label
    max_train_s = max((rows[s].get("train_time_ms", 0) for s in rows), default=0) / 1000
    duration_tag = f" ({int(round(max_train_s / 60))}min)" if max_train_s > 10 else ""
    tie_tag = "tied" if tied else "untied"
    if embed_lambda == 0.0:
        label = f"baseline/{tie_tag}{duration_tag}"
    else:
        loss_tag = "L2" if embed_l2 else "cos"
        topk_tag = f" K={embed_topk}" if embed_topk > 0 else ""
        label = f"λ={embed_lambda} {loss_tag}{topk_tag}/{tie_tag}{duration_tag}"

    return dict(
        label=label,
        lam=embed_lambda,
        l2=embed_l2,
        topk=embed_topk,
        tied=tied,
        steps=steps,
        times_s=times_s,
        ce=ce_vals,
        embed=embed_vals,
        step_avg_ms=timing_vals,
        val_steps=val_steps,
        val_times_s=val_times_s,
        val_bpb=val_bpb_vals,
        run_id=run_id,
    )


# ── plotting ──────────────────────────────────────────────────────────────────

METRICS = [
    ("CE loss",               "ce",          "embed",        False),
    ("Embed loss",            "embed",        None,           False),
    ("ms / step (cum. avg)",  "step_avg_ms",  None,           False),
    ("val_bpb",               None,           None,           True),   # val only
]

def main(paths: list[Path]) -> None:
    runs = [parse_log(p) for p in paths]
    runs.sort(key=lambda r: (r["l2"], not r["tied"], r["topk"], r["lam"]))

    n_rows = 4
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 14),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.25})
    fig.suptitle("Embedding-loss lambda sweep", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors

    for i, run in enumerate(runs):
        color  = colors[i % len(colors)]
        label  = run["label"]
        steps  = run["steps"]
        times  = run["times_s"]

        # ── row 0: CE loss ──
        ax_l, ax_r = axes[0]
        ax_l.plot(steps, run["ce"], label=label, color=color, linewidth=1.5)
        ax_r.plot(times, run["ce"], label=label, color=color, linewidth=1.5)

        # ── row 1: embed loss ──
        ax_l, ax_r = axes[1]
        ev = run["embed"]
        if any(v is not None for v in ev):
            cs = [s for s, v in zip(steps, ev) if v is not None]
            ct = [t for t, v in zip(times,  ev) if v is not None]
            cv = [v for v in ev if v is not None]
            ax_l.plot(cs, cv, label=label, color=color, linewidth=1.5)
            ax_r.plot(ct, cv, label=label, color=color, linewidth=1.5)

        # ── row 2: ms/step ──
        ax_l, ax_r = axes[2]
        ax_l.plot(steps, run["step_avg_ms"], label=label, color=color, linewidth=1.5)
        ax_r.plot(times, run["step_avg_ms"], label=label, color=color, linewidth=1.5)

        # ── row 3: val_bpb ──
        ax_l, ax_r = axes[3]
        if run["val_steps"]:
            ax_l.plot(run["val_steps"],  run["val_bpb"], label=label, color=color,
                      linewidth=1.5, marker="o", markersize=5)
            ax_r.plot(run["val_times_s"], run["val_bpb"], label=label, color=color,
                      linewidth=1.5, marker="o", markersize=5)

    ylabels = ["CE loss", "Embed loss", "ms / step (cum. avg)", "val_bpb"]
    for row, ylabel in enumerate(ylabels):
        for col in range(2):
            ax = axes[row][col]
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, alpha=0.3)

    for col, xlabel in enumerate(["Step", "Wall-clock time (s)"]):
        axes[n_rows - 1][col].set_xlabel(xlabel)

    # column titles
    axes[0][0].set_title("vs Steps", fontsize=11)
    axes[0][1].set_title("vs Wall-clock time", fontsize=11)

    out = Path("sweep_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main([Path(p) for p in sys.argv[1:]])
