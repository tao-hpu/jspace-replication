"""B: the workspace developmental figure -- three curves vs training step.

Reads the B3 artifacts for one (or more) Pythia model and draws a 3-panel
figure on a log training-step axis:

  (1) effective dimension of the transported cone (centered): J vs raw. A
      widening gap = the J-lens compresses the workspace below the embedding.
  (2) coordinate drift cos(J_l^T W_U, W_U): mean over layers and deep-layer.
  (3) probe readability: median rank of the answer (deep layer) and multi-hop
      intermediate (mid layer) in the J-lens readout. Lower = more readable.

Run:  .venv/bin/python experiments/e11-development/make_dev_figures.py --model EleutherAI/pythia-410m-deduped
      (pass --model repeatedly to overlay families)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "figures"))
import plot_utils as pu  # noqa: E402

RES = ROOT / "results"
OUT = pathlib.Path(__file__).parent  # gitignored; figures/ created under it

# Mislabeled upstream Pythia checkpoints to exclude from the curves. The
# pythia-1.4b-deduped `step16` branch ships the step-143000 weights: all 292 tensors are
# elementwise identical to the endpoint (compare_step16_weights.py ->
# results/b3_step16_weight_check.json), while both immediate neighbours (step8, step32) sit
# at a centered PR of ~175. The checkpoint is dropped rather than plotted as a spurious dip.
BAD_CHECKPOINTS = {("pythia-1.4b-deduped", 16)}


def _drop_bad(tag: str, rows: list) -> list:
    kept = [r for r in rows if (tag, r["step"]) not in BAD_CHECKPOINTS]
    if len(kept) != len(rows):
        dropped = sorted(r["step"] for r in rows if (tag, r["step"]) in BAD_CHECKPOINTS)
        print(f"  {tag}: excluded mislabeled checkpoint(s) {dropped}")
    return kept


def load(stem: str, tag: str):
    p = RES / f"{stem}_{tag}.json"
    return json.load(open(p)) if p.exists() else None


def _boot_ci(rows: list, key: str, n_boot: int = 2000, seed: int = 0):
    """95% bootstrap CI (over probe items) of the median rank, per checkpoint.
    Returns (lo, hi) arrays, or None when raw ranks are absent (older JSONs)."""
    if not all(r[key].get("ranks") for r in rows):
        return None
    rng = np.random.default_rng(seed)
    lo, hi = [], []
    for r in rows:
        a = np.asarray(r[key]["ranks"])
        med = np.median(rng.choice(a, size=(n_boot, len(a))), axis=1)
        lo.append(np.percentile(med, 2.5))
        hi.append(np.percentile(med, 97.5))
    return np.array(lo), np.array(hi)


def _pow10_label(v, _pos=None) -> str:
    """Log-axis tick label in plain digits (1, 10, 100, 1k, 10k, 100k).
    Mathtext 10^k puts the exponent at 70% size, i.e. 6.3pt at a 9pt base,
    below the 9pt minimum for figure text; plain labels stay at 9pt."""
    if v <= 0:
        return ""
    k = int(round(np.log10(v)))
    if not np.isclose(v, 10.0 ** k):
        return ""
    return {0: "1", 1: "10", 2: "100"}.get(k, f"{10 ** (k - 3)}k")


def _x(rows):
    # step 0 has no log position; nudge to 0.5 so it plots on a log axis.
    return np.array([max(r["step"], 0.5) for r in rows], float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="repeatable: EleutherAI/pythia-410m-deduped ...")
    args = ap.parse_args()
    pu.setup()
    tags = [m.split("/")[-1] for m in args.model]
    colors = [pu.BLUE, pu.VERMILLION, pu.GREEN, pu.PURPLE]

    # 2x2 at the full text width (figure* in the paper): panels get ~3in each,
    # the model legend sits in a row above the axes so it covers no data, and
    # the shared step axis is labelled once, under the bottom row.
    fig, axes = plt.subplots(2, 2, figsize=(pu.FULL_W, 4.6), sharex=True)
    axes = axes.ravel()
    ax_dim, ax_drift, ax_ans, ax_int = axes

    for tag, c in zip(tags, colors):
        curves = load("b3_curves", tag)
        ranks = load("b3_probe_rank", tag)
        short = tag.replace("pythia-", "").replace("-deduped", "")
        if curves:
            rows = _drop_bad(tag, curves["rows"])
            x = _x(rows)
            # the figure-level model legend is built from the solid series of
            # panel (a); the caption explains line styles (solid J vs dashed raw)
            ax_dim.plot(x, [r["eff_dim_jl_centered"] for r in rows], "o-",
                        color=c, label=short)
            ax_dim.plot(x, [r["eff_dim_raw_centered"] for r in rows], "--",
                        color=c, alpha=0.45)
            ci = load("b3_effdim_ci", tag)
            if ci:
                crows = _drop_bad(tag, ci["rows"])
                ax_dim.fill_between(_x(crows),
                                    [r["eff_dim_jl_centered_ci95"][0] for r in crows],
                                    [r["eff_dim_jl_centered_ci95"][1] for r in crows],
                                    color=c, alpha=0.15, lw=0)
            ax_drift.plot(x, [r["mean_drift"] for r in rows], "o-", color=c)
            ax_drift.plot(x, [r["deep_drift"] for r in rows], ":", color=c,
                          alpha=0.55)
        if ranks:
            rr = _drop_bad(tag, ranks["rows"])
            xr = _x(rr)
            for ax, key in ((ax_ans, "answer"), (ax_int, "intermediate")):
                med = [r[key].get("median_rank", np.nan) for r in rr]
                ax.plot(xr, med, "o-", color=c)
                ci = _boot_ci(rr, key)
                if ci is not None:
                    ax.fill_between(xr, ci[0], ci[1], color=c,
                                    alpha=0.15, lw=0)

    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_pow10_label))
    for ax in (ax_ans, ax_int):
        ax.set_xlabel("training step")
    ax_dim.set_ylabel("eff. dim (centered)")
    ax_dim.set_title("(a) workspace dimension")
    # the caption defines drift as cos(J_l^T W_U, W_U); a mathtext label
    # would set its sub/superscripts below 9pt
    ax_drift.set_ylabel("drift (cosine)")
    ax_drift.set_title("(b) coordinate drift")
    for ax, title in ((ax_ans, "(c) answer rank"),
                      (ax_int, "(d) multi-hop intermediate rank")):
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_pow10_label))
        ax.set_title(title)
        ax.set_ylabel("median rank in readout")

    handles, labels = ax_dim.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               handlelength=2.0, columnspacing=1.6, borderaxespad=0.2)

    (OUT / "figures").mkdir(exist_ok=True)
    pu.save(fig, OUT, "b_development", top=0.94)


if __name__ == "__main__":
    main()
