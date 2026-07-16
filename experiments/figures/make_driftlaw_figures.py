"""Figures for the drift law and the writability gate.

Reads ONLY results/*.json, including the merged per-layer table written by
experiments/e7-perspectival-capture/analyze_e7_driftlaw.py, which must be run
first. No numbers are hardcoded. Style follows plot_utils.py (Okabe-Ito).

Run:
    .venv/bin/python experiments/e7-perspectival-capture/analyze_e7_driftlaw.py
    .venv/bin/python experiments/figures/make_driftlaw_figures.py [--out DIR]

Outputs (PDF + PNG preview), by default into results/figures/:
    fig_capture_vs_depth   single-layer restate capture vs fractional depth
    fig_capture_vs_drift   (a) per-layer capture vs drift, all profiled models;
                           (b) Gemma-2-9B cross-domain replication
    fig_drift_ordering     drift at matched frac depth 0.8 vs late-half capture
    fig_amplitude          per-layer relative edit amplitude vs fractional depth
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
A3 = RESULTS / "e7_driftlaw_table.json"
OUT = RESULTS / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_utils  # noqa: E402
from plot_utils import (  # noqa: E402
    BLACK, BLUE, SKY, VERMILLION, ORANGE, GREEN, PURPLE, GREY, FULL_W, COL_W,
)

# tag -> (display name, colour, marker, linestyle, tier). Tier is the
# three-level generation split the pooled regression uses: the writability
# gate is graded gemma > hybrid Qwen > dense Qwen, so the profiled models are
# panelled by tier, not by vendor.
MODELS = {
    "gemma2-2b": ("Gemma-2-2B", BLUE, "o", "-", "gemma"),
    "gemma2-9b": ("Gemma-2-9B", SKY, "s", "--", "gemma"),
    "gemma2-27b": ("Gemma-2-27B", BLACK, "p", ":", "gemma"),
    "qwen35-9b": ("Qwen3.5-9B", ORANGE, "X", "-", "hybrid"),
    "qwen36-27b": ("Qwen3.6-27B", GREEN, "*", "--", "hybrid"),
    "qwen17b": ("Qwen3-1.7B", VERMILLION, "^", "-", "dense"),
    "qwen4b": ("Qwen3-4B", ORANGE, "D", "--", "dense"),
    "qwen8b": ("Qwen3-8B", GREEN, "v", "-.", "dense"),
    "qwen14b": ("Qwen3-14B", PURPLE, "P", ":", "dense"),
}

TIER_TITLES = {
    "gemma": "Gemma-2",
    "hybrid": "Qwen3.5/3.6 (hybrid)",
    "dense": "Qwen3 (dense)",
}

# Models with half-band arms, drift, and amplitude but no single-layer sweep
# yet; they appear only in the matched-depth ordering figure. Promote to
# MODELS (with a tier) once their profile JSON lands. Empty as of the
# 2026-07-16 qwen36-27b landing: all nine profiled models are in MODELS.
ORDERING_EXTRA = {}

# cross-domain arms of Gemma-2-9B: domain -> (file infix, display, colour, marker)
DOMAINS = {
    "capitals": ("", "Capitals", SKY, "s"),
    "states": ("states_", "US states", VERMILLION, "^"),
    "currency": ("currency_", "Currencies", GREEN, "D"),
}


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_profile(tag: str, infix: str = "", key: str = "flip"):
    """Per-layer single-layer profile: (frac_depths, rate%, layers).

    ``key`` is "flip" (answer flip; floor-free, the default per the
    2026-07-15 metric audit) or "restate_swapped" (capture; carries the
    incidental-mention floor).
    """
    path = RESULTS / f"e7_profile_{infix}{tag}.json"
    rows = load_json(path)["summary"]["profile"]
    fracs = [r["frac_depth"] for r in rows]
    caps = [100.0 * r[key] for r in rows]
    layers = [r["layer"] for r in rows]
    return fracs, caps, layers


def load_drift(tag: str, infix: str = ""):
    """Per-layer coordinate drift: dict layer -> (frac_depth, drift_entity)."""
    rows = load_json(RESULTS / f"e7_drift_{infix}{tag}.json")["profile"]
    return {r["layer"]: (r["frac_depth"], r["drift_entity"]) for r in rows}


def load_amplitude(tag: str, infix: str = ""):
    """Per-layer edit amplitude: (frac_depths, rel_edit_mean, edit_norm_mean)."""
    rows = load_json(RESULTS / f"e7_amplitude_{infix}{tag}.json")["profile"]
    fracs = [r["frac_depth"] for r in rows]
    rel = [r["rel_edit_mean"] for r in rows]
    norm = [r["edit_norm_mean"] for r in rows]
    return fracs, rel, norm


def drift_capture_pairs(tag: str, infix: str = "", key: str = "flip"):
    """Join single-layer profile and drift profile on layer index."""
    drift = load_drift(tag, infix)
    _, caps, layers = load_profile(tag, infix, key)
    x, y = [], []
    for layer, cap in zip(layers, caps):
        if layer in drift:
            x.append(drift[layer][1])
            y.append(cap)
    return x, y


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    print(f"  wrote {OUT / name}.pdf .png")


# ---------------------------------------------------------------- figure 1
def fig_capture_vs_depth():
    """Three tier panels of single-layer FLIP vs depth.

    Flip is the floor-free gate signal (2026-07-15 audit): open models
    (Gemma-2B/9B, hybrid 9B) peak level at 23-25%, closed models (all dense
    Qwen3 and Gemma-2-27B) never leave zero.
    """
    tiers = ("gemma", "hybrid", "dense")
    fig, axes = plt.subplots(
        1, 3, figsize=(FULL_W, 2.5), sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    panels = dict(zip(tiers, axes))

    for tag, (name, color, marker, ls, tier) in MODELS.items():
        fracs, flips, _ = load_profile(tag)
        panels[tier].plot(
            fracs, flips, marker=marker, linestyle=ls, color=color,
            label=name, markersize=3.5, linewidth=1.2,
        )

    for tier, ax in panels.items():
        ax.set_title(TIER_TITLES[tier])
        ax.set_xlabel("Fractional depth of edited layer")
        ax.set_xlim(0.2, 1.0)
        # 1.0 only on the last panel: adjacent "1.0"/"0.2" labels collide.
        ax.set_xticks([0.2, 0.4, 0.6, 0.8] + ([1.0] if ax is axes[-1] else []))
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Single-layer answer flip (%)")
    axes[0].set_ylim(bottom=-1.5)
    save(fig, "fig_capture_vs_depth")
    plt.close(fig)


DRIFT_LABEL = ("Coordinate drift of edited layer,"
               " $\\cos(J_\\ell^\\top W_U, W_U)$")


# ---------------------------------------------------------------- figure 2
def fig_capture_vs_drift():
    """(a) all-model capture-vs-drift scatter; (b) Gemma-2-9B cross-domain."""
    a3 = load_json(A3)
    table = a3["table"]
    stats = a3["stats"]["per_model"]

    fig, axes = plt.subplots(
        1, 2, figsize=(FULL_W, 2.9), gridspec_kw={"wspace": 0.24},
    )
    ax, axd = axes

    # ---- panel (a): all profiled models, capitals domain (a3 table) --------
    # Flip is the floor-free metric; open models (any positive flip layer
    # beyond a single item) are coloured with their Spearman, closed models
    # (all dense Qwen3 and Gemma-2-27B) are the grey zero floor.
    closed = [tag for tag in MODELS
              if stats[tag]["flip_layers_pos"] == 0
              or stats[tag]["peak_flip"] * stats[tag]["none_arm"]["n_items"] <= 1.5]
    for tag, (name, color, marker, _, tier) in MODELS.items():
        rows = [r for r in table if r["model"] == tag]
        x = [r["drift"] for r in rows]
        y = [100.0 * r["flip"] for r in rows]
        if tag not in closed:
            s = stats[tag]
            ax.scatter(
                x, y, s=16, color=color, marker=marker, zorder=3,
                label=f"{name} ($\\rho$ = {s['spearman_flip']:.2f},"
                      f" n = {s['n']})",
            )
        else:
            ax.scatter(x, y, s=9, color=GREY, marker=marker, label=name,
                       alpha=0.75, zorder=2)

    closed_max = max(100.0 * r["flip"] for r in table
                     if r["model"] in closed)
    ax.annotate("Closed gate (dense Qwen3 +\nGemma-2-27B): all layers at\n"
                f"0–{closed_max:.1f}% (single items)",
                xy=(0.56, 14.2), fontsize=6.5, color=GREY, linespacing=1.4)

    ax.set_title(f"(a) {len(MODELS)} models, capitals")
    ax.set_xlabel(DRIFT_LABEL)
    ax.set_ylabel("Single-layer answer flip (%)")
    ax.legend(loc="upper right", fontsize=6.5, labelspacing=0.35,
              handletextpad=0.2, borderaxespad=0.3)

    # ---- panel (b): Gemma-2-9B across three swap domains -------------------
    cross = stats["gemma2-9b"]["cross_domain"]
    per_dom = cross["per_domain"]
    for domain, (infix, disp, color, marker) in DOMAINS.items():
        x, y = drift_capture_pairs("gemma2-9b", infix)
        s = per_dom[domain]["flip"]
        axd.scatter(
            x, y, s=16, color=color, marker=marker, zorder=3,
            label=f"{disp} ($\\rho$ = {s['spearman']:.2f},"
                  f" {s['n_items']} items)",
        )

    pooled = cross["pooled_flip"]
    axd.annotate(
        f"Pooled Spearman $\\rho$ = {pooled['spearman']:.2f}\n"
        f"95% CI [{pooled['lo']:.2f}, {pooled['hi']:.2f}],"
        f" {pooled['n_layer_obs']} layer obs.",
        xy=(0.36, 21.5), fontsize=6.5, color="black",
    )

    axd.set_title("(b) Gemma-2-9B, three domains")
    axd.set_xlabel(DRIFT_LABEL)
    axd.set_ylabel("Single-layer answer flip (%)")
    axd.legend(loc="upper right", fontsize=6.5, labelspacing=0.35,
               handletextpad=0.2, borderaxespad=0.3)

    for a in axes:
        a.set_ylim(-1.5, 40.0)
        a.set_xlim(0.05, 1.0)

    save(fig, "fig_capture_vs_drift")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_drift_ordering():
    """Drift at matched fractional depth 0.8 vs late-half-band capture."""
    boot = load_json(RESULTS / "bootstrap_ci.json")["models"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    for tag, (name, color, marker, _, _) in {**MODELS, **ORDERING_EXTRA}.items():
        # drift at the band layer nearest fractional depth 0.8
        drift = load_drift(tag)
        frac, d08 = min(drift.values(), key=lambda fd: abs(fd[0] - 0.8))

        # late-half-band capture (half2 restate) with bootstrap 95% CI
        arms = load_json(
            RESULTS / f"e7_perspectival_{tag}.json")["summary"]["arms"]
        cap = 100.0 * arms["half2"]["restate_swapped"]
        ci = boot[tag]["e7"]["half2"]["restate_swapped"]
        lo, hi = 100.0 * ci["lo"], 100.0 * ci["hi"]

        ax.errorbar(
            d08, cap, yerr=[[cap - lo], [hi - cap]],
            fmt=marker, color=color, capsize=2, elinewidth=0.8,
            markersize=5, label=name,
        )
        dx, dy, ha = 0.012, 0.0, "left"
        if tag == "qwen4b":
            dy = 3.0  # lift the label off the errorbar cap
        if tag == "qwen8b":
            dx, ha = -0.012, "right"  # 8B sits left of the 14B/1.7B pair
        ax.annotate(name, xy=(d08 + dx, cap + dy), fontsize=6.5,
                    color=color, ha=ha, va="center")

    ax.set_xlim(right=0.95)  # room for the right-side labels
    ax.set_xlabel("Coordinate drift at matched fractional depth 0.8")
    ax.set_ylabel("Late-half-band restate capture (%)")
    ax.set_ylim(bottom=-4)
    save(fig, "fig_drift_ordering")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
# The amplitude comparison is a two-sided family argument (Gemma vs Qwen), so
# both Qwen tiers share the right panel; the tier split stays visible through
# each model's own colour/marker.
AMP_GROUPS = {
    "gemma": ("gemma",),
    "qwen": ("hybrid", "dense"),
}


def family_envelope(group: str, grid):
    """Min/max relative edit magnitude of a group over the depths it covers.

    Each model is linearly interpolated onto ``grid``; grid points outside a
    model's band are NaN and drop out, and points covered by no model of the
    group are removed entirely (returned as a mask).
    """
    curves = []
    for tag, (_, _, _, _, tier) in MODELS.items():
        if tier not in AMP_GROUPS[group]:
            continue
        fracs, rel, _ = load_amplitude(tag)
        curves.append(np.interp(grid, fracs, rel, left=np.nan, right=np.nan))
    stack = np.vstack(curves)
    covered = ~np.isnan(stack).all(axis=0)
    lo = np.nanmin(stack[:, covered], axis=0)
    hi = np.nanmax(stack[:, covered], axis=0)
    return grid[covered], lo, hi


def fig_amplitude():
    """Relative edit magnitude per layer: Qwen's band is not amplitude-starved.

    Each panel plots one family and shades the other family's min/max envelope,
    so the overlap (or lack of it) is readable without cross-panel eye traffic.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(FULL_W, 2.7), sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    panels = {"gemma": axes[0], "qwen": axes[1]}
    titles = {"gemma": "Gemma-2 family", "qwen": "Qwen3 / 3.5 family"}
    others = {"gemma": "qwen", "qwen": "gemma"}
    other_names = {"gemma": "Qwen", "qwen": "Gemma-2"}

    grid = np.linspace(0.20, 1.0, 161)
    for group, ax in panels.items():
        gx, lo, hi = family_envelope(others[group], grid)
        ax.fill_between(
            gx, 100.0 * lo, 100.0 * hi, color=GREY, alpha=0.22,
            linewidth=0, zorder=1,
            label=f"{other_names[group]} envelope",
        )

    for tag, (name, color, marker, ls, tier) in MODELS.items():
        fracs, rel, _ = load_amplitude(tag)
        group = "gemma" if tier == "gemma" else "qwen"
        panels[group].plot(
            fracs, [100.0 * v for v in rel], marker=marker, linestyle=ls,
            color=color, label=name, markersize=3.5, linewidth=1.2, zorder=3,
        )

    for group, ax in panels.items():
        ax.set_title(titles[group])
        ax.set_xlabel("Fractional depth of edited layer")
        ax.set_xlim(0.2, 1.0)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left", ncol=1, handletextpad=0.4,
                  borderaxespad=0.3)
    axes[0].set_ylabel(r"Relative edit magnitude"
                       r" $\|\Delta h_\ell\| / \|h_\ell\|$ (%)")
    save(fig, "fig_amplitude")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
# Sliding fixed-width (k=6) window zone maps. Skipped silently for models
# whose e7_slide6_*.json has not landed yet.
ZONE_MODELS = ("gemma2-2b", "gemma2-9b", "qwen17b", "qwen4b", "gemma2-27b",
               "qwen8b", "qwen14b", "qwen36-27b")


def load_zone(tag: str, infix: str = ""):
    """Slide-mode windows: (centre frac depths, flip%, full-band flip%)."""
    path = RESULTS / f"e7_slide6_{infix}{tag}.json"
    if not path.exists():
        return None
    s = load_json(path)["summary"]
    n_layers = s["n_layers"]
    xs, ys = [], []
    full = None
    for w in s["windows"]:
        if w["arm"] == "full":
            full = 100.0 * w["flip"]
        elif w["arm"].startswith("s"):
            xs.append((w["layers"][0] + w["layers"][1]) / 2 / n_layers)
            ys.append(100.0 * w["flip"])
    return xs, ys, full


def fig_writable_zone():
    """Zone maps: answer flip of a width-6 window vs window-centre depth.

    One fixed-width instrument, three shapes (2026-07-15 log entries):
    ceiling plateau (Gemma-2-2B), monotone decline (Qwen3-1.7B), interior
    peak (Qwen3-4B). The hollow-marker curve repeats Qwen3-4B on the
    currency domain: the onset and plateau do not move with the task;
    the cutoff does (2026-07-16 three-domain triangulation).
    """
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    plotted = False
    for tag in ZONE_MODELS:
        zone = load_zone(tag)
        if zone is None:
            continue
        plotted = True
        name, color, marker, ls, _ = {**MODELS, **ORDERING_EXTRA}[tag]
        xs, ys, full = zone
        ax.plot(xs, ys, marker=marker, linestyle=ls, color=color,
                markersize=3.5, linewidth=1.2,
                label=f"{name} (full band {full:.0f}%)")
    cur = load_zone("qwen4b", "currency_")
    if cur is not None:
        xs, ys, full = cur
        _, color, marker, _, _ = MODELS["qwen4b"]
        ax.plot(xs, ys, marker=marker, linestyle=":", color=color,
                markersize=3.5, linewidth=1.0, markerfacecolor="none",
                label=f"Qwen3-4B, currency (full {full:.0f}%)")
    if not plotted:
        print("  fig_writable_zone: no e7_slide6_*.json found, skipped")
        plt.close(fig)
        return

    ax.set_xlabel("Fractional depth of window centre (width 6)")
    ax.set_ylabel("Answer flip (%)")
    ax.set_ylim(-3.0, 104.0)
    # the Gemma plateau owns the top of the axes, so the legend sits above
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              fontsize=6.5, labelspacing=0.35, handletextpad=0.4,
              columnspacing=0.8, frameon=False)
    save(fig, "fig_writable_zone")
    plt.close(fig)


# ---------------------------------------------------------------- figure 6
def zone_cutoff_frac(tag: str):
    """Window-centre fractional depth where zone-map flip crosses 25% of peak."""
    zone = load_zone(tag)
    if zone is None:
        return None
    xs, ys, _ = zone
    peak = max(ys)
    thr = 0.25 * peak
    ipk = ys.index(peak)
    for i in range(ipk, len(ys) - 1):
        if ys[i] >= thr > ys[i + 1]:
            return xs[i] + (xs[i + 1] - xs[i]) * (ys[i] - thr) / (ys[i] - ys[i + 1])
    return None


def fig_deadline():
    """Zone-map cutoff vs consumption deadline, one point per model.

    The deadline (mean commit layer of the answer token under a plain logit
    lens, capitals, no interventions) sits 0.05-0.19 of depth after the
    zone-map cutoff in every model and preserves its ordering (Spearman
    0.857, n=7): the deadline model is ordinal, not metric.
    """
    xs, ys, tags = [], [], []
    for tag in ZONE_MODELS:
        cut = zone_cutoff_frac(tag)
        dl_path = RESULTS / f"e7_deadline_{tag}.json"
        if cut is None or not dl_path.exists():
            continue
        dl = load_json(dl_path)["domains"]["capitals"]
        if not dl["n_used"]:
            continue
        xs.append(cut)
        ys.append(dl["mean_commit_frac"])
        tags.append(tag)
    if len(xs) < 3:
        print("  fig_deadline: not enough models with both measures, skipped")
        return

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(order):
            r[i] = k + 1
        return r
    rc, rd = ranks(xs), ranks(ys)
    n = len(xs)
    rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(rc, rd)) / (n * (n * n - 1))

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    lo, hi = 0.45, 1.0
    ax.plot([lo, hi], [lo, hi], color=GREY, linewidth=0.8, linestyle="--",
            zorder=1, label="cutoff = deadline")
    for x, y, tag in zip(xs, ys, tags):
        name, color, marker, _, _ = {**MODELS, **ORDERING_EXTRA}[tag]
        ax.scatter([x], [y], color=color, marker=marker, s=30, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, -8),
                    fontsize=5.5, color=color)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Zone-map cutoff (fractional depth, 25% of peak flip)")
    ax.set_ylabel("Consumption deadline (fractional depth)")
    ax.annotate(f"Spearman $\\rho$ = {rho:.2f} (n = {n})\ndeadline later everywhere",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7)
    ax.legend(loc="lower right", fontsize=6.5, frameon=False)
    save(fig, "fig_deadline")
    plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT,
                    help="directory for the PDF/PNG figures (default results/figures)")
    OUT = ap.parse_args().out

    plot_utils.setup()
    fig_capture_vs_depth()
    fig_capture_vs_drift()
    fig_drift_ordering()
    fig_amplitude()
    fig_writable_zone()
    fig_deadline()


if __name__ == "__main__":
    main()
