"""Generate all paper figures from results/*.json. No UI screenshots.

Run:  .venv/bin/python experiments/figures/make_figures.py
Regenerating after any experiment rerun is safe: figures read only the
canonical JSON artifacts (plus bootstrap_ci.json for error bars).
"""

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import plot_utils as pu

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results"

MODEL_LABEL = {"qwen4b": "Qwen3-4B", "qwen17b": "Qwen3-1.7B"}


def load(name: str):
    return json.load(open(RES / name))


def fig_e6_dose() -> None:
    """Dose curves with bootstrap CIs; covert subset dissociation visible."""
    ci = load("bootstrap_ci.json")["models"]
    fig, axes = plt.subplots(1, 2, figsize=(pu.FULL_W, 2.4), sharey=True)
    for ax, mk in zip(axes, ("qwen4b", "qwen17b")):
        d = load(f"e6_covert_register_{mk}.json")
        alphas = [a for a in d["alphas"] if f"{a:g}" in d["summary"]["per_alpha"]]
        x = np.array(alphas, float)
        s = d["summary"]["per_alpha"]
        series = [
            ("flip_lang", "language flips", pu.BLUE, "o", "-",
             [s[f"{a:g}"]["register_flip_lang"] for a in alphas]),
            ("flip_full", "full flips (lang + answer)", pu.VERMILLION, "s", "-",
             [s[f"{a:g}"]["register_flip_full"] for a in alphas]),
            ("preserved", "answer preserved", pu.GREEN, "^", "--",
             [s[f"{a:g}"]["register_preserved"] for a in alphas]),
            ("covert", "full flips, covert subset", pu.PURPLE, "D", "-.",
             [s[f"{a:g}"]["covert_register_flip_full"] for a in alphas]),
            ("random", "random control (full flips)", pu.GREY, "x", ":",
             [s[f"{a:g}"]["random_flip_full"] for a in alphas]),
        ]
        for key, label, color, marker, ls, ys in series:
            ys = np.array(ys) * 100
            err = None
            if key in ("flip_full", "covert"):
                ck = "flip_full" if key == "flip_full" else "covert_subset_flip_full"
                blocks = [ci[mk]["e6"].get(f"alpha={a:g}", {}).get(ck) for a in alphas]
                if all(blocks):
                    lo = ys - np.array([b["lo"] for b in blocks]) * 100
                    hi = np.array([b["hi"] for b in blocks]) * 100 - ys
                    err = np.vstack([lo, hi])
            ax.errorbar(x, ys, yerr=err, label=label, color=color, marker=marker,
                        linestyle=ls, capsize=2, elinewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:g}" for a in alphas])
        n, ncov = d["summary"]["n_ok"], sum(
            1 for r in d["records"] if r["baseline_ok"] and r["covert"])
        ax.set_title(f"{MODEL_LABEL[mk]}  (n = {n}, covert n = {ncov})")
        ax.set_xlabel(r"dose $\alpha$ (fraction of measured gap)")
        ax.set_ylim(-4, 104)
    axes[0].set_ylabel("% of baseline-valid items")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.06))
    pu.save(fig, RES, "e6_dose_curves")
    plt.close(fig)


LADDER = ["qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"]
CONTROLS = ["pythia-70m", "gpt2-124m", "selftrained-124m"]
NICE = {"pythia-70m": "pythia\n70M", "gpt2-124m": "GPT-2\n124M",
        "selftrained-124m": "self-tr.\n124M", "qwen3-1.7b": "1.7B",
        "qwen3-4b": "4B", "qwen3-8b": "8B", "qwen3-14b": "14B",
        "qwen3-32b": "32B*", "qwen3.6-27b": "27B†"}


def fig_cone() -> None:
    """Panel A: raw vs transported eff-dim across the ladder (log y).
    Panel B: per-layer transported eff-dim profiles for 4B and 14B."""
    geo = {e["model"]: e for e in load("cone_geometry.json")}
    prof = load("cone_profile.json")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(pu.FULL_W, 2.5),
                                 gridspec_kw={"width_ratios": [1.45, 1]})

    order = CONTROLS + LADDER + ["qwen3.6-27b"]
    xs = np.arange(len(order))
    raw = [geo[m]["raw"]["eff_dim"] for m in order]
    jl = [geo[m]["jlens"]["eff_dim"] for m in order]
    ax.plot(xs[3:8], raw[3:8], color=pu.GREY, marker="o", linestyle="-", zorder=2)
    ax.plot(xs[3:8], jl[3:8], color=pu.VERMILLION, marker="s", linestyle="-", zorder=3)
    ax.scatter(xs[:3], raw[:3], color=pu.GREY, marker="o", zorder=2)
    ax.scatter(xs[:3], jl[:3], color=pu.VERMILLION, marker="s", zorder=3)
    ax.scatter(xs[8], raw[8], color=pu.GREY, marker="o", zorder=2)
    ax.scatter(xs[8], jl[8], facecolor=pu.VERMILLION, edgecolor=pu.BLACK,
               marker="*", s=90, linewidth=0.6, zorder=4)
    # under-fitted 32B: hollow marker
    ax.scatter(xs[7], jl[7], facecolor="white", edgecolor=pu.VERMILLION,
               marker="s", zorder=5)
    ax.plot([], [], color=pu.GREY, marker="o", linestyle="-", label=r"raw $W_U$ directions")
    ax.plot([], [], color=pu.VERMILLION, marker="s", linestyle="-",
            label=r"$J_\ell^\top W_U$ (transported)")
    for i, m in enumerate(order):
        dy = 8 if m == "selftrained-124m" else -11
        ax.annotate(f"{jl[i]:.1f}", (xs[i], jl[i]), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=6.5, color=pu.VERMILLION)
    ax.axvline(2.5, color=pu.BLACK, linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[m] for m in order], fontsize=6.5)
    ax.set_ylabel("effective dimensionality (PR)")
    ax.set_title("A. Transport collapses directional diversity")
    ax.legend(loc="upper left")

    for mk, color, ls in (("qwen3-4b", pu.VERMILLION, "-"), ("qwen3-14b", pu.BLUE, "--")):
        rows = sorted([e for e in prof if e["model"] == mk], key=lambda e: e["layer"])
        depth = np.array([e["layer"] for e in rows]) / rows[-1]["layer"]
        bx.plot(depth, [e["j_eff"] for e in rows], color=color, linestyle=ls,
                marker="o" if mk == "qwen3-4b" else "s", markersize=2.5,
                label=f"{NICE[mk]} transported")
        bx.axhline(geo[mk]["raw"]["eff_dim"], color=color, linewidth=0.7,
                   linestyle=":", alpha=0.7)
    bx.annotate("14B raw", (0.99, 121.8 * 1.12), fontsize=6.5, color=pu.BLUE,
                ha="right", va="bottom")
    bx.annotate("4B raw", (0.99, 74.9 * 0.88), fontsize=6.5, color=pu.VERMILLION,
                ha="right", va="top")
    bx.axvspan(10 / 34, 1.0, color=pu.GREY, alpha=0.12, linewidth=0)
    bx.annotate("4B intervention band", (0.31, 1.35), fontsize=6.5, color="0.35",
                ha="left")
    bx.set_yscale("log")
    bx.set_yticks([1, 2, 5, 10, 20, 50, 100])
    bx.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
    bx.set_xlabel("relative depth (layer / last layer)")
    bx.set_title("B. Collapse severity is per-model, not depth")
    # legend sits upper-left: lower right is occupied by the recovering 4B/14B
    # curves plus the intervention-band annotation (labels collided there)
    bx.legend(loc="upper left", framealpha=0.9)
    pu.save(fig, RES, "cone_geometry")
    plt.close(fig)


SETS = ["multilingual", "typo", "poetry", "order-ops", "association", "multihop"]
SET_LABEL = {"multilingual": "multi-\nlingual", "typo": "typo", "poetry": "poetry",
             "order-ops": "order-\nops", "association": "assoc.",
             "multihop": "multi-\nhop"}


def fig_e4b() -> None:
    """Covert-strict rates per set family: registers survive, plans die."""
    ci = load("bootstrap_ci.json")["models"]
    fig, ax = plt.subplots(figsize=(pu.COL_W, 2.3))
    width, xs = 0.38, np.arange(len(SETS))
    for off, mk, color in ((-width / 2, "qwen4b", pu.BLUE), (width / 2, "qwen17b", pu.SKY)):
        d = load(f"e4b_covert_{mk}.json")["sets"]
        vals = np.array([d[s]["stats"]["covert_strict"] for s in SETS]) * 100
        blocks = [ci[mk]["e4b"][s]["covert_strict"] for s in SETS]
        err = np.vstack([vals - np.array([b["lo"] for b in blocks]) * 100,
                         np.array([b["hi"] for b in blocks]) * 100 - vals])
        ax.bar(xs + off, vals, width, color=color, label=MODEL_LABEL[mk],
               yerr=err, capsize=2, error_kw={"elinewidth": 0.8})
        floors = np.array([d[s]["stats"]["ctrl_covert_strict"] for s in SETS]) * 100
        ax.plot(xs + off, floors, linestyle="none", marker="_", markersize=11,
                markeredgewidth=1.4, color=pu.BLACK, zorder=5)
    ax.plot([], [], linestyle="none", marker="_", color=pu.BLACK,
            label="permutation floor")
    ax.axvline(1.5, color=pu.BLACK, linewidth=0.6, linestyle=":", alpha=0.6)
    ax.text(0.75, 44.5, "context registers", ha="center", fontsize=7, style="italic")
    ax.text(3.6, 44.5, "content plans", ha="center", fontsize=7, style="italic")
    ax.set_xticks(xs)
    ax.set_xticklabels([SET_LABEL[s] for s in SETS], fontsize=7)
    ax.set_ylabel("covert-strict readouts (%)")
    ax.set_ylim(0, 48)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86))
    pu.save(fig, RES, "e4b_register_plan")
    plt.close(fig)


E7_LADDER = [("qwen17b", 1.7), ("qwen4b", 4.0), ("qwen8b", 8.0), ("qwen14b", 14.0)]


def fig_e7() -> None:
    """Scale ladder: behavioral capture is stable across 1.7B-14B while the
    self-report channel changes sign, specificity, and shape at every scale."""
    for mk, _ in E7_LADDER:
        d = load(f"e7_perspectival_{mk}.json")
        if "summary" not in d:
            print(f"  e7 {mk}: results not ready yet, skipping")
            return
    ci = load("bootstrap_ci.json")["models"]
    params = np.array([p for _, p in E7_LADDER])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(pu.FULL_W, 2.4))

    def series(ax_, arm, key, label, color, marker, scale=100.0, dx=1.0):
        vals, lo, hi = [], [], []
        for mk, _ in E7_LADDER:
            b = (ci[mk]["e7"][f"yes_margin_delta_{arm}_vs_none"] if key == "margin"
                 else ci[mk]["e7"][arm][key])
            vals.append(b["est"] * scale)
            lo.append(b["lo"] * scale)
            hi.append(b["hi"] * scale)
        vals = np.array(vals)
        err = np.vstack([vals - np.array(lo), np.array(hi) - vals])
        ax_.errorbar(params * dx, vals, yerr=err, label=label, color=color,
                     marker=marker, markersize=4, linewidth=1.0,
                     capsize=2, elinewidth=0.8)

    # panel A: capture, stable everywhere
    series(ax, "full", "flip", "answer flips (full)", pu.VERMILLION, "o")
    series(ax, "full", "restate_swapped", "question restated as edited (full)", pu.BLUE, "s")
    series(ax, "half2", "restate_swapped", "restated as edited (late half-band)", pu.SKY, "D")
    series(ax, "randdir", "flip", "random-direction control", pu.GREY, "x")
    ax.set_xscale("log")
    ax.set_xticks(params)
    ax.set_xticklabels(["1.7B", "4B", "8B", "14B"])
    ax.minorticks_off()
    ax.set_ylim(-4, 104)
    ax.set_ylabel("% of items")
    ax.set_xlabel("model scale")
    ax.set_title("Capture is scale-stable")
    ax.legend(loc="center left", fontsize=6, framealpha=0.9)

    # panel B: the self-report margin, unstable in sign and specificity
    bx.axhline(0.0, color=pu.BLACK, linewidth=0.6)
    series(bx, "full", "margin", "entity swap (full)", pu.VERMILLION, "o", scale=1.0)
    series(bx, "randdir", "margin", "random-direction control", pu.GREY, "x", scale=1.0)
    bx.set_xscale("log")
    bx.set_xticks(params)
    bx.set_xticklabels(["1.7B", "4B", "8B", "14B"])
    bx.minorticks_off()
    bx.set_ylabel(r"$\Delta$ yes log-odds vs clean")
    bx.set_xlabel("model scale")
    bx.set_title("Self-report is not")
    bx.legend(loc="upper right", fontsize=6, framealpha=0.9)
    pu.save(fig, RES, "e7_capture")
    plt.close(fig)


if __name__ == "__main__":
    pu.setup()
    print("generating figures into results/figures/")
    fig_e6_dose()
    fig_cone()
    fig_e4b()
    fig_e7()
