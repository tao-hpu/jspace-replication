"""Trajectory-level checks for paper B added in the November 2026 preprint revision.

Reads only existing records (results/b3_curves_*.json, results/b3_probe_rank_*.json); no refits.

1. Excess path length of the late log10(median intermediate rank) trajectory
   (steps 13000..143000, same statistic as run_b3_consolidation_test.py) for every trajectory
   that covers the late window: Pythia 410M, 1B, 1.4B-deduped, 1.4B non-deduplicated, OLMo-1B.
   Paired bootstrap (2000 resamples, seed 0, same item indices for every model and checkpoint)
   for: excess path 1B minus non-dedup, non-dedup minus dedup; endpoint log10 median
   1B minus non-dedup, non-dedup minus dedup.
2. Lens-specific compression: ratio of centered raw to centered transported effective dimension
   at the mid layer, per step and model, plus the cone mean cosine and uncentered transported
   effective dimension (anisotropy diagnostics).
3. Onset sensitivity: first step at which each stage registers under alternative criteria,
   for the five trajectories. Collapse: transported centered eff-dim < 0.5 x step-0 value, and
   step of the largest one-interval drop. Answer: top-10 rate >= 20%, median rank < 100.
   Intermediate: (a) top-10 rate first non-zero, (b) median rank < 1000,
   (c) median rank < 500 and < foil/2 (the criterion of make_olmo_table.py).

Run:  .venv/bin/python experiments/e11-development/run_b3_trajectory_checks.py
Out:  results/b3_trajectory_checks.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LATE = [13000, 23000, 43000, 73000, 103000, 143000]
N_BOOT, SEED = 2000, 0
EXCLUDE = {"pythia-1.4b-deduped": {16}}  # branch carries the step-143000 weights (Appendix A)

# Trajectory name -> (curves tag, probe-rank file tag). The _foil files hold the same real
# intermediate ranks as the plain files plus the foil; the non-dedup run exists only as _foil.
TRAJ = {
    "pythia-410m-deduped": ("pythia-410m-deduped", "pythia-410m-deduped_foil"),
    "pythia-1b-deduped": ("pythia-1b-deduped", "pythia-1b-deduped_foil"),
    "pythia-1.4b-deduped": ("pythia-1.4b-deduped", "pythia-1.4b-deduped_foil"),
    "pythia-1.4b": ("pythia-1.4b", "pythia-1.4b_foil"),
    "OLMo-1B-0724-hf": ("OLMo-1B-0724-hf", "OLMo-1B-0724-hf_foil"),
}


def rows(path: Path, model: str) -> dict:
    data = json.loads(path.read_text())["rows"]
    return {r["step"]: r for r in data if r["step"] not in EXCLUDE.get(model, set())}


def late_ranks(probe: dict) -> np.ndarray:
    return np.array([probe[s]["intermediate"]["ranks"] for s in LATE], dtype=float)


def excess_and_end(ranks: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    lm = np.log10(np.median(ranks[:, idx], axis=1))
    return float(np.sum(np.abs(np.diff(lm))) - abs(lm[-1] - lm[0])), float(lm[-1])


def first(steps, cond):
    for s in steps:
        if cond(s):
            return s
    return None


def main() -> None:
    curves, probes = {}, {}
    for name, (ct, pt) in TRAJ.items():
        curves[name] = rows(ROOT / f"results/b3_curves_{ct}.json", name)
        probes[name] = rows(ROOT / f"results/b3_probe_rank_{pt}.json", name)

    # The _foil files were produced by a later run; a few single-item ranks differ by 1 from the
    # plain files (numerical ties), medians are unchanged. Where a plain full-trajectory file
    # exists, take the real intermediate ranks from it so that the numbers match
    # run_b3_consolidation_test.py exactly; foil ranks always come from the _foil file.
    max_rank_gap = {}
    for m in ["pythia-410m-deduped", "pythia-1b-deduped", "pythia-1.4b-deduped"]:
        plain = rows(ROOT / f"results/b3_probe_rank_{m}.json", m)
        gap = 0
        for s in probes[m]:
            a = np.array(plain[s]["intermediate"]["ranks"]); b = np.array(probes[m][s]["intermediate"]["ranks"])
            gap = max(gap, int(np.abs(a - b).max()))
            probes[m][s]["intermediate"] = {**plain[s]["intermediate"]}
            probes[m][s]["answer"] = plain[s]["answer"]
        max_rank_gap[m] = gap

    out: dict = {"late_steps": LATE, "n_boot": N_BOOT, "seed": SEED,
                 "max_rank_gap_plain_vs_foil_file": max_rank_gap}

    # 1. excess path length and paired bootstrap
    R = {m: late_ranks(probes[m]) for m in TRAJ}
    n = R["pythia-1b-deduped"].shape[1]
    full = np.arange(n)
    obs = {m: excess_and_end(R[m], full) for m in TRAJ}
    out["excess_path_length"] = {m: obs[m][0] for m in TRAJ}
    out["late_log10_median_trajectory"] = {
        m: [float(x) for x in np.log10(np.median(R[m], axis=1))] for m in TRAJ}
    pairs = [("pythia-1b-deduped", "pythia-1.4b"), ("pythia-1.4b", "pythia-1.4b-deduped"),
             ("pythia-1b-deduped", "pythia-1.4b-deduped")]
    rng = np.random.default_rng(SEED)
    boot = {f"{a}-{b}": {"excess": [], "end": []} for a, b in pairs}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        st = {m: excess_and_end(R[m], idx) for m in {x for p in pairs for x in p}}
        for a, b in pairs:
            boot[f"{a}-{b}"]["excess"].append(st[a][0] - st[b][0])
            boot[f"{a}-{b}"]["end"].append(st[a][1] - st[b][1])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out["paired_bootstrap"] = {}
    for a, b in pairs:
        k = f"{a}-{b}"
        out["paired_bootstrap"][k] = {
            "excess_path_diff_obs": obs[a][0] - obs[b][0],
            "excess_path_diff_ci95": ci(boot[k]["excess"]),
            "endpoint_log10_median_diff_obs": obs[a][1] - obs[b][1],
            "endpoint_log10_median_diff_ci95": ci(boot[k]["end"]),
        }

    # 2. lens-specific compression ratio and anisotropy diagnostics
    out["ratio_raw_over_jl"] = {}
    for m in TRAJ:
        c = curves[m]
        out["ratio_raw_over_jl"][m] = {
            s: {"jl_c": c[s]["eff_dim_jl_centered"], "raw_c": c[s]["eff_dim_raw_centered"],
                "ratio": c[s]["eff_dim_raw_centered"] / c[s]["eff_dim_jl_centered"],
                "jl_uncentered": c[s]["eff_dim_jl"], "cone_cos_jl": c[s]["cone_cos_jl"]}
            for s in sorted(c)}

    # 3. onset sensitivity
    onset = {}
    for m in TRAJ:
        c, p = curves[m], probes[m]
        cs, ps = sorted(c), sorted(p)
        e0 = c[0]["eff_dim_jl_centered"] if 0 in c else None
        drops = [(c[a]["eff_dim_jl_centered"] - c[b]["eff_dim_jl_centered"], b)
                 for a, b in zip(cs, cs[1:])]
        foil = lambda s: p[s].get("intermediate_foil", {}).get("median_rank", np.inf)
        onset[m] = {
            "first_step_available": ps[0],
            "collapse_below_half_step0": first(cs, lambda s: c[s]["eff_dim_jl_centered"] < 0.5 * e0) if e0 else None,
            "collapse_largest_interval_drop_ends_at": max(drops)[1] if drops else None,
            "answer_top10_ge_20pct": first(ps, lambda s: p[s]["answer"]["top10_rate"] >= 0.2),
            "answer_median_lt_100": first(ps, lambda s: p[s]["answer"]["median_rank"] < 100),
            "inter_top10_first_nonzero": first(ps, lambda s: p[s]["intermediate"]["top10_rate"] > 0),
            "inter_top10_first_nonzero_after_step0": first([s for s in ps if s > 0], lambda s: p[s]["intermediate"]["top10_rate"] > 0),
            "inter_median_lt_1000": first(ps, lambda s: p[s]["intermediate"]["median_rank"] < 1000),
            "inter_median_lt_500_and_lt_half_foil": first(ps, lambda s: p[s]["intermediate"]["median_rank"] < 500 and p[s]["intermediate"]["median_rank"] < foil(s) / 2),
        }
    out["onset"] = onset

    # 4. (round 2) Leave-one-checkpoint-out: drop each late checkpoint in turn, recompute the
    #    excess path length of every trajectory and the paired bootstrap of the differences
    #    (same seed and same resampled item indices as section 1). Tests whether the 1B
    #    excess path rests on a single checkpoint.
    loo = {}
    for drop in LATE:
        keep = [i for i, s in enumerate(LATE) if s != drop]
        Rk = {m: R[m][keep] for m in TRAJ}
        obs_k = {m: excess_and_end(Rk[m], full)[0] for m in TRAJ}
        rng_k = np.random.default_rng(SEED)
        bk = {f"{a}-{b}": [] for a, b in pairs}
        for _ in range(N_BOOT):
            idx = rng_k.integers(0, n, size=n)
            st = {m: excess_and_end(Rk[m], idx)[0] for m in {x for p in pairs for x in p}}
            for a, b in pairs:
                bk[f"{a}-{b}"].append(st[a] - st[b])
        loo[drop] = {"excess_path_length": obs_k,
                     "paired": {f"{a}-{b}": {"diff_obs": obs_k[a] - obs_k[b],
                                             "ci95": ci(bk[f"{a}-{b}"])} for a, b in pairs}}
    out["leave_one_checkpoint_out"] = loo

    # 5. (round 2) Matched median-rank thresholds: first step at which the answer median and
    #    the intermediate median fall below the same threshold T, plus the asymmetric pairing
    #    used in Appendix E (answer < 100, intermediate < 1000).
    matched = {}
    for m in TRAJ:
        p = probes[m]; ps = sorted(p)
        a_on = lambda T: first(ps, lambda s: p[s]["answer"]["median_rank"] < T)
        i_on = lambda T: first(ps, lambda s: p[s]["intermediate"]["median_rank"] < T)
        matched[m] = {str(T): {"answer": a_on(T), "intermediate": i_on(T)} for T in (1000, 500, 200, 100)}
        matched[m]["answer_lt_100_vs_inter_lt_1000"] = {"answer": a_on(100), "intermediate": i_on(1000)}
    out["matched_threshold_onsets"] = matched

    # 6. (round 2) Which multi-hop items (indices into the 29) reach the intermediate top 10
    #    at each checkpoint: shows whether a top-10 onset is a single non-recurring item.
    out["intermediate_top10_items"] = {
        m: {s: [int(i) for i in np.where(np.array(probes[m][s]["intermediate"]["ranks"]) < 10)[0]]
            for s in sorted(probes[m])} for m in TRAJ}

    # 7. (round 2) First step at which the raw/transported ratio reaches 1.5 (from step 2000 on,
    #    i.e. after the step-1000 transient).
    out["ratio_first_ge_1p5_after_2000"] = {
        m: first(sorted(s for s in curves[m] if s >= 2000),
                 lambda s: out["ratio_raw_over_jl"][m][s]["ratio"] >= 1.5) for m in TRAJ}

    dest = ROOT / "results/b3_trajectory_checks.json"
    dest.write_text(json.dumps(out, indent=2))
    print("excess path:", {m: round(v, 2) for m, v in out["excess_path_length"].items()})
    for k, v in out["paired_bootstrap"].items():
        print(k, {kk: (round(x, 2) if isinstance(x, float) else [round(y, 2) for y in x]) for kk, x in v.items()})
    for m in TRAJ:
        print(m, {s: round(v["ratio"], 2) for s, v in out["ratio_raw_over_jl"][m].items()})
    for m, v in onset.items():
        print(m, v)
    print("written:", dest)


if __name__ == "__main__":
    main()
