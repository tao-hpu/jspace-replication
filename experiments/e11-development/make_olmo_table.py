"""Side-by-side developmental table: OLMo-1B-0724 vs Pythia-1B-deduped (same 16 layers,
d=2048, ~2M tokens/step, so steps align). Reads the B3 result JSONs and prints Markdown for
the cross-family comparison plus a one-line ordering verdict per family.
  .venv/bin/python experiments/e11-development/make_olmo_table.py [--olmo OLMo-1B-0724-hf] [--pythia pythia-1b-deduped]
"""
from __future__ import annotations
import argparse, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]

def load(tag):
    c = {r["step"]: r for r in json.load(open(ROOT / f"results/b3_curves_{tag}.json"))["rows"]}
    pf = ROOT / f"results/b3_probe_rank_{tag}_foil.json"
    p = {r["step"]: r for r in json.load(open(pf))["rows"]} if pf.exists() else {}
    return c, p

def row(c, p, s):
    cr, pr = c.get(s), p.get(s)
    ed = f"{cr['eff_dim_jl_centered']:.1f}" if cr else "-"
    dr = f"{cr['mean_drift']:+.3f}" if cr else "-"
    if pr:
        a = pr["answer"]; i = pr["intermediate"]; f = pr.get("intermediate_foil")
        an = f"{a['median_rank']:.0f} / {100*a['top10_rate']:.0f}%"
        inr = f"{i['median_rank']:.0f} / {100*i['top10_rate']:.0f}%"
        fo = f"{f['median_rank']:.0f}" if f else "-"
    else:
        an = inr = fo = "-"
    return ed, dr, an, inr, fo

def onset(c, p):
    """First step at which each stage registers. The paper's primary criteria (Section 3.2):
    collapse = checkpoint interval with the largest drop of the transported centered eff-dim
    (reported as the step that ends it); answer = answer top-10 rate >= 20%; intermediate =
    first checkpoint after step 0 with a non-zero intermediate top-10 rate. Alternatives,
    reported in the paper's onset-sensitivity table: eff-dim below half its step-0 value;
    answer median rank < 100; intermediate median rank < 1000; intermediate median rank < 500
    and below foil/2. Returns dict of steps (None if never). See also
    run_b3_trajectory_checks.py, which computes the same table for all five trajectories."""
    steps = sorted(c)
    e0 = c[steps[0]]["eff_dim_jl_centered"] if steps and steps[0] == 0 else None
    def first(cond, keys):
        for s in sorted(keys):
            if cond(s): return s
        return None
    drops = [(c[a]["eff_dim_jl_centered"] - c[b]["eff_dim_jl_centered"], b) for a, b in zip(steps, steps[1:])]
    foil = lambda s: p[s]["intermediate_foil"]["median_rank"] if "intermediate_foil" in p[s] else float("inf")
    return {
        "PRIMARY collapse (largest drop, ends at)": max(drops)[1] if drops else None,
        "PRIMARY answer top-10>=20%": first(lambda s: p[s]["answer"]["top10_rate"] >= 0.2, p),
        "PRIMARY inter top-10>0 (after step 0)": first(lambda s: s > 0 and p[s]["intermediate"]["top10_rate"] > 0, p),
        "alt collapse eff-dim<step0/2": first(lambda s: c[s]["eff_dim_jl_centered"] < 0.5 * e0, c) if e0 else None,
        "alt answer median<100": first(lambda s: p[s]["answer"]["median_rank"] < 100, p),
        "alt inter median<1000": first(lambda s: p[s]["intermediate"]["median_rank"] < 1000, p),
        "alt inter median<500&<foil/2": first(lambda s: p[s]["intermediate"]["median_rank"] < 500 and
                                              p[s]["intermediate"]["median_rank"] < foil(s) / 2, p),
    }

ap = argparse.ArgumentParser()
ap.add_argument("--olmo", default="OLMo-1B-0724-hf")
ap.add_argument("--pythia", default="pythia-1b-deduped")
ap.add_argument("--steps", type=int, nargs="*", default=[0, 1000, 2000, 3000, 4000, 6000, 9000, 13000, 23000, 43000, 73000, 103000, 143000])
args = ap.parse_args()
oc, op = load(args.olmo); pc, pp = load(args.pythia)
print(f"| step | eff-dim OLMo | eff-dim Pythia | drift OLMo | drift Pythia | answer OLMo | answer Pythia | intermediate OLMo | intermediate Pythia | foil OLMo | foil Pythia |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for s in args.steps:
    o = row(oc, op, s); q = row(pc, pp, s)
    print(f"| {s} | {o[0]} | {q[0]} | {o[1]} | {q[1]} | {o[2]} | {q[2]} | {o[3]} | {q[3]} | {o[4]} | {q[4]} |")
print("\nonset steps (OLMo):  ", onset(oc, op))
print("onset steps (Pythia):", onset(pc, pp))

# --- Summary statistics quoted in paper B Section 4.5 (added 2026-11-17) ---------------------
# Induction max over heads per step, late-window (13000-143000) median-of-medians and mean top-10
# for real vs foil intermediates, endpoint compression ratio raw/transported centered eff-dim, and
# a paired bootstrap (2000 resamples, seed 0, same item indices for both models) of the endpoint
# intermediate difference OLMo minus Pythia: top-10 rate (points) and log10 median rank.
import numpy as np

def induction(tag):
    f = ROOT / f"results/b3_induction_{tag}_heads.json"
    if not f.exists(): f = ROOT / f"results/b3_induction_{tag}.json"
    return {r["step"]: r["max_score"] for r in json.load(open(f))["rows"]}

LATE = [13000, 23000, 43000, 73000, 103000, 143000]
print("\ninduction max (OLMo):  ", {s: round(v, 3) for s, v in sorted(induction(args.olmo).items())})
print("induction max (Pythia):", {s: round(v, 3) for s, v in sorted(induction(args.pythia).items())})
for name, c, p in [(args.olmo, oc, op), (args.pythia, pc, pp)]:
    real = [p[s]["intermediate"]["median_rank"] for s in LATE]
    foil = [p[s]["intermediate_foil"]["median_rank"] for s in LATE]
    t_real = np.mean([p[s]["intermediate"]["top10_rate"] for s in LATE])
    t_foil = np.mean([p[s]["intermediate_foil"]["top10_rate"] for s in LATE])
    e = c[143000]
    n_int = len(p[143000]["intermediate"]["ranks"])
    print(f"{name}: late median-of-medians real {np.median(real):.0f} foil {np.median(foil):.0f}; "
          f"late top-10 real {100*t_real:.1f}% foil {100*t_foil:.1f}%; "
          f"endpoint eff-dim jl {e['eff_dim_jl_centered']:.1f} raw {e['eff_dim_raw_centered']:.1f} "
          f"ratio {e['eff_dim_raw_centered']/e['eff_dim_jl_centered']:.2f}; "
          f"endpoint top-10 items {round(p[143000]['intermediate']['top10_rate']*n_int)}/{n_int}")
ro = np.array(op[143000]["intermediate"]["ranks"], float)
rp = np.array(pp[143000]["intermediate"]["ranks"], float)
rng = np.random.default_rng(0); dt, dl = [], []
for _ in range(2000):
    idx = rng.integers(0, len(ro), size=len(ro))
    dt.append(100 * (np.mean(ro[idx] <= 10) - np.mean(rp[idx] <= 10)))
    dl.append(np.log10(np.median(rp[idx])) - np.log10(np.median(ro[idx])))
print(f"endpoint top-10 diff OLMo-Pythia: {100*(np.mean(ro<=10)-np.mean(rp<=10)):.1f} pts, "
      f"95% CI [{np.percentile(dt,2.5):.0f}, {np.percentile(dt,97.5):.0f}]")
print(f"endpoint log10 median diff Pythia-OLMo: {np.log10(np.median(rp))-np.log10(np.median(ro)):.2f}, "
      f"95% CI [{np.percentile(dl,2.5):.2f}, {np.percentile(dl,97.5):.2f}]")
for name, c in [(args.olmo, oc), (args.pythia, pc)]:
    print(f"{name} raw centered eff-dim / ratio raw:jl:",
          {s: f"{c[s]['eff_dim_raw_centered']:.1f} / {c[s]['eff_dim_raw_centered']/c[s]['eff_dim_jl_centered']:.2f}"
           for s in args.steps if s in c})
