#!/usr/bin/env python3
"""#17 layer-1: zero-cost prompt-set sensitivity reanalysis of existing JSONs.

No new model runs. For each headline number we ask: would the conclusion
survive a different draw from the same item pool?

  E6 (register@0.125, baseline_ok only):
    - rates by category (capital-template vs one-off facts) and by src
    - half-subsample percentile band (1000 seeded draws of n/2 items)
  E7 (per model):
    - capture (restate_b) half-subsample band
    - detection margin (mean yes_margin full - none): jackknife extremes
      and drop-top-3-influence value (is the mean carried by outliers?)
    - same for randdir margin (specificity check at 8B)
  E4b:
    - covert-hit concentration per set: share of covert hits owned by the
      top 3 items (a register/plan verdict carried by 2 items is fragile)

Output: results/sensitivity_reanalysis.json + printed summary.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
ALPHA = "register@0.125"
N_DRAWS = 1000


def rate(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def half_band(flags, seed=0):
    """2.5/97.5 percentile of the rate over random half-subsamples."""
    rng = random.Random(seed)
    n = len(flags)
    draws = sorted(rate(rng.sample(flags, n // 2)) for _ in range(N_DRAWS))
    return draws[int(0.025 * N_DRAWS)], draws[int(0.975 * N_DRAWS)]


def jackknife_mean(vals):
    n = len(vals)
    tot = sum(vals)
    loo = [(tot - v) / (n - 1) for v in vals]
    return min(loo), max(loo)


def drop_top3_mean(deltas):
    """Mean after removing the 3 items with largest |value|."""
    keep = sorted(deltas, key=abs)[:-3]
    return rate(keep)


def e6_category(prompt):
    return "capital" if ("The capital of" in prompt or "的首都是" in prompt) else "other"


def analyze_e6(path):
    d = json.load(open(path))
    recs = [r for r in d["records"] if r["baseline_ok"] and ALPHA in r["arms"]]
    out = {"n_ok": len(recs)}
    for key in ("flip_lang", "flip_full"):
        flags = [r["arms"][ALPHA][key] for r in recs]
        lo, hi = half_band(flags)
        out[key] = {
            "overall": rate(flags),
            "half_band_95": [lo, hi],
            "by_src": {s: rate([f for f, r in zip(flags, recs) if r["src"] == s]) for s in ("en", "zh")},
            "by_category": {
                c: {"n": len([1 for r in recs if e6_category(r["prompt"]) == c]),
                    "rate": rate([f for f, r in zip(flags, recs) if e6_category(r["prompt"]) == c])}
                for c in ("capital", "other")
            },
            "by_src_x_category": {
                f"{s}/{c}": {"n": len([1 for r in recs if r["src"] == s and e6_category(r["prompt"]) == c]),
                             "rate": rate([f for f, r in zip(flags, recs)
                                           if r["src"] == s and e6_category(r["prompt"]) == c])}
                for s in ("en", "zh") for c in ("capital", "other")
            },
        }
    return out


def analyze_e7(path):
    d = json.load(open(path))
    recs = [r for r in d["records"] if r["baseline_ok"]]
    out = {"n_ok": len(recs)}
    cap = [r["arms"]["full"]["restate_b"] for r in recs]
    lo, hi = half_band(cap)
    out["capture_restate_full"] = {"overall": rate(cap), "half_band_95": [lo, hi]}
    for arm in ("full", "randdir"):
        deltas = [r["arms"][arm]["yes_margin"] - r["arms"]["none"]["yes_margin"] for r in recs]
        jlo, jhi = jackknife_mean(deltas)
        out[f"margin_delta_{arm}"] = {
            "mean": rate(deltas),
            "jackknife_loo_range": [jlo, jhi],
            "drop_top3_influence": drop_top3_mean(deltas),
            "n_pos": sum(1 for x in deltas if x > 0),
        }
    return out


def analyze_e4b(path):
    d = json.load(open(path))
    out = {}
    for name, s in d["sets"].items():
        targets = [row for row in s["rows"] if row["kind"] == "target"]
        hits = [row for row in targets
                if row["j"] is not None and row["j"] < 10
                and row["m"] is not None and row["m"] >= 100]
        # cross-check the hit definition against the stored summary stat
        stored = s["stats"]["covert_strict"]
        recomputed = len(hits) / len(targets) if targets else 0.0
        assert abs(stored - recomputed) < 1e-9, f"{name}: {stored} vs {recomputed}"
        per_item = {}
        for row in hits:
            per_item[row["item"]] = per_item.get(row["item"], 0) + 1
        counts = sorted(per_item.values(), reverse=True)
        total = sum(counts)
        out[name] = {
            "covert_strict_rate": stored,
            "covert_hits": total,
            "items_with_hits": len(counts),
            "n_items": len({row["item"] for row in targets}),
            "top3_share": (sum(counts[:3]) / total) if total else None,
        }
    return out


def main():
    result = {"alpha_arm": ALPHA, "n_half_draws": N_DRAWS, "e6": {}, "e7": {}, "e4b": {}}
    for tag, fname in [("4b", "e6_covert_register_qwen4b.json"), ("1.7b", "e6_covert_register_qwen17b.json")]:
        p = R / fname
        if p.exists():
            result["e6"][tag] = analyze_e6(p)
    for tag, fname in [("1.7b", "e7_perspectival_qwen17b.json"), ("4b", "e7_perspectival_qwen4b.json"),
                       ("8b", "e7_perspectival_qwen8b.json"), ("14b", "e7_perspectival_qwen14b.json")]:
        p = R / fname
        if p.exists():
            result["e7"][tag] = analyze_e7(p)
    for tag, fname in [("4b", "e4b_covert_qwen4b.json"), ("1.7b", "e4b_covert_qwen17b.json")]:
        p = R / fname
        if p.exists():
            result["e4b"][tag] = analyze_e4b(p)

    out_path = R / "sensitivity_reanalysis.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}\n")

    for m, e in result["e6"].items():
        print(f"E6 {m} (n_ok={e['n_ok']})")
        for key in ("flip_lang", "flip_full"):
            x = e[key]
            lo, hi = x["half_band_95"]
            cats = x["by_category"]
            print(f"  {key}: {x['overall']:.1%}  half-band [{lo:.1%}, {hi:.1%}]  "
                  f"en {x['by_src']['en']:.1%} / zh {x['by_src']['zh']:.1%}  "
                  f"capital {cats['capital']['rate']:.1%} (n={cats['capital']['n']}) / "
                  f"other {cats['other']['rate']:.1%} (n={cats['other']['n']})")
            cells = x["by_src_x_category"]
            print("      " + "  ".join(f"{k}: {v['rate']:.0%} (n={v['n']})" for k, v in cells.items()))
    for m, e in result["e7"].items():
        print(f"E7 {m} (n_ok={e['n_ok']})")
        c = e["capture_restate_full"]
        print(f"  capture(restate,full): {c['overall']:.1%}  half-band [{c['half_band_95'][0]:.1%}, {c['half_band_95'][1]:.1%}]")
        for arm in ("full", "randdir"):
            x = e[f"margin_delta_{arm}"]
            print(f"  Δmargin {arm}: {x['mean']:+.3f}  LOO [{x['jackknife_loo_range'][0]:+.3f}, {x['jackknife_loo_range'][1]:+.3f}]  "
                  f"drop-top3 {x['drop_top3_influence']:+.3f}  pos {x['n_pos']}/{e['n_ok']}")
    for m, e in result["e4b"].items():
        print(f"E4b {m} covert-hit concentration (top-3 item share):")
        for name, s in e.items():
            share = f"{s['top3_share']:.0%}" if s["top3_share"] is not None else "--"
            print(f"  {name:13s} hits={s['covert_hits']:3d} items={s['items_with_hits']:3d} top3={share}")


if __name__ == "__main__":
    main()
