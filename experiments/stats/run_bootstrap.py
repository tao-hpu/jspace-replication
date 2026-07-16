"""Bootstrap confidence intervals for all headline rates.

Reads item-level records from results/*.json and emits results/bootstrap_ci.json
with percentile bootstrap CIs (B=10,000, seed=0) for every rate quoted in the
paper draft, plus paired differences where the comparison is the claim.

Method notes:
- Single rates and paired differences use the ordinary percentile bootstrap
  over items. Items are independent facts/prompts, so item resampling is the
  right unit.
- E4b readouts nest inside items (several intermediate tokens per item), so
  those use a cluster bootstrap: resample items, keep all their rows. A flat
  row bootstrap would understate the interval.
- E4b covert-strict row rule (verified against stored stats): lens rank < 10
  and mouth rank >= 100 (missing mouth rank counts as covert).
- E7 zone maps (slide6) and consumption deadlines carry item-level records
  and get ordinary item bootstraps. Deadline domain pools are different fact
  sets, so cross-domain gaps are reported per domain, never as paired
  differences. Healing curves store per-layer aggregates only (no item
  records) and are not bootstrappable from the result files.

Run:  .venv/bin/python experiments/stats/run_bootstrap.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
B = 10_000
SEED = 0
MODELS = ["qwen17b", "qwen4b"]
# the E7 scale ladder extends beyond the models that ran the full suite; the
# Gemma cross-family points are included only once their run has landed, so a
# partially-run ladder never crashes the bootstrap.
E7_MODELS = [m for m in ["qwen17b", "qwen4b", "qwen8b", "qwen14b",
                         "gemma2-2b", "gemma2-9b", "gemma2-27b",
                         "qwen36-27b", "qwen35-9b"]
             if (RESULTS / f"e7_perspectival_{m}.json").exists()]

rng = np.random.default_rng(SEED)


def ci(values: np.ndarray) -> dict:
    """Percentile bootstrap CI for the mean of a 0/1 (or real) item vector."""
    n = len(values)
    idx = rng.integers(0, n, size=(B, n))
    means = values[idx].mean(axis=1)
    return {
        "est": float(values.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n": n,
    }


def ci_cluster(clusters: list[np.ndarray]) -> dict:
    """Cluster bootstrap: resample clusters, pool their rows, take the mean."""
    n = len(clusters)
    total = np.concatenate(clusters)
    means = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, n, size=n)
        means[b] = np.concatenate([clusters[i] for i in pick]).mean()
    return {
        "est": float(total.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n_clusters": n,
        "n_rows": int(len(total)),
    }


def load(name: str) -> dict:
    return json.load(open(RESULTS / name))


def e6_block(model: str) -> dict:
    d = load(f"e6_covert_register_{model}.json")
    recs = [r for r in d["records"] if r["baseline_ok"]]
    out = {}
    for alpha in d["alphas"]:
        key = f"register@{alpha}"
        if key not in recs[0]["arms"]:
            continue
        reg = [r["arms"][key] for r in recs]
        rnd = [r["arms"][f"random@{alpha}"] for r in recs]
        blk = {
            "flip_lang": ci(np.array([x["flip_lang"] for x in reg], float)),
            "flip_full": ci(np.array([x["flip_full"] for x in reg], float)),
            "preserved": ci(np.array([x["preserved"] for x in reg], float)),
            "flip_full_minus_random": ci(
                np.array(
                    [float(a["flip_full"]) - float(b["flip_full"]) for a, b in zip(reg, rnd)]
                )
            ),
        }
        covert = [r for r in recs if r["covert"]]
        if covert:
            blk["covert_subset_flip_full"] = ci(
                np.array([r["arms"][key]["flip_full"] for r in covert], float)
            )
        out[f"alpha={alpha}"] = blk
    return out


def e7_block(model: str) -> dict:
    d = load(f"e7_perspectival_{model}.json")
    recs = [r for r in d["records"] if r["baseline_ok"]]
    mid = str(d["read_layers"][1])
    out = {}
    for arm in recs[0]["arms"]:
        a = [r["arms"][arm] for r in recs]
        out[arm] = {
            "flip": ci(np.array([x["ans_b"] for x in a], float)),
            "restate_swapped": ci(np.array([x["restate_b"] for x in a], float)),
            "says_yes": ci(np.array([x["says_yes"] for x in a], float)),
            "lens_swap_outranks": ci(
                np.array(
                    [x["lens_best"][mid]["b"] < x["lens_best"][mid]["a"] for x in a], float
                )
            ),
        }
    for arm in ("full", "randdir"):
        out[f"yes_margin_delta_{arm}_vs_none"] = ci(
            np.array(
                [
                    r["arms"][arm]["yes_margin"] - r["arms"]["none"]["yes_margin"]
                    for r in recs
                ]
            )
        )
    return out


def e6t_block(model: str) -> dict:
    """Typo-register erasure: correction survival per dose, erase vs random."""
    d = load(f"e6t_typo_register_{model}.json")
    ok = [r for r in d["records"] if r["correction"]["baseline"]["fixed"]]
    out = {}
    for alpha in d["alphas"]:
        blk = {}
        for arm in ("erase", "random"):
            key = f"{arm}@{alpha:g}"
            blk[f"correction_{arm}"] = ci(
                np.array([r["correction"][key]["fixed"] for r in ok], float)
            )
        out[f"alpha={alpha:g}"] = blk
    return out


def e4b_block(model: str) -> dict:
    d = load(f"e4b_covert_{model}.json")
    out = {}
    for set_name, s in d["sets"].items():
        blk = {}
        for kind in ("target", "control"):
            rows = [r for r in s["rows"] if r["kind"] == kind]
            by_item: dict[str, list[float]] = {}
            for r in rows:
                covert = float(
                    r["j"] is not None
                    and r["j"] < 10
                    and (r["m"] is None or r["m"] >= 100)
                )
                by_item.setdefault(r["item"], []).append(covert)
            blk["covert_strict" if kind == "target" else "floor"] = ci_cluster(
                [np.array(v) for v in by_item.values()]
            )
        out[set_name] = blk
    return out


def slide6_block(fname: str) -> dict:
    """Width-6 zone map: per-window flip / restate CIs over items."""
    d = load(fname)
    recs = [r for r in d["records"] if r["baseline_ok"]]
    out = {}
    for arm in recs[0]["arms"]:
        a = [r["arms"][arm] for r in recs]
        out[arm] = {
            "flip": ci(np.array([x["ans_b"] for x in a], float)),
            "restate_swapped": ci(np.array([x["restate_b"] for x in a], float)),
        }
    return out


def phrasing_block(model: str) -> dict:
    """Report-phrasing robustness: paired margin deltas (full / randdir vs
    none) per phrasing, decision-token grading (run_e7_phrasing.py)."""
    d = load(f"e7_phrasing_{model}.json")
    recs = [r for r in d["records"] if r["baseline_ok"]]
    out = {}
    for ph in d["phrasings"]:
        blk = {}
        for arm in ("full", "randdir"):
            blk[f"yes_margin_delta_{arm}_vs_none"] = ci(
                np.array(
                    [
                        r["arms"][arm]["reports"][ph]["yes_margin"]
                        - r["arms"]["none"]["reports"][ph]["yes_margin"]
                        for r in recs
                    ]
                )
            )
        for arm in ("none", "full", "randdir"):
            blk[f"says_yes_{arm}"] = ci(
                np.array([r["arms"][arm]["reports"][ph]["says_yes"] for r in recs], float)
            )
        out[ph] = blk
    return out


def deadline_block(model: str) -> dict:
    """Consumption deadline: CI over per-item commit layers, per domain."""
    d = load(f"e7_deadline_{model}.json")
    n_layers = d["n_layers"]
    out = {}
    for dom, blk in d["domains"].items():
        if not blk["n_used"]:
            out[dom] = {"n": 0, "n_skipped": blk["n_skipped"]}
            continue
        c = ci(np.array(blk["commit_layers"], float))
        c["n_skipped"] = blk["n_skipped"]
        c["frac"] = {k: c[k] / n_layers for k in ("est", "lo", "hi")}
        out[dom] = c
    return out


def paired_hits_block(name: str, model: str, arms: list[str]) -> dict:
    d = load(f"{name}_{model}.json")
    recs = [r for r in d["records"] if r["baseline_ok"]]
    out = {arm: ci(np.array([r[f"{arm}_hit"] for r in recs], float)) for arm in arms}
    if len(arms) >= 2:
        a0, a1 = arms[0], arms[1]
        out[f"{a1}_minus_{a0}"] = ci(
            np.array([float(r[f"{a1}_hit"]) - float(r[f"{a0}_hit"]) for r in recs])
        )
    return out


def main() -> None:
    report = {
        "seed": SEED,
        "B": B,
        "notes": [
            "e7 says_yes is the greedy-text criterion; at qwen17b it carries an "
            "~81% baseline yes bias (none arm) and must not be quoted as a "
            "detection rate there — use yes_margin_delta_* instead.",
            "yes_margin_delta_* values are log-odds (logits), not rates.",
        ],
        "models": {},
    }
    for model in E7_MODELS:
        report["models"][model] = {"e7": e7_block(model)}
    for model in MODELS:
        report["models"][model].update({
            "e6": e6_block(model),
            "e6t": e6t_block(model),
            "e4b": e4b_block(model),
            # E2: arm a = intermediate-entity swap, arm b = answer-token control
            "e2": paired_hits_block("e2", model, ["a", "b"]),
            # E2p: same arms with mass-mean probe directions
            "e2p": paired_hits_block("e2p", model, ["a", "b"]),
            # E2t: same arms with trained-probe directions
            "e2t": paired_hits_block("e2t", model, ["a", "b"]),
            # E5: b = answer source, c = intermediate source, d = absent word
            "e5": paired_hits_block("e5", model, ["c", "b", "d"]),
        })
    # E7 follow-up families, appended AFTER every pre-existing block so the
    # shared rng sequence (seed=0) reproduces the earlier CIs bit for bit.
    # Zone maps: e7_slide6_{model}.json (capitals) and
    # e7_slide6_{states|currency}_{model}.json; discovered by glob so a
    # partially-run ladder never crashes the bootstrap.
    for p in sorted(RESULTS.glob("e7_slide6_*.json")):
        parts = p.stem.split("_")  # e7, slide6, [domain,] model
        domain, model = (parts[2], parts[3]) if len(parts) == 4 else ("capitals", parts[2])
        report["models"].setdefault(model, {})[f"e7_slide6_{domain}"] = slide6_block(p.name)
    for p in sorted(RESULTS.glob("e7_deadline_*.json")):
        model = p.stem.split("e7_deadline_")[1]
        report["models"].setdefault(model, {})["e7_deadline"] = deadline_block(model)
    # Report-phrasing robustness runs (run_e7_phrasing.py); seed/smoke-tagged
    # files are working artifacts and stay out of the headline CIs.
    for p in sorted(RESULTS.glob("e7_phrasing_*.json")):
        model = p.stem.split("e7_phrasing_")[1]
        if "_seed" in model or "_smoke" in model:
            continue
        report["models"].setdefault(model, {})["e7_phrasing"] = phrasing_block(model)
    out_path = RESULTS / "bootstrap_ci.json"
    json.dump(report, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}")
    for model in MODELS:
        m = report["models"][model]
        a125 = m["e6"].get("alpha=0.125", {})
        if a125:
            f = a125["flip_full"]
            print(
                f"  {model} e6 flip_full@0.125: {f['est']:.1%} [{f['lo']:.1%}, {f['hi']:.1%}] (n={f['n']})"
            )
        r = m["e7"]["full"]["restate_swapped"]
        print(
            f"  {model} e7 full restate_swapped: {r['est']:.1%} [{r['lo']:.1%}, {r['hi']:.1%}]"
        )


if __name__ == "__main__":
    main()
