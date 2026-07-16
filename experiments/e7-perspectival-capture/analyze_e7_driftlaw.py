"""Merge the E7 single-layer profiles with coordinate drift and harvest
amplitude into one per-layer table, and compute the statistics the drift-law
paper reports.

Reads only results/*.json; loads no model and needs no GPU.

Inputs (all under results/):
  e7_profile_{tag}.json    single-layer capture profile, one arm per band layer
  e7_drift_{tag}.json      per-layer coordinate drift, cos(J_l^T W_U, W_U)
  e7_amplitude_{tag}.json  per-layer harvest amplitude, generation-free
  and the cross-domain variants e7_{profile,drift}_{states,currency}_{tag}.json

Output:
  results/e7_driftlaw_table.json
    .table   one row per (model, band layer): capture, flip, drift, rel_edit
    .stats   per-model and pooled Spearman(capture, drift); item-bootstrap
             CIs for the Gemma coefficients, per domain and pooled across
             domains; the two-factor OLS decomposition (drift, amplitude,
             family indicator), pooled and within-Gemma.

Metric caveat (2026-07-15 audit): restate capture is a substring grade with
an incidental-mention floor that grows with model scale (none-arm restate:
Gemma 2B 5.4% -> 9B 7.1% -> 27B 10.7%); at 27B it swamps the per-layer
signal. The flip metric's none-arm floor is exactly 0 in every model, so the
per-model stats and the generation regression are also reported on flips;
read those first.

Run:
  .venv/bin/python experiments/e7-perspectival-capture/analyze_e7_driftlaw.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

B_BOOT = 10_000
SEED = 0

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "e7_driftlaw_table.json"

MODELS = {
    "gemma2-2b": "Gemma-2-2B",
    "gemma2-9b": "Gemma-2-9B",
    "gemma2-27b": "Gemma-2-27B",
    "qwen17b": "Qwen3-1.7B",
    "qwen4b": "Qwen3-4B",
    "qwen8b": "Qwen3-8B",
    "qwen14b": "Qwen3-14B",
    "qwen35-9b": "Qwen3.5-9B",
    "qwen36-27b": "Qwen3.6-27B",
}

# Three-level family/generation coding for the pooled regression: the hybrid
# DeltaNet Qwens (3.5/3.6) sit systematically above the dense-Qwen capture
# trend at matched drift, so lumping them under one "qwen" dummy would smear
# the very residual the regression is trying to isolate.
GENERATION = {
    "gemma2-2b": "gemma", "gemma2-9b": "gemma", "gemma2-27b": "gemma",
    "qwen17b": "qwen-dense", "qwen4b": "qwen-dense",
    "qwen8b": "qwen-dense", "qwen14b": "qwen-dense",
    "qwen35-9b": "qwen-hybrid", "qwen36-27b": "qwen-hybrid",
}

# The six models of the original drift-law table; their binary-family
# regression is kept as a frozen block so previously reported coefficients
# stay traceable after the table grows.
LEGACY_SIX = ("gemma2-2b", "gemma2-9b", "qwen17b", "qwen4b", "qwen8b", "qwen14b")

# Gemma-2-9B was swept in two further item pools; each carries its own drift
# curve, because drift is averaged over that pool's own swapped entity tokens.
CROSS_DOMAINS = ("states", "currency")


def load(stem: str):
    p = RESULTS / f"{stem}.json"
    return json.loads(p.read_text()) if p.exists() else None


def spearman(xs, ys) -> float:
    """Rank correlation with midranks for ties; no scipy dependency."""

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def bootstrap_ci(rates: np.ndarray, rng) -> dict:
    """95% percentile item bootstrap of a mean over a 0/1 item vector."""
    n = len(rates)
    idx = rng.integers(0, n, size=(B_BOOT, n))
    boot = rates[idx].mean(axis=1)
    return {
        "rate": float(rates.mean()),
        "lo": float(np.percentile(boot, 2.5)),
        "hi": float(np.percentile(boot, 97.5)),
        "n_items": n,
    }


def build_table():
    """One row per (model, band layer), joining capture, drift, amplitude."""
    table, per_model, prof_records = [], {}, {}
    for tag in MODELS:
        prof, drift, amp = load(f"e7_profile_{tag}"), load(f"e7_drift_{tag}"), load(f"e7_amplitude_{tag}")
        if prof is None:
            print(f"skip {tag}: no profile")
            continue
        drift_by_layer = {d["layer"]: d["drift_entity"] for d in drift["profile"]} if drift else {}
        amp_by_layer = {a["layer"]: a for a in amp["profile"]} if amp else {}
        rows = []
        for p in prof["summary"]["profile"]:
            al = amp_by_layer.get(p["layer"], {})
            row = {
                "model": tag,
                "layer": p["layer"],
                "frac_depth": p["frac_depth"],
                "capture": p["restate_swapped"],
                "flip": p["flip"],
                "drift": drift_by_layer.get(p["layer"]),
                "rel_edit": al.get("rel_edit_mean"),
                "amp": al.get("amp_mean"),
                "edit_norm": al.get("edit_norm_mean"),
            }
            rows.append(row)
            table.append(row)
        per_model[tag] = (rows, prof["summary"])
        prof_records[tag] = prof["records"]
    return table, per_model, prof_records


def spearman_ci_for(domains, rng, key: str = "restate_b") -> dict | None:
    """Point estimate + item-bootstrap CI of Spearman(capture, drift).

    ``domains`` is a list of (drift_vector, layers, records). Capture rates are
    means over an item pool; drift is lens-plus-unembedding geometry with no
    item-sampling variance at the entity-set level, so the honest resampling
    unit is the item: resample items, recompute every layer's capture rate, and
    re-rank the resulting curve against the fixed drift curve. Stacking several
    domains resamples items within each domain independently.

    ``key`` picks the grade: "restate_b" (capture; incidental-mention floor)
    or "ans_b" (flip; floor-free). Returns None when the signal is
    identically zero (no ranks to correlate, e.g. flips on a closed model).

    The interval prices item-sampling noise only. It does not model dependence
    between layers of one model, which is real and would widen it.
    """
    caps, drifts = [], []
    for drift_v, layers, recs in domains:
        caps.append(np.array([[int(r["arms"][f"L{l}"][key]) for l in layers] for r in recs]))
        drifts.append(np.array(drift_v))
    if not any(c.any() for c in caps):
        return None
    drift_all = np.concatenate(drifts)
    point = spearman(list(drift_all), list(np.concatenate([c.mean(axis=0) for c in caps])))
    sps = []
    for _ in range(B_BOOT):
        rates = [c[rng.integers(0, c.shape[0], size=c.shape[0])].mean(axis=0) for c in caps]
        s = spearman(list(drift_all), list(np.concatenate(rates)))
        if not np.isnan(s):
            sps.append(s)
    sps = np.array(sps)
    return {
        "spearman": point,
        "lo": float(np.percentile(sps, 2.5)),
        "hi": float(np.percentile(sps, 97.5)),
        "B": B_BOOT,
        "seed": SEED,
        "n_items": int(sum(c.shape[0] for c in caps)),
        "n_layer_obs": int(len(drift_all)),
    }


def ols(target, *cols):
    X = np.column_stack([np.ones(len(target))] + list(cols))
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    sse = float(resid @ resid)
    sst = float(((target - target.mean()) ** 2).sum())
    return 1 - sse / sst, [float(b) for b in beta], sse


def regression(table) -> dict | None:
    """capture ~ drift + amplitude (+ family), pooled per-layer OLS.

    The decisive question: does the generation-free amplitude axis absorb the
    matched-drift family gap (Qwen's single-layer floor vs Gemma's humps), or
    does the writability gate stand as a family property? Run on the original
    six-model table with the binary gemma/qwen dummy, so the reported
    coefficients keep meaning the same thing as the table grows.
    """
    rows = [r for r in table
            if r["model"] in LEGACY_SIX
            and r["drift"] is not None and r["rel_edit"] is not None]
    if not rows:
        return None
    y = np.array([r["capture"] for r in rows])
    x_drift = np.array([r["drift"] for r in rows])
    x_amp = np.array([r["rel_edit"] for r in rows])
    x_fam = np.array([1.0 if r["model"].startswith("gemma") else 0.0 for r in rows])

    r2_d, beta_d, sse_d = ols(y, x_drift)
    r2_da, beta_da, sse_da = ols(y, x_drift, x_amp)
    r2_df, beta_df, sse_df = ols(y, x_drift, x_fam)
    r2_daf, beta_daf, sse_daf = ols(y, x_drift, x_amp, x_fam)
    out = {
        "n": len(rows),
        "note": "original six models; per-layer pooled OLS; amplitude = rel_edit_mean (edit norm / residual norm); family: gemma=1 qwen=0",
        "capture~drift": {"r2": r2_d, "beta": beta_d},
        "capture~drift+amp": {"r2": r2_da, "beta": beta_da, "partial_r2_amp_given_drift": 1 - sse_da / sse_d},
        "capture~drift+fam": {"r2": r2_df, "beta": beta_df, "partial_r2_fam_given_drift": 1 - sse_df / sse_d},
        "capture~drift+amp+fam": {
            "r2": r2_daf,
            "beta": beta_daf,
            "partial_r2_fam_given_drift_amp": 1 - sse_daf / sse_da,
            "partial_r2_amp_given_drift_fam": 1 - sse_daf / sse_df,
        },
    }
    g = [i for i, r in enumerate(rows) if r["model"].startswith("gemma")]
    if len(g) >= 8:
        yg, dg, ag = y[g], x_drift[g], x_amp[g]
        r2_gd, _, sse_gd = ols(yg, dg)
        r2_gda, _, sse_gda = ols(yg, dg, ag)
        out["within_gemma"] = {
            "n": len(g),
            "r2_drift": r2_gd,
            "r2_drift+amp": r2_gda,
            "partial_r2_amp_given_drift": 1 - sse_gda / sse_gd,
        }
    return out


def regression_generation(table, target: str = "capture") -> dict | None:
    """target ~ drift (+ generation dummies), pooled OLS over every model
    with a profile; target is "capture" (restate, floor-contaminated) or
    "flip" (floor-free).

    The nine-model extension of the family question: with hybrid-generation
    Qwens in the pool, "family" is three-valued (gemma / dense Qwen / hybrid
    Qwen; dense Qwen is the reference level). Reported alongside, never
    instead of, the frozen six-model block above.
    """
    rows = [r for r in table if r["drift"] is not None]
    tags = sorted({r["model"] for r in rows})
    if len(tags) <= len(LEGACY_SIX):
        return None  # nothing beyond the legacy block yet
    y = np.array([r[target] for r in rows])
    x_drift = np.array([r["drift"] for r in rows])
    x_g = np.array([1.0 if GENERATION[r["model"]] == "gemma" else 0.0 for r in rows])
    x_h = np.array([1.0 if GENERATION[r["model"]] == "qwen-hybrid" else 0.0 for r in rows])

    r2_d, beta_d, sse_d = ols(y, x_drift)
    r2_dg, beta_dg, sse_dg = ols(y, x_drift, x_g, x_h)
    out = {
        "n": len(rows),
        "models": tags,
        "note": "all profiled models; generation dummies gemma & qwen-hybrid, reference = dense qwen; beta order [const, drift, gemma, hybrid]",
        "capture~drift": {"r2": r2_d, "beta": beta_d},
        "capture~drift+gen": {
            "r2": r2_dg, "beta": beta_dg,
            "partial_r2_gen_given_drift": 1 - sse_dg / sse_d,
        },
    }
    amp_rows = [i for i, r in enumerate(rows) if r["rel_edit"] is not None]
    if len(amp_rows) == len(rows):
        x_amp = np.array([r["rel_edit"] for r in rows])
        r2_dag, beta_dag, sse_dag = ols(y, x_drift, x_amp, x_g, x_h)
        out["capture~drift+amp+gen"] = {
            "r2": r2_dag, "beta": beta_dag,
            "partial_r2_gen_given_drift_amp": None if sse_d == 0 else 1 - sse_dag / ols(y, x_drift, x_amp)[2],
        }
    return out


def main() -> None:
    table, per_model, prof_records = build_table()
    rng = np.random.default_rng(SEED)

    paired = [(r["drift"], r["capture"]) for r in table if r["drift"] is not None]
    stats = {
        "pooled_spearman_drift_capture": spearman([p[0] for p in paired], [p[1] for p in paired]),
        "n_layers_pooled": len(paired),
        "per_model": {},
    }

    for tag, (rows, summ) in per_model.items():
        pm = [(r["drift"], r["capture"]) for r in rows if r["drift"] is not None]
        if len(pm) < 5:
            continue
        # The unedited arm of the same run is the floor every profile is read
        # against: substring grading is not noiseless on Gemma.
        none_items = np.array([float(r["arms"]["none"]["restate_b"]) for r in prof_records[tag]])
        pm_f = [(r["drift"], r["flip"]) for r in rows if r["drift"] is not None]
        stats["per_model"][tag] = {
            "spearman": spearman([p[0] for p in pm], [p[1] for p in pm]),
            "n": len(pm),
            "peak_capture": max(r["capture"] for r in rows),
            "none_arm": bootstrap_ci(none_items, np.random.default_rng(SEED)),
            "full_band": summ["full"]["restate_swapped"],
            # Flip metric: none-arm flip floor is 0 in every model, so these
            # are the uncontaminated per-layer gate numbers.
            "spearman_flip": spearman([p[0] for p in pm_f], [p[1] for p in pm_f]),
            "peak_flip": max(r["flip"] for r in rows),
            "flip_layers_pos": sum(1 for r in rows if r["flip"] > 0),
            "none_flip": float(np.mean(
                [float(r["arms"]["none"]["ans_b"]) for r in prof_records[tag]])),
        }
        hi = stats["per_model"][tag]["none_arm"]["hi"]
        stats["per_model"][tag]["layers_above_none_ci"] = sum(
            1 for r in rows if r["capture"] > hi
        )

    for tag in ("gemma2-2b", "gemma2-9b", "gemma2-27b", "qwen35-9b"):
        if tag not in stats["per_model"]:
            continue
        rows = per_model[tag][0]
        layers = [r["layer"] for r in rows if r["drift"] is not None]
        drift_v = [r["drift"] for r in rows if r["drift"] is not None]
        recs = [r for r in prof_records[tag] if r.get("baseline_ok")]
        base = (drift_v, layers, recs)
        stats["per_model"][tag]["spearman_ci"] = spearman_ci_for([base], rng)
        stats["per_model"][tag]["spearman_flip_ci"] = spearman_ci_for(
            [base], rng, key="ans_b")

        entries, by_domain = [base], {"capitals": {
            "capture": stats["per_model"][tag]["spearman_ci"],
            "flip": stats["per_model"][tag]["spearman_flip_ci"],
        }}
        for dom in CROSS_DOMAINS:
            dprof, ddrift = load(f"e7_profile_{dom}_{tag}"), load(f"e7_drift_{dom}_{tag}")
            if dprof is None or ddrift is None:
                continue
            dmap = {d["layer"]: d["drift_entity"] for d in ddrift["profile"]}
            dlayers = [p["layer"] for p in dprof["summary"]["profile"] if p["layer"] in dmap]
            drecs = [r for r in dprof["records"] if r.get("baseline_ok")]
            entry = ([dmap[l] for l in dlayers], dlayers, drecs)
            entries.append(entry)
            by_domain[dom] = {
                "capture": spearman_ci_for([entry], rng),
                "flip": spearman_ci_for([entry], rng, key="ans_b"),
            }
        if len(entries) > 1:
            stats["per_model"][tag]["cross_domain"] = {
                "per_domain": by_domain,
                "pooled": spearman_ci_for(entries, rng),
                "pooled_flip": spearman_ci_for(entries, rng, key="ans_b"),
            }

    stats["regression"] = regression(table)
    stats["regression_generation"] = regression_generation(table)
    stats["regression_generation_flip"] = regression_generation(table, target="flip")

    OUT.write_text(json.dumps({"table": table, "stats": stats}, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
