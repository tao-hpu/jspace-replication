"""Threshold sweep and exact tests for the covert audit (no model loaded).

For every official lens-eval set and both Qwen3 models this script reads the
stored per-row ranks of results/e4b_covert_{qwen4b,qwen17b}.json and reports

  * the covert rate (lens rank < 10 and mouth rank >= T) of the target rows
    and of the permutation-control rows at T in {10, 50, 100, 500};
  * at the strict cut T = 100, a one-sided Fisher exact test of target against
    control (H1: target rate > control rate), with Holm adjustment over the
    12 set-by-model tests;
  * the same one-sided test for the multilingual slots reported in the paper
    (language name, relation and input words pooled, English answer), read
    from results/e4b_multilingual_slots.json, with Holm adjustment over those
    six tests.

run_e4_covert.py computes two cuts in the same run (mouth rank >= 10 and
>= 100); the paper leads with the stricter one. Target and control rows share
a readout position, so the Fisher test treats paired rows as independent.

Run:  .venv/bin/python experiments/e4-lens-eval/analyze_e4b_sweep.py
Output: results/e4b_sweep_tests.json
"""

from __future__ import annotations

import json
import pathlib
from math import comb

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results"
MODELS = ("qwen4b", "qwen17b")
CUTS = (10, 50, 100, 500)
STRICT = 100


def fisher_greater(a: int, n1: int, c: int, n2: int) -> float:
    """One-sided Fisher exact test, H1: target rate > control rate."""
    k, n = a + c, n1 + n2
    return sum(comb(k, x) * comb(n - k, n1 - x) for x in range(a, min(k, n1) + 1)) / comb(n, n1)


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m, running, adj = len(ps), 0.0, [0.0] * len(ps)
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * ps[i]))
        adj[i] = running
    return adj


def covert(rows: list[dict], cut: int) -> int:
    return sum(1 for r in rows if r["j"] < 10 and r["m"] >= cut)


def main() -> None:
    sweep, tests = {}, []
    for model in MODELS:
        sets = json.load(open(RES / f"e4b_covert_{model}.json"))["sets"]
        sweep[model] = {}
        for name, s in sets.items():
            tgt = [r for r in s["rows"] if r["kind"] == "target"]
            ctl = [r for r in s["rows"] if r["kind"] == "control"]
            sweep[model][name] = {
                str(t): dict(target=covert(tgt, t), control=covert(ctl, t),
                             n_target=len(tgt), n_control=len(ctl))
                for t in CUTS}
            a, c = covert(tgt, STRICT), covert(ctl, STRICT)
            tests.append(dict(model=model, set=name, target=a, n_target=len(tgt),
                              control=c, n_control=len(ctl),
                              p_one_sided=fisher_greater(a, len(tgt), c, len(ctl))))
    for t, p in zip(tests, holm([t["p_one_sided"] for t in tests])):
        t["p_holm"] = p

    slots = json.load(open(RES / "e4b_multilingual_slots.json"))
    slot_tests = []
    for model in MODELS:
        d = slots[model]
        for label, row in (("language name", d["per_slot"][0]),
                           ("relation and input (English)", d["slots_1_2"]),
                           ("answer (English)", d["per_slot"][3])):
            slot_tests.append(dict(model=model, slot=label, target=row["covert"],
                                   control=row["control"], n=row["n"],
                                   p_one_sided=row["p_one_sided"]))
    for t, p in zip(slot_tests, holm([t["p_one_sided"] for t in slot_tests])):
        t["p_holm"] = p

    print("covert rate, target (control), 4B / 1.7B")
    for name in sweep[MODELS[0]]:
        cells = []
        for t in CUTS:
            parts = []
            for model in MODELS:
                x = sweep[model][name][str(t)]
                parts.append(f"{100 * x['target'] / x['n_target']:.1f} ({100 * x['control'] / x['n_control']:.1f})")
            cells.append(f"T={t}: " + " / ".join(parts))
        print(f"  {name:13s} " + " | ".join(cells))
    print("\nstrict cut, one-sided Fisher, Holm over 12 set-by-model tests")
    for t in tests:
        print(f"  {t['model']:8s} {t['set']:13s} {t['target']}/{t['n_target']} vs {t['control']}/{t['n_control']}"
              f"  p={t['p_one_sided']:.3g}  holm={t['p_holm']:.3g}")
    print("\nmultilingual slots, one-sided Fisher, Holm over 6 tests")
    for t in slot_tests:
        print(f"  {t['model']:8s} {t['slot']:29s} {t['target']}/{t['n']} vs {t['control']}/{t['n']}"
              f"  p={t['p_one_sided']:.3g}  holm={t['p_holm']:.3g}")
    (RES / "e4b_sweep_tests.json").write_text(json.dumps(
        dict(cuts=list(CUTS), strict_cut=STRICT, sweep=sweep, set_tests=tests, slot_tests=slot_tests), indent=2))


if __name__ == "__main__":
    main()
