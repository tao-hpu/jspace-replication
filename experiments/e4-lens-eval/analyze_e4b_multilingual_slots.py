"""Per-slot breakdown of the strict covert readout on the official
multilingual lens-eval set.

Every item of lens-eval-multilingual.json lists four intermediates in a fixed
order, e.g. spanish-opposite-big: ["Spanish", "opposite", "big", "small"]:

    slot 0  the language name
    slot 1  the relation word
    slot 2  the English form of the prompt's input word
    slot 3  the English form of the answer (the model's next token is the
            answer in the prompt language, e.g. "pequeño")

run_e4_covert.py writes, per model, one target row per (item, slot) in this
order and one permutation-control row per (item, slot) that scores item
(i+7) % 107's intermediates at item i's position. This script regroups the
stored rows by slot and applies the strict criterion (lens rank < 10 and
mouth rank >= 100). No model is loaded.

Run:  .venv/bin/python experiments/e4-lens-eval/analyze_e4b_multilingual_slots.py
Output: results/e4b_multilingual_slots.json
"""

from __future__ import annotations

import json
import pathlib
from math import comb

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "third_party/jacobian-lens/data/evaluations/lens-eval-multilingual.json"
RES = ROOT / "results"
SLOTS = ["language name", "relation word", "input word (English)", "answer (English)"]


def fisher_greater(a: int, n1: int, c: int, n2: int) -> float:
    """One-sided Fisher exact test, H1: target rate > control rate."""
    k, n = a + c, n1 + n2
    return sum(comb(k, x) * comb(n - k, n1 - x) for x in range(a, min(k, n1) + 1)) / comb(n, n1)


def main() -> None:
    items = json.load(open(DATA))["items"]
    assert all(len(it["intermediates"]) == 4 for it in items)
    out = {}
    for model in ("qwen4b", "qwen17b"):
        rows = json.load(open(RES / f"e4b_covert_{model}.json"))["sets"]["multilingual"]["rows"]
        counts = {}
        for kind in ("target", "control"):
            rs = [r for r in rows if r["kind"] == kind]
            assert len(rs) == 4 * len(items)
            c = [0, 0, 0, 0]
            for idx, r in enumerate(rs):
                assert r["item"] == items[idx // 4]["name"]
                c[idx % 4] += r["j"] < 10 and r["m"] >= 100
            counts[kind] = c
        n = len(items)
        per_slot = []
        for s, name in enumerate(SLOTS):
            t, c = counts["target"][s], counts["control"][s]
            per_slot.append(dict(slot=s, name=name, covert=t, control=c, n=n,
                                 p_one_sided=fisher_greater(t, n, c, n)))
        t12 = counts["target"][1] + counts["target"][2]
        c12 = counts["control"][1] + counts["control"][2]
        out[model] = dict(per_slot=per_slot,
                          slots_1_2=dict(covert=t12, control=c12, n=2 * n,
                                         p_one_sided=fisher_greater(t12, 2 * n, c12, 2 * n)),
                          total=dict(covert=sum(counts["target"]), control=sum(counts["control"]), n=4 * n))
        print(model)
        for d in per_slot:
            print(f"  slot {d['slot']} {d['name']:22s} {d['covert']:3d}/{n} ({100*d['covert']/n:4.1f}%) "
                  f"vs control {d['control']:2d}/{n} ({100*d['control']/n:4.1f}%)  p={d['p_one_sided']:.2g}")
        s12 = out[model]["slots_1_2"]
        print(f"  slots 1+2 {s12['covert']}/{2*n} ({100*s12['covert']/(2*n):.1f}%) vs {s12['control']}/{2*n} "
              f"({100*s12['control']/(2*n):.1f}%)  p={s12['p_one_sided']:.2g}")
        print(f"  total {out[model]['total']}")
    (RES / "e4b_multilingual_slots.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
