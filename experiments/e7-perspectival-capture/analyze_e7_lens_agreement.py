"""E7 lens-behavior item agreement: does the fresh-forward lens read agree,
item by item, with what the model actually did?

The capture claim needs the lens read to be more than decoration: on an item
whose restatement was rewritten to the swapped question, the lens should read
the swapped entity, and on an item whose restatement was not rewritten it
should not. This script scores that concordance.

Criterion (both sides read off the full-band arm of run_e7.py):

    lens side      at the middle read layer (read_layers[1], the layer
                   run_e7.py already reports as lens_swap_outranks_mid),
                   the swapped entity b outranks the original entity a in
                   the hook-free continuation forward taken during the
                   restatement stage.
    behavior side  restate_b: the restatement reproduces the swapped
                   question.

    agreement = #{items where the two sides match} / n_ok

The middle read layer is fixed by run_e7.py, not chosen here; no threshold
is fitted. Reported alongside are the two cells that matter for the text:
captured items the lens disagrees with (should be 0) and non-captured items
the lens claims (the false-positive side).

Reads the released results only, so it re-derives the number without a GPU.

Run:  python3 experiments/e7-perspectival-capture/analyze_e7_lens_agreement.py
      -> results/e7_lens_agreement.json
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

TAGS = [
    "qwen17b", "qwen4b", "qwen8b", "qwen14b",
    "qwen35-9b", "qwen36-27b",
    "gemma2-2b", "gemma2-9b", "gemma2-27b",
]


def score(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    # run_e7.py scores only items that pass the clean baseline gate (its `ok`
    # set); use the same filter so n matches every other column of that run.
    records = [
        r for r in blob["records"]
        if r.get("baseline_ok") and "full" in r.get("arms", {})
    ]
    if not records:
        return None

    layers = sorted(records[0]["arms"]["full"]["lens_best"], key=int)
    if len(layers) < 2:
        return None
    mid = layers[1]

    agree = captured = lens_hits = disagree_captured = claim_uncaptured = 0
    for r in records:
        arm = r["arms"]["full"]
        cell = arm["lens_best"][mid]
        lens_says_swapped = cell["b"] < cell["a"]
        behaved_swapped = bool(arm["restate_b"])
        captured += behaved_swapped
        lens_hits += lens_says_swapped
        if lens_says_swapped == behaved_swapped:
            agree += 1
        elif behaved_swapped:
            disagree_captured += 1
        else:
            claim_uncaptured += 1

    n = len(records)
    return {
        "mid_read_layer": int(mid),
        "n": n,
        "captured": captured,
        "lens_says_swapped": lens_hits,
        "agree": agree,
        "agreement": agree / n,
        "captured_the_lens_disagrees_with": disagree_captured,
        "uncaptured_the_lens_claims": claim_uncaptured,
    }


def main() -> None:
    out = {}
    for tag in TAGS:
        got = score(RESULTS / f"e7_perspectival_{tag}.json")
        if got is None:
            continue
        out[tag] = got
        print(
            f"{tag:<12} L{got['mid_read_layer']:<3} "
            f"agree {got['agree']}/{got['n']} = {got['agreement']:.3f}   "
            f"captured {got['captured']}, "
            f"captured-disagreed {got['captured_the_lens_disagrees_with']}, "
            f"uncaptured-claimed {got['uncaptured_the_lens_claims']}"
        )

    dest = RESULTS / "e7_lens_agreement.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
