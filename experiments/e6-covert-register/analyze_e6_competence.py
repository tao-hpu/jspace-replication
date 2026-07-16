"""E6 follow-up: is the cross-family efficacy gap explained by target-language
competence? A reanalysis of the stored E6 runs, no model forwards.

The paper's cross-family paragraph reads Gemma-2-2B's lower full-flip efficacy
(25.0% vs Qwen's 55.6%/64.6% at alpha=0.125) as "capability-coupled": weaker
Chinese, so the register flips the language but retrieves the answer less
often. That reading was inferred from the language-only vs full-flip gap, not
from an independent competence measurement. This script supplies the
measurement from data the E6 runs already contain: every item exists in both
languages, and the *baseline* arm of the opposite-language variant is exactly
"does the model produce the target-language answer under native prompting,
no intervention".

Definitions, per model and per flip direction (en->zh on English prompts,
zh->en on Chinese):

  flip-eligible   source-side baseline correct (the E6 flip denominator;
                  reproduces the paper's pooled n exactly: 72 / 63 / 79)
  target-known    the paired opposite-language prompt's baseline is also
                  correct, i.e. the model demonstrably knows the answer in
                  the target language
  coverage        P(target-known | flip-eligible)
  full flip | known / unknown
                  register@0.125 full-flip rate on the two subsets

Two findings (2026-07-16, from the stored headline runs):
  1. WITHIN each model, competence gates retrieval: full flip roughly doubles
     on target-known items in all three models.
  2. ACROSS models, competence explains nothing: coverage is flat
     (Gemma 75.0% vs Qwen 79.4%/81.0%) and the efficacy gap survives intact
     on the target-known subset (27.8% vs 62.0%/71.9%).
So the cross-family modulation belongs to the register write itself, not to
missing answer knowledge.

Run:  .venv/bin/python experiments/e6-covert-register/analyze_e6_competence.py
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
ALPHA = "0.125"  # the operating point every headline E6 flip number uses
MODELS = ("gemma2-2b", "qwen17b", "qwen4b")


def analyze(model: str) -> dict:
    d = json.loads((ROOT / "results" / f"e6_covert_register_{model}.json").read_text())
    pairs: dict[tuple[str, str], dict] = {}
    for r in d["records"]:
        pairs.setdefault((r["en"], r["zh"]), {})[r["src"]] = r

    out = {"alpha": ALPHA, "n_pairs": len(pairs), "directions": {}, "pooled": {}}
    pooled = {"eligible": 0, "known": 0,
              "full_known": 0, "full_unknown": 0, "lang_known": 0}
    for src, tgt in (("en", "zh"), ("zh", "en")):
        st = {"eligible": 0, "known": 0, "full_known": 0, "full_unknown": 0,
              "lang_known": 0}
        for pair in pairs.values():
            if len(pair) < 2:
                continue
            r = pair[src]
            if not r["arms"]["baseline"][f"hit_{src}"]:
                continue  # not flip-eligible
            arm = r["arms"].get(f"register@{ALPHA}")
            if arm is None:
                continue
            known = pair[tgt]["arms"]["baseline"][f"hit_{tgt}"]
            st["eligible"] += 1
            if known:
                st["known"] += 1
                st["full_known"] += int(arm["flip_full"])
                st["lang_known"] += int(arm["flip_lang"])
            else:
                st["full_unknown"] += int(arm["flip_full"])
        out["directions"][f"{src}->{tgt}"] = st
        for k in pooled:
            pooled[k] += st[k]
    unk = pooled["eligible"] - pooled["known"]
    out["pooled"] = {
        **pooled,
        "coverage": pooled["known"] / pooled["eligible"],
        "full_flip_given_known": pooled["full_known"] / pooled["known"],
        "full_flip_given_unknown": (pooled["full_unknown"] / unk) if unk else None,
        "lang_flip_given_known": pooled["lang_known"] / pooled["known"],
    }
    return out


def main() -> None:
    results = {m: analyze(m) for m in MODELS}
    for m, r in results.items():
        p = r["pooled"]
        unk = p["eligible"] - p["known"]
        print(f"{m:12s} eligible={p['eligible']} coverage={p['coverage']:.1%} "
              f"full|known={p['full_known']}/{p['known']}={p['full_flip_given_known']:.1%} "
              f"full|unknown={p['full_unknown']}/{unk}"
              f"={p['full_flip_given_unknown']:.1%} "
              f"lang|known={p['lang_flip_given_known']:.1%}")
    out = ROOT / "results" / "e6_competence.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
