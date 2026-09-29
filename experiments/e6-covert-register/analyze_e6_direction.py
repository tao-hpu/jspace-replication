"""Language-register full flips by direction, fact template and covert status
(no model loaded).

Reads results/e6_covert_register_{qwen17b,qwen4b,gemma2-2b}.json and, at the
operating point register@0.125, counts full flips (language flipped and answer
correct) over the prompts that pass the clean-answer gate, split by

  * direction: src "en" is en->zh, src "zh" is zh->en;
  * template: "capital" (The capital of X is / X的首都是) vs every other fact;
  * covert status: whether the target-language form had mouth rank >= 100.

Run:  .venv/bin/python experiments/e6-covert-register/analyze_e6_direction.py
Output: results/e6_direction_template.json
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results"
MODELS = ("qwen17b", "qwen4b", "gemma2-2b")
ARM = "register@0.125"
DIRECTION = {"en": "en->zh", "zh": "zh->en"}


def template(prompt: str) -> str:
    return "capital" if ("The capital of" in prompt or "的首都是" in prompt) else "other"


def count(recs: list[dict]) -> dict:
    return dict(full=sum(r["arms"][ARM]["flip_full"] for r in recs),
                lang=sum(r["arms"][ARM]["flip_lang"] for r in recs), n=len(recs))


def main() -> None:
    out = {}
    for model in MODELS:
        recs = [r for r in json.load(open(RES / f"e6_covert_register_{model}.json"))["records"]
                if r["baseline_ok"]]
        cells = {}
        for src, name in DIRECTION.items():
            d = [r for r in recs if r["src"] == src]
            cells[name] = dict(all=count(d),
                               **{t: count([r for r in d if template(r["prompt"]) == t])
                                  for t in ("capital", "other")})
            for cov in (True, False):
                key = "covert" if cov else "non_covert"
                cells[name][key] = {t: count([r for r in d if r["covert"] == cov and template(r["prompt"]) == t])
                                    for t in ("capital", "other")}
        out[model] = dict(n_gate=len(recs), cells=cells)
        print(f"{model} (n={len(recs)})")
        for name, c in cells.items():
            print(f"  {name}: capital {c['capital']['full']}/{c['capital']['n']}  "
                  f"other {c['other']['full']}/{c['other']['n']}  "
                  f"all {c['all']['full']}/{c['all']['n']} = {100 * c['all']['full'] / c['all']['n']:.1f}%  "
                  f"(language flips {c['all']['lang']}/{c['all']['n']})")
            for key in ("covert", "non_covert"):
                cc = c[key]
                print(f"      {key:10s} capital {cc['capital']['full']}/{cc['capital']['n']}  "
                      f"other {cc['other']['full']}/{cc['other']['n']}")
    (RES / "e6_direction_template.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
