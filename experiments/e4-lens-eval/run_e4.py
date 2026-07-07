"""E4: lens sensitivity vs false-positive rate on the six official eval sets.

The external review's remaining unquantified critique: the lens produces
plausible-looking readouts that are not actually there ("false positives").
This run puts numbers on it. For every item in each `lens-eval-*.json`,
at the set's designated readout position (per the evaluations README):

  - pass@k of the item's true intermediates  — sensitivity, the paper's
    lens-quality metric (min-over-layers rank, fraction of intermediates
    with rank < k, averaged over items);
  - pass@k of a *permutation control* — another item's intermediates scored
    identically at the same position. Any hit here is a false positive: the
    criterion fired on same-distribution content that is not in this prompt;
  - both of the above for the vanilla logit lens, so the J-lens improvement
    is measured against its 2020 baseline on the same footing.

Readout positions: poetry = last newline token (end of line 1); all other
sets = final prompt token (for multihop/multilingual/order-ops the `target`
is not part of the prompt, so "the token immediately preceding target" is
the final token).

order-ops intermediates are expanded to synonym sets per the README
(digit/word forms for numbers, symbol/word forms for operations); rank is
the min over single-token forms.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e4-lens-eval/run_e4.py [1.7b|4b]
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch
import transformers

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import MODELS  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e3-poetry"))
from run_e3 import min_rank, newline_position  # noqa: E402

EVAL_DIR = ROOT / "third_party/jacobian-lens/data/evaluations"
SETS = ["multihop", "multilingual", "poetry", "order-ops", "association", "typo"]
PASS_K = (1, 5, 10)
CTRL_OFFSET = 7  # fixed-offset permutation control, as in E3

NUMBER_WORDS = {
    "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
    "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve",
    "13": "thirteen", "15": "fifteen", "16": "sixteen", "20": "twenty",
    "24": "twenty-four",
}
OP_FORMS = {
    "addition": ["addition", "+", "plus", "add", "added", "sum"],
    "subtraction": ["subtraction", "-", "minus", "subtract", "subtracted", "difference"],
    "multiplication": ["multiplication", "*", "times", "multiply", "multiplied", "product"],
    "division": ["division", "/", "divided", "divide", "quotient"],
    "mod": ["mod", "%", "modulo", "remainder"],
    "squared": ["squared", "square", "^", "exponent", "power"],
}


def synonym_forms(slug: str, intermediate: str) -> list[str]:
    if slug != "order-ops":
        return [intermediate]
    if intermediate in OP_FORMS:
        return OP_FORMS[intermediate]
    return [intermediate] + ([NUMBER_WORDS[intermediate]] if intermediate in NUMBER_WORDS else [])


def token_ids(tok, slug: str, intermediate: str) -> list[int]:
    """Single-token encodings (leading space) of the intermediate's forms."""
    ids = []
    for form in synonym_forms(slug, intermediate):
        enc = tok.encode(" " + form.strip(), add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    if not ids:  # fall back to the first token of the plain form
        ids.append(tok.encode(" " + intermediate.strip(), add_special_tokens=False)[0])
    return ids


def best_rank(lens_logits, pos_idx: int, ids: list[int]) -> int:
    return min(min_rank(lens_logits, pos_idx, t)[0] for t in ids)


def main(model_key: str = "4b") -> None:
    model_id, lens_file, _ = MODELS[model_key]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens", filename=lens_file)
    print(f"model={model_id} layers={lens.source_layers[0]}..{lens.source_layers[-1]}")

    all_sets = {}
    for slug in SETS:
        items = json.load(open(EVAL_DIR / f"lens-eval-{slug}.json"))["items"]
        ids_per_item = [[token_ids(tok, slug, w) for w in it["intermediates"]] for it in items]

        records = []
        for i, it in enumerate(items):
            prompt = it["prompt"].rstrip()
            pos = newline_position(tok, prompt) if slug == "poetry" else -1
            j_logits, _, _ = lens.apply(model, prompt, positions=[pos], use_jacobian=True)
            l_logits, _, _ = lens.apply(model, prompt, positions=[pos], use_jacobian=False)
            ctrl = ids_per_item[(i + CTRL_OFFSET) % len(items)]
            records.append({
                "name": it["name"],
                "j_ranks": [best_rank(j_logits, 0, ids) for ids in ids_per_item[i]],
                "j_ctrl_ranks": [best_rank(j_logits, 0, ids) for ids in ctrl],
                "l_ranks": [best_rank(l_logits, 0, ids) for ids in ids_per_item[i]],
                "l_ctrl_ranks": [best_rank(l_logits, 0, ids) for ids in ctrl],
            })

        def pass_at(key):
            return {k: sum(sum(r < k for r in rec[key]) / len(rec[key]) for rec in records)
                       / len(records) for k in PASS_K}

        summary = {"n": len(items), "j": pass_at("j_ranks"), "j_ctrl": pass_at("j_ctrl_ranks"),
                   "logit": pass_at("l_ranks"), "logit_ctrl": pass_at("l_ctrl_ranks")}
        all_sets[slug] = {"summary": summary, "records": records}
        print(f"{slug:14s} n={len(items):3d}  " +
              "  ".join(f"{lbl} p@1={summary[lbl][1]:.1%} p@10={summary[lbl][10]:.1%}"
                        for lbl in ("j", "j_ctrl", "logit", "logit_ctrl")))

    out = ROOT / "results" / f"e4_qwen{model_key.replace('.', '')}.json"
    out.write_text(json.dumps({"model": model_id, "sets": all_sets}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "4b")
