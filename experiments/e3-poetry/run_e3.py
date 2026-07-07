"""E3: rhyme planning readout (C3), expected negative.

Official ``lens-eval-poetry.json``: 98 couplets whose prompt stops right
before the line-2 rhyme word. The planning claim: the rhyme word is already
readable in the lens at the *end of line 1* (the last newline token), before
the model has produced any of line 2.

Per item, at the last-newline position and at the final prompt token:

  - J-lens rank of the rhyme word (min over all fitted layers) — the claim.
  - Logit-lens rank at the same positions — is apparent planning lens-specific?
  - Permutation control: J-lens rank of *another item's* rhyme word at the
    same newline position — the base rate a "planning hit" must beat.
  - Greedy 6-token continuation — does the model even complete the couplet
    with the intended rhyme word? Planning readout is only meaningful on
    items the model actually rhymes as intended.

Metric: pass@k (rank < k) for k in {1, 5, 10}, following the upstream
evaluations README ("min-over-layers lens rank <= k", all layers, single
position).

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e3-poetry/run_e3.py [1.7b|4b]
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
from run_e1 import MODELS, greedy_gen, text_match  # noqa: E402

DATA = ROOT / "third_party/jacobian-lens/data/evaluations/lens-eval-poetry.json"
PASS_K = (1, 5, 10)


def newline_position(tok, prompt: str) -> int:
    """Index of the token covering the last newline character."""
    char = prompt.rfind("\n")
    assert char >= 0, "poetry prompt must contain a newline"
    offsets = tok(prompt, return_offsets_mapping=True).offset_mapping
    for i, (a, b) in enumerate(offsets):
        if a <= char < b:
            return i
    raise ValueError("no token covers the newline")


def min_rank(lens_logits: dict[int, torch.Tensor], pos_idx: int, token_id: int) -> tuple[int, int]:
    """(best rank over layers, argmin layer) for token_id at position pos_idx."""
    best, best_layer = None, None
    for layer, logits in lens_logits.items():
        v = logits[pos_idx]
        r = int((v > v[token_id]).sum())
        if best is None or r < best:
            best, best_layer = r, layer
    return best, best_layer


def main(model_key: str = "4b") -> None:
    model_id, lens_file, _ = MODELS[model_key]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens", filename=lens_file)
    print(f"model={model_id} layers={lens.source_layers[0]}..{lens.source_layers[-1]}")

    items = json.load(open(DATA))["items"]
    targets = [tok.encode(" " + it["intermediates"][0].strip(), add_special_tokens=False)[0]
               for it in items]

    records = []
    for i, it in enumerate(items):
        prompt = it["prompt"].rstrip()
        nl = newline_position(tok, prompt)
        tid = targets[i]
        ctrl_tid = targets[(i + 7) % len(items)]  # fixed-offset permutation control

        j_logits, _, _ = lens.apply(model, prompt, positions=[nl, -1], use_jacobian=True)
        l_logits, _, _ = lens.apply(model, prompt, positions=[nl, -1], use_jacobian=False)

        j_nl, j_nl_layer = min_rank(j_logits, 0, tid)
        ctrl_nl, _ = min_rank(j_logits, 0, ctrl_tid)
        j_fin, _ = min_rank(j_logits, 1, tid)
        l_nl, _ = min_rank(l_logits, 0, tid)
        l_fin, _ = min_rank(l_logits, 1, tid)

        _, gen = greedy_gen(hf, tok, prompt, device)
        records.append({
            "name": it["name"], "target": it["intermediates"][0],
            "baseline_hit": text_match(gen, it["intermediates"][0]), "gen": gen,
            "j_rank_newline": j_nl, "j_layer_newline": j_nl_layer,
            "ctrl_rank_newline": ctrl_nl,
            "logit_rank_newline": l_nl,
            "j_rank_final": j_fin, "logit_rank_final": l_fin,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(items)}")

    def pass_at(rows, key):
        return {k: sum(r[key] < k for r in rows) / len(rows) for k in PASS_K}

    rhymed = [r for r in records if r["baseline_hit"]]
    summary = {
        "n": len(records), "n_rhymed": len(rhymed),
        "newline_j": pass_at(records, "j_rank_newline"),
        "newline_j_rhymed_only": pass_at(rhymed, "j_rank_newline") if rhymed else None,
        "newline_control": pass_at(records, "ctrl_rank_newline"),
        "newline_logit": pass_at(records, "logit_rank_newline"),
        "final_j": pass_at(records, "j_rank_final"),
        "final_logit": pass_at(records, "logit_rank_final"),
        "median_j_rank_newline": sorted(r["j_rank_newline"] for r in records)[len(records) // 2],
        "median_ctrl_rank_newline": sorted(r["ctrl_rank_newline"] for r in records)[len(records) // 2],
    }

    print(f"\nbaseline rhymes as intended: {len(rhymed)}/{len(records)}")
    for key in ("newline_j", "newline_j_rhymed_only", "newline_control", "newline_logit",
                "final_j", "final_logit"):
        print(f"  {key:24s} " + (" ".join(f"pass@{k}={v:.1%}" for k, v in summary[key].items())
                                 if summary[key] else "n/a"))
    print(f"  median newline rank: target {summary['median_j_rank_newline']}, "
          f"control {summary['median_ctrl_rank_newline']}")

    out = ROOT / "results" / f"e3_qwen{model_key.replace('.', '')}.json"
    out.write_text(json.dumps({"model": model_id, "records": records, "summary": summary}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "4b")
