"""E5: is the swap specific to the source concept, or is it coefficient
steering in disguise?

The swap h' = h - (h.dA)dA + (h.dA)dB harvests whatever amplitude the stream
has along dA and injects it along dB. It never checks that the harvested
amplitude *belongs* to concept A: related concepts have correlated transported
directions, so a swap sourced from a non-answer word can carry the answer's
own energy into the target. If that works as well as sourcing from the answer
itself, "we rewrote the model's answer representation" is not a supported
reading of the official demo; the mechanism is directed steering whose
strength happens to be borrowed from the source direction.

Three arms on the official 90 two-hop items, all injecting the SAME target
direction (swap_answer), varying only the source:

  arm B  source = answer            (the official answer-substitution, from E2)
  arm C  source = intermediate      (in the workspace, related, NOT the answer)
  arm D  source = an absent word    ("piano"; h.dA ~ 0, so injection ~ 0 --
                                     the do-nothing control the mechanism
                                     predicts)

Also recorded per arm: the harvested amplitude mean|h.dA| (band mean), and
per item the cosine between the transported source and answer directions at
the band's middle layer.

Predictions if the swap is mere coefficient steering: C flips comparably to
B while cos(d_C, d_answer) >> 0, and D flips nothing with near-zero
amplitude. Arm B outcomes are reused from the E2 results file (identical
configuration).

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e5-specificity/run_e5.py [1.7b|4b]
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch
import transformers

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.interventions import SwapHooks, token_direction  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import MODELS, greedy_gen, text_match  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e2-probe-swap"))
from run_e2 import mcnemar_exact_p  # noqa: E402

DATA = ROOT / "third_party/jacobian-lens/data/experiments/probe-swap.json"
ABSENT = "piano"  # not present in any of the 90 prompts


def main(model_key: str = "4b") -> None:
    model_id, lens_file, _ = MODELS[model_key]
    prior = json.loads((ROOT / "results" / f"e2_qwen{model_key.replace('.', '')}.json").read_text())
    lens_by_name = {r["name"]: r for r in prior["records"]}

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens", filename=lens_file)
    W = hf.get_output_embeddings().weight
    band = list(lens.source_layers)
    mid = band[len(band) // 2]
    print(f"model={model_id} band={band[0]}..{band[-1]} (cosines at L{mid})")

    items = json.load(open(DATA))["items"]
    tid = lambda w: tok.encode(" " + w.strip(), add_special_tokens=False)[0]  # noqa: E731

    records = []
    for it in items:
        assert ABSENT not in it["prompt"].lower()
        prompt = it["prompt"].rstrip()
        target = tid(it["swap_answer"])

        ctx_c = SwapHooks(model.layers, lens, W, tid(it["intermediate"]), target, band)
        _, c_text = greedy_gen(hf, tok, prompt, device, swap_ctx=ctx_c)
        c_amp = sum(ctx_c.coeff_abs.values()) / len(ctx_c.coeff_abs)

        ctx_d = SwapHooks(model.layers, lens, W, tid(ABSENT), target, band)
        _, d_text = greedy_gen(hf, tok, prompt, device, swap_ctx=ctx_d)
        d_amp = sum(ctx_d.coeff_abs.values()) / len(ctx_d.coeff_abs)

        d_ans = token_direction(lens, W, tid(it["answer"]), mid)
        cos_c = float(d_ans @ token_direction(lens, W, tid(it["intermediate"]), mid))
        cos_d = float(d_ans @ token_direction(lens, W, tid(ABSENT), mid))

        lr = lens_by_name[it["name"]]
        records.append({
            "name": it["name"], "baseline_ok": lr["baseline_ok"],
            "b_hit": lr["b_hit"],  # E2 arm B: source = answer
            "c_hit": text_match(c_text, it["swap_answer"]), "c_text": c_text,
            "d_hit": text_match(d_text, it["swap_answer"]), "d_text": d_text,
            "c_amp": c_amp, "d_amp": d_amp,
            "cos_intermediate_answer": cos_c, "cos_absent_answer": cos_d,
        })

    ok = [r for r in records if r["baseline_ok"]]
    rate = lambda k: sum(r[k] for r in ok) / len(ok)  # noqa: E731
    bc_b = sum(r["b_hit"] and not r["c_hit"] for r in ok)
    bc_c = sum(r["c_hit"] and not r["b_hit"] for r in ok)

    print(f"\nbaseline correct (from E2): {len(ok)}/{len(records)}")
    print(f"arm B (source=answer)       hit: {rate('b_hit'):.1%}")
    print(f"arm C (source=intermediate) hit: {rate('c_hit'):.1%}   "
          f"mean amp {sum(r['c_amp'] for r in ok)/len(ok):.2f}   "
          f"mean cos(d_int, d_ans) {sum(r['cos_intermediate_answer'] for r in ok)/len(ok):.3f}")
    print(f"arm D (source=absent)       hit: {rate('d_hit'):.1%}   "
          f"mean amp {sum(r['d_amp'] for r in ok)/len(ok):.2f}   "
          f"mean cos(d_abs, d_ans) {sum(r['cos_absent_answer'] for r in ok)/len(ok):.3f}")
    print(f"B vs C discordant: B-only {bc_b}, C-only {bc_c};  exact McNemar p = "
          f"{mcnemar_exact_p(bc_b, bc_c):.4f}")

    out = ROOT / "results" / f"e5_qwen{model_key.replace('.', '')}.json"
    out.write_text(json.dumps({"model": model_id, "band": [band[0], band[-1]], "mid_layer": mid,
                               "records": records}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "4b")
