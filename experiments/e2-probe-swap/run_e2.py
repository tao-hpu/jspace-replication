"""E2: thought-swap (C2) with the control the public review lacked.

Official 90 two-hop prompts. Three measurements per item:

  baseline   greedy continuation of the raw prompt; correct if it matches
             ``answer``.
  arm A      swap the *intermediate* entity (intermediate -> swap_to) across
             the band at all prompt positions; hit if the continuation
             matches ``swap_answer``. This is the paper's headline
             "rewriting a thought" intervention.
  arm B      control: swap the *answer* token directly (answer -> swap_answer)
             with the identical mechanism, band, and positions; hit if the
             continuation matches ``swap_answer``. This operationalizes the
             review's critique that A may be "close to substituting the
             final answer token directly".

If A does not beat B by a clear margin (paired McNemar over the 90 items),
the intermediate-rewriting interpretation does not hold.

Known deviation from upstream: the official experiment swaps along a
*linear-probe* direction; we use the Jacobian-lens direction (same operator
as E1) for both arms, so the A-vs-B comparison stays internally fair.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e2-probe-swap/run_e2.py [1.7b|4b]
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
from src.interventions import SwapHooks  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import MODELS, greedy_gen, text_match  # noqa: E402

DATA = ROOT / "third_party/jacobian-lens/data/experiments/probe-swap.json"


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main(model_key: str = "4b") -> None:
    model_id, lens_file, _ = MODELS[model_key]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens", filename=lens_file)
    W = hf.get_output_embeddings().weight
    band = list(lens.source_layers)  # E1 sweep: early-mid layers carry the effect
    print(f"model={model_id} band={band[0]}..{band[-1]}")

    items = json.load(open(DATA))["items"]
    records = []
    for it in items:
        prompt = it["prompt"].rstrip()
        tid = lambda w: tok.encode(" " + w.strip(), add_special_tokens=False)[0]  # noqa: E731

        _, base_text = greedy_gen(hf, tok, prompt, device)
        base_ok = text_match(base_text, it["answer"])

        ctx_a = SwapHooks(model.layers, lens, W, tid(it["intermediate"]), tid(it["swap_to"]), band)
        _, a_text = greedy_gen(hf, tok, prompt, device, swap_ctx=ctx_a)

        ctx_b = SwapHooks(model.layers, lens, W, tid(it["answer"]), tid(it["swap_answer"]), band)
        _, b_text = greedy_gen(hf, tok, prompt, device, swap_ctx=ctx_b)

        records.append({
            "name": it["name"], "category": it["category"], "baseline_ok": base_ok,
            "base_text": base_text, "a_text": a_text, "b_text": b_text,
            "a_hit": text_match(a_text, it["swap_answer"]),
            "b_hit": text_match(b_text, it["swap_answer"]),
            "a_stayed": text_match(a_text, it["answer"]),
            "b_stayed": text_match(b_text, it["answer"]),
        })

    ok = [r for r in records if r["baseline_ok"]]
    a_rate = sum(r["a_hit"] for r in ok) / len(ok)
    b_rate = sum(r["b_hit"] for r in ok) / len(ok)
    only_a = sum(r["a_hit"] and not r["b_hit"] for r in ok)
    only_b = sum(r["b_hit"] and not r["a_hit"] for r in ok)
    p = mcnemar_exact_p(only_a, only_b)

    print(f"\nbaseline correct: {len(ok)}/{len(records)}")
    print(f"arm A (intermediate swap) hit: {a_rate:.1%}   stayed: {sum(r['a_stayed'] for r in ok)/len(ok):.1%}")
    print(f"arm B (answer-token ctrl) hit: {b_rate:.1%}   stayed: {sum(r['b_stayed'] for r in ok)/len(ok):.1%}")
    print(f"discordant pairs: A-only {only_a}, B-only {only_b};  exact McNemar p = {p:.4f}")

    out = ROOT / "results" / f"e2_qwen{model_key.replace('.','')}.json"
    out.write_text(json.dumps({"model": model_id, "band": [band[0], band[-1]],
                               "records": records,
                               "summary": {"n_ok": len(ok), "a_hit": a_rate, "b_hit": b_rate,
                                            "only_a": only_a, "only_b": only_b, "p": p}}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "4b")
