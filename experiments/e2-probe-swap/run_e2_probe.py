"""E2p: rerun the E2 arms along probe directions instead of lens directions.

E2 (run_e2.py) found the answer-token control (B) significantly beating the
intermediate swap (A), but both arms used Jacobian-lens directions while the
official experiment swaps along *linear-probe* directions. This follow-up
removes that deviation: each entity gets a mass-mean probe direction

    d_X(l) = normalize( mean_l(X) - mean_l(all entities) )

where mean_l(X) is the layer-l residual at the entity's final token, averaged
over neutral template contexts. The mass-mean direction is the standard
closed-form stand-in for a trained linear probe. The swap mechanism, band,
positions, grading, and items are identical to E2, so any A-vs-B gap change
is attributable to the direction source alone.

Reads baseline correctness and the lens-arm outcomes from the E2 results file
(same greedy decoding, so the baseline is unchanged), then reports:

  - A' (probe intermediate swap) vs B' (probe answer swap)  — the E2 question
  - A  (lens) vs A' (probe)                                 — does the probe
    direction rescue the intermediate swap?

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e2-probe-swap/run_e2_probe.py [1.7b|4b]
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
from src.interventions import DirectionSwapHooks  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import MODELS, greedy_gen, text_match  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e2-probe-swap"))
from run_e2 import mcnemar_exact_p  # noqa: E402

DATA = ROOT / "third_party/jacobian-lens/data/experiments/probe-swap.json"

# Neutral contexts with the entity in final position, so the layer-l residual
# at position -1 is the entity representation in context.
TEMPLATES = [
    "I have been reading a lot about {}",
    "The article was mainly about {}",
    "Yesterday we had a long conversation about {}",
    "The quiz question was about {}",
    "Her favorite topic is {}",
    "The documentary is about {}",
    "He wrote an essay on {}",
    "The next chapter covers {}",
    "Everyone was talking about {}",
    "The lecture focused on {}",
    "This page is about {}",
    "The answer is {}",
]


@torch.no_grad()
def entity_means(hf, tok, entities: list[str], band: list[int], device) -> dict[str, torch.Tensor]:
    """{entity: [len(band), d_model] float32 mean residual at final token}."""
    means = {}
    for i, ent in enumerate(entities):
        acc = None
        for tpl in TEMPLATES:
            ids = tok(tpl.format(ent), return_tensors="pt").to(device)
            hs = hf(**ids, output_hidden_states=True).hidden_states
            # hidden_states[l + 1] is the output of block l
            act = torch.stack([hs[l + 1][0, -1].float() for l in band])
            acc = act if acc is None else acc + act
        means[ent] = acc / len(TEMPLATES)
        if (i + 1) % 25 == 0:
            print(f"  probe means: {i + 1}/{len(entities)} entities")
    return means


def probe_dirs(means: dict[str, torch.Tensor], band: list[int]) -> dict[str, dict[int, torch.Tensor]]:
    """Mass-mean directions: subtract the grand mean, normalize per layer."""
    grand = torch.stack(list(means.values())).mean(0)  # [len(band), d]
    out = {}
    for ent, mu in means.items():
        centered = mu - grand
        out[ent] = {l: centered[j] / centered[j].norm() for j, l in enumerate(band)}
    return out


def main(model_key: str = "4b") -> None:
    model_id, _, _ = MODELS[model_key]
    prior_path = ROOT / "results" / f"e2_qwen{model_key.replace('.', '')}.json"
    prior = json.loads(prior_path.read_text())
    band = list(range(prior["band"][0], prior["band"][1] + 1))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)  # exposes .layers; keeps tokenizer flags as in E2
    print(f"model={model_id} band={band[0]}..{band[-1]}")

    items = json.load(open(DATA))["items"]
    lens_by_name = {r["name"]: r for r in prior["records"]}

    entities = sorted({it[k] for it in items for k in ("intermediate", "swap_to", "answer", "swap_answer")})
    print(f"fitting mass-mean directions for {len(entities)} entities x {len(TEMPLATES)} templates")
    dirs = probe_dirs(entity_means(hf, tok, entities, band, device), band)

    records = []
    for it in items:
        prompt = it["prompt"].rstrip()
        pair = lambda a, b: {l: (dirs[a][l], dirs[b][l]) for l in band}  # noqa: E731

        with_a = DirectionSwapHooks(model.layers, pair(it["intermediate"], it["swap_to"]))
        _, a_text = greedy_gen(hf, tok, prompt, device, swap_ctx=with_a)

        with_b = DirectionSwapHooks(model.layers, pair(it["answer"], it["swap_answer"]))
        _, b_text = greedy_gen(hf, tok, prompt, device, swap_ctx=with_b)

        lens_rec = lens_by_name[it["name"]]
        records.append({
            "name": it["name"], "category": it["category"],
            "baseline_ok": lens_rec["baseline_ok"],
            "a_text": a_text, "b_text": b_text,
            "a_hit": text_match(a_text, it["swap_answer"]),
            "b_hit": text_match(b_text, it["swap_answer"]),
            "a_stayed": text_match(a_text, it["answer"]),
            "b_stayed": text_match(b_text, it["answer"]),
            "lens_a_hit": lens_rec["a_hit"], "lens_b_hit": lens_rec["b_hit"],
        })

    ok = [r for r in records if r["baseline_ok"]]
    a_rate = sum(r["a_hit"] for r in ok) / len(ok)
    b_rate = sum(r["b_hit"] for r in ok) / len(ok)
    ab_a = sum(r["a_hit"] and not r["b_hit"] for r in ok)
    ab_b = sum(r["b_hit"] and not r["a_hit"] for r in ok)
    p_ab = mcnemar_exact_p(ab_a, ab_b)
    # paired: did switching A from lens to probe direction change outcomes?
    aa_probe = sum(r["a_hit"] and not r["lens_a_hit"] for r in ok)
    aa_lens = sum(r["lens_a_hit"] and not r["a_hit"] for r in ok)
    p_aa = mcnemar_exact_p(aa_probe, aa_lens)

    print(f"\nbaseline correct (from E2): {len(ok)}/{len(records)}")
    print(f"arm A' (probe intermediate) hit: {a_rate:.1%}   stayed: {sum(r['a_stayed'] for r in ok)/len(ok):.1%}")
    print(f"arm B' (probe answer ctrl)  hit: {b_rate:.1%}   stayed: {sum(r['b_stayed'] for r in ok)/len(ok):.1%}")
    print(f"A' vs B' discordant: A'-only {ab_a}, B'-only {ab_b};  exact McNemar p = {p_ab:.4f}")
    print(f"A lens->probe: gained {aa_probe}, lost {aa_lens};  exact McNemar p = {p_aa:.4f}")
    print(f"(lens arms were: A {sum(r['lens_a_hit'] for r in ok)/len(ok):.1%}, "
          f"B {sum(r['lens_b_hit'] for r in ok)/len(ok):.1%})")

    out = ROOT / "results" / f"e2p_qwen{model_key.replace('.', '')}.json"
    out.write_text(json.dumps({
        "model": model_id, "band": [band[0], band[-1]], "n_templates": len(TEMPLATES),
        "records": records,
        "summary": {"n_ok": len(ok), "a_hit": a_rate, "b_hit": b_rate,
                    "only_a": ab_a, "only_b": ab_b, "p_a_vs_b": p_ab,
                    "a_gain_vs_lens": aa_probe, "a_loss_vs_lens": aa_lens,
                    "p_a_lens_vs_probe": p_aa},
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "4b")
