"""A5: is the writability gate quantitative (dilution) or structural?

run_e7_profile.py finds that a single-layer swap captures the restatement 0.0%
of the time at *every* band layer of Qwen3-1.7B, while Gemma-2 captures from a
single layer (up to 32.1% at L14 on Gemma-2-2B). run_e7_persistence.py rules out
the obvious mechanism: the Qwen edit does not dissipate. Injected at any band
layer it is still there at the last lens layer, with a least-squares survival
coefficient of 0.72-1.18, matching Gemma's. What differs is the company it
keeps. Qwen amplifies the disturbance about 10x on its way down (Gemma: ~2.5x)
and almost none of that growth points along the swap, so at the readout the swap
is present but diluted: cos(D, E) ~ 0.05 on Qwen against ~0.21 on both Gemmas.

Dilution is a hypothesis, not a finding, and it makes a sharp prediction: if the
swap is merely outnumbered, then turning it up should let it back through. This
script turns it up. Single-layer swap at one band layer, amplitude scaled by

    h' = h + alpha * c * (d_b - d_a),    c = h . d_a

which is exactly SwapHooks at alpha = 1 (so the alpha = 1 column must reproduce
the run_e7_profile number for that layer, and is checked against it).

    capture recovers as alpha rises  =>  the gate is quantitative. The swap is
        readable in principle and is simply outnumbered at unit amplitude.
    capture stays at 0 through alpha = 16, while the answer flips (or the model
        degrades) =>  the gate is structural. Whatever reads the question at
        restatement time does not consult this direction at a single layer, at
        any amplitude, and "cumulative" is doing real work rather than papering
        over a magnitude deficit.

Both outcomes are publishable and they say different things, which is the point.
Gemma-2-2B runs as the positive control: its capture should already be nonzero at
alpha = 1 and should saturate rather than appear.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e7-perspectival-capture/run_e7_gatetest.py [1.7b|2-2b|...] [capitals]
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import torch
import transformers

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.interventions import SwapHooks  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import BAND_START_FRAC, MODELS, family  # noqa: E402

from run_e7 import DOMAINS, staged_gen  # noqa: E402

# 1 is the published single-layer swap; the rest ask how far it has to be turned
# up before the readout notices. 16x is well past the point where the edit norm
# exceeds the ambient residual, so a flat zero across this range is a real answer.
ALPHAS = [1.0, 2.0, 4.0, 8.0, 16.0]


def make_scaled(blocks, lens, W, ta, tb, layers, alpha):
    """SwapHooks with its edit scaled by alpha (alpha=1 == exact SwapHooks).

    The hook applies h - c*d_a + c*d_b with c = h.d_a. Substituting
    d_b -> d_a + alpha*(d_b - d_a) turns that into h + alpha*c*(d_b - d_a):
    the same edit, scaled, with the harvest coefficient c untouched.
    """
    hooks = SwapHooks(blocks, lens, W, ta, tb, layers)
    for l in layers:
        da, db = hooks._dirs[l]
        hooks._dirs[l] = (da, da + alpha * (db - da))
    return hooks


def main(model_key: str = "1.7b", domain: str = "capitals") -> None:
    facts, template, out_tag = DOMAINS[domain]
    model_id, lens_file, _ = MODELS[model_key]
    hub_id = model_id
    model_id = os.environ.get("JSPACE_MODEL_DIR", model_id)
    lens_repo = os.environ.get("JSPACE_LENS_DIR", "neuronpedia/jacobian-lens")
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(lens_repo, filename=lens_file)
    W = hf.get_output_embeddings().weight
    n_answer = int(os.environ.get("E7_NANS", "12"))

    band = [l for l in lens.source_layers if l >= round(BAND_START_FRAC * model.n_layers)]
    # the layer to test. Default: mid-band, which is where Gemma's single-layer
    # capture peaks (L14/26 = 0.54 on 2-2B, L18/42 = 0.43 on 9B) and where Qwen's
    # is, like everywhere else in its band, zero.
    inject = int(os.environ.get("E7_LAYER", str(band[len(band) // 2])))
    if inject not in band:
        raise SystemExit(f"E7_LAYER={inject} is not a band layer ({band[0]}..{band[-1]})")
    read_layers = [band[len(band) // 2]]
    limit = int(os.environ.get("E7_N", str(len(facts))))
    print(f"model={hub_id} n_layers={model.n_layers} band={band[0]}..{band[-1]} "
          f"inject=L{inject} (frac {inject / model.n_layers:.2f}) alphas={ALPHAS}", flush=True)

    arms = {"none": None, **{f"a{al}": al for al in ALPHAS}}
    records = []
    for i, (a, cap_a) in enumerate(facts[:limit]):
        b, cap_b = facts[(i + 1) % len(facts)]
        ta = tok.encode(" " + a, add_special_tokens=False)[0]
        tb = tok.encode(" " + b, add_special_tokens=False)[0]
        prompt = template.format(a=a)
        rec = {"a": a, "b": b, "arms": {}}
        for arm, alpha in arms.items():
            ctx = (None if alpha is None
                   else make_scaled(model.layers, lens, W, ta, tb, [inject], alpha))
            ans, restate, _rep, _ranks, _rs = staged_gen(
                hf, tok, lens, W, prompt, device, (ta, tb), read_layers,
                swap_ctx=ctx, n_answer=n_answer, n_report=1)
            rec["arms"][arm] = {
                "answer": ans, "restate": restate,
                "ans_a": cap_a.lower() in ans.lower(),
                "ans_b": cap_b.lower() in ans.lower(),
                "restate_a": a.lower() in restate.lower(),
                "restate_b": b.lower() in restate.lower(),
            }
        rec["baseline_ok"] = rec["arms"]["none"]["ans_a"]
        records.append(rec)
        row = " ".join(f"a{al:g}:{int(rec['arms'][f'a{al}']['restate_b'])}" for al in ALPHAS)
        print(f"[{i+1}/{limit}] {a:12s}->{b:12s} ok={int(rec['baseline_ok'])} capture: {row}",
              flush=True)

    ok = [r for r in records if r["baseline_ok"]]
    n = len(ok)
    print(f"\nbaseline correct: {n}/{len(records)}")

    def rate(arm, key):
        return sum(r["arms"][arm][key] for r in ok) / n

    summary = {"n": len(records), "n_ok": n, "inject_layer": inject,
               "frac_depth": inject / model.n_layers, "alphas": ALPHAS, "arms": {}}
    print(f"\n{'arm':>6s} {'flip':>7s} {'capture':>8s} {'restate_a':>10s} {'ans_a':>7s}")
    for arm in arms:
        summary["arms"][arm] = {
            "flip": rate(arm, "ans_b"), "capture": rate(arm, "restate_b"),
            "restate_orig": rate(arm, "restate_a"), "ans_orig": rate(arm, "ans_a"),
        }
        s = summary["arms"][arm]
        print(f"{arm:>6s} {s['flip']:7.1%} {s['capture']:8.1%} "
              f"{s['restate_orig']:10.1%} {s['ans_orig']:7.1%}")

    # the verdict this script exists to deliver. Judged on the PEAK over the
    # ladder, not on the largest alpha: the response is non-monotone (overshoot
    # past the target wrecks the state, fluently), so a16 alone would misread a
    # curve that recovers at a2-a8 and dies at a16 as "no recovery".
    base = summary["arms"]["none"]["capture"]
    a1 = summary["arms"]["a1.0"]["capture"]
    peak_arm = max((f"a{al}" for al in ALPHAS), key=lambda k: summary["arms"][k]["capture"])
    peak = summary["arms"][peak_arm]["capture"]
    verdict = (f"QUANTITATIVE: capture recovers with amplitude "
               f"(peak {peak:.1%} at {peak_arm})"
               if peak > max(base, a1) + 0.10 else
               "STRUCTURAL: capture stays at the none-arm floor at every alpha")
    summary["verdict"] = verdict
    print(f"\ncapture: none {base:.1%} | alpha=1 {a1:.1%} | peak {peak:.1%} at {peak_arm}")
    print(f"VERDICT: {verdict}")

    # layer in the filename: one ladder per injection layer, no overwrites
    out = ROOT / "results" / f"e7_gatetest_{out_tag}{family(hub_id)}{model_key.replace('.', '')}_L{inject}.json"
    out.write_text(json.dumps({
        "model": hub_id, "domain": domain, "band": [band[0], band[-1]],
        "n_layers": model.n_layers, "records": records, "summary": summary,
    }, ensure_ascii=False, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1.7b",
         sys.argv[2] if len(sys.argv) > 2 else "capitals")
