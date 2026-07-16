"""E7 dose profile: capture as a function of *where* in the band the swap
lands, at single-layer resolution.

The headline E7 run bisects the band into half1/half2 and finds, on the Qwen
ladder, that the early half captures the restatement while the late half does
not -- but the cross-family Gemma-2-2B check has the late half capturing about
as well as the full band (restate_swapped 80.4% vs 83.9%). half1/half2 is too
coarse to say *where* the two families diverge. This sweeps a single-layer
swap across every band layer and records flip + restate_swapped per layer, so
the capture-vs-depth curve can be read directly and overlaid across models at
matched depth (Gemma-2-2B, 26 layers, vs Qwen3-1.7B, 28 layers).

Same machinery as run_e7 (shared KV cache, hook active on the prompt forward
only, substring grading); only the arm set changes: instead of none/full/
half1/half2/randdir it is one single-layer swap arm per band layer, plus none
and full for anchoring.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e7-perspectival-capture/run_e7_profile.py [2-2b|1.7b|...] [capitals]
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

# same scale-ladder / cross-family registrations as run_e7
from run_e7 import DOMAINS, staged_gen  # noqa: E402


def main(model_key: str = "2-2b", domain: str = "capitals") -> None:
    facts, template, out_tag = DOMAINS[domain]
    model_id, lens_file, _ = MODELS[model_key]
    # JSPACE_MODEL_DIR / JSPACE_LENS_DIR / JSPACE_DEVICE_MAP: same opt-in knobs
    # as run_e7. All unset = the previous single-device Hub path, unchanged.
    hub_id = model_id
    model_id = os.environ.get("JSPACE_MODEL_DIR", model_id)
    lens_repo = os.environ.get("JSPACE_LENS_DIR", "neuronpedia/jacobian-lens")
    device_map = os.environ.get("JSPACE_DEVICE_MAP")
    if device_map:
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device_map).eval()
        device = hf.device
    else:
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
    n_layers = model.n_layers
    read_layers = [band[len(band) // 2]]  # unused for grading; staged_gen needs a target layer
    print(f"model={model_id} n_layers={n_layers} band={band[0]}..{band[-1]} ({len(band)} layers)")

    out = ROOT / "results" / f"e7_profile_{out_tag}{family(hub_id)}{model_key.replace('.', '')}.json"
    ckpt = out.with_suffix(".partial.json")

    # arms: none, full (anchors), then one single-layer swap per band layer
    arms: dict[str, object] = {"none": None, "full": band}
    for l in band:
        arms[f"L{l}"] = [l]

    records = []
    for i, (a, cap_a) in enumerate(facts):
        b, cap_b = facts[(i + 1) % len(facts)]
        ta = tok.encode(" " + a, add_special_tokens=False)[0]
        tb = tok.encode(" " + b, add_special_tokens=False)[0]
        prompt = template.format(a=a)
        rec = {"a": a, "b": b, "arms": {}}
        for arm, layers in arms.items():
            ctx = None if layers is None else SwapHooks(model.layers, lens, W, ta, tb, layers)
            ans, restate, _report, _ranks, _rl = staged_gen(
                hf, tok, lens, W, prompt, device, (ta, tb), read_layers,
                swap_ctx=ctx, n_answer=n_answer, n_report=1)
            rec["arms"][arm] = {
                "ans_b": cap_b.lower() in ans.lower(),
                "ans_a": cap_a.lower() in ans.lower(),
                "restate_b": b.lower() in restate.lower(),
                "restate_a": a.lower() in restate.lower(),
            }
        rec["baseline_ok"] = rec["arms"]["none"]["ans_a"]
        records.append(rec)
        # checkpoint after every item: a multi-hour sharded 27B sweep must not
        # lose everything to a crash in the last hour
        ckpt.write_text(json.dumps({
            "model": hub_id, "domain": domain, "partial": True,
            "records": records,
        }, ensure_ascii=False))
        print(f"{a:8s}->{b:8s} ok={int(rec['baseline_ok'])} "
              f"full_flip={int(rec['arms']['full']['ans_b'])} "
              f"full_rst={int(rec['arms']['full']['restate_b'])}")
        # running per-layer restate profile every 14 baseline-ok items, so the
        # single-layer capture curve can be eyeballed long before all 56 finish
        okc = [r for r in records if r["baseline_ok"]]
        if len(okc) and len(okc) % 14 == 0:
            row = " ".join(f"L{l}:{sum(r['arms'][f'L{l}']['restate_b'] for r in okc)/len(okc)*100:.0f}"
                           for l in band)
            print(f"  [profile @ {len(okc)} ok] restate% by layer: {row}", flush=True)

    ok = [r for r in records if r["baseline_ok"]]
    n = len(ok)
    print(f"\nbaseline correct: {n}/{len(records)}")

    def rate(arm, key):
        return sum(r["arms"][arm][key] for r in ok) / n

    # per-layer profile, ordered by depth; fractional depth = layer / n_layers
    profile = []
    for l in band:
        arm = f"L{l}"
        profile.append({
            "layer": l, "frac_depth": l / n_layers,
            "flip": rate(arm, "ans_b"), "restate_swapped": rate(arm, "restate_b"),
        })
    summary = {
        "n": len(records), "n_ok": n, "n_layers": n_layers,
        "band": [band[0], band[-1]],
        "full": {"flip": rate("full", "ans_b"), "restate_swapped": rate("full", "restate_b")},
        "none": {"flip": rate("none", "ans_b"), "restate_swapped": rate("none", "restate_b")},
        "profile": profile,
    }

    print(f"\n{'layer':>5s} {'frac':>5s} {'flip':>6s} {'restate':>8s}")
    for p in profile:
        print(f"{p['layer']:5d} {p['frac_depth']:5.2f} {p['flip']:6.1%} {p['restate_swapped']:8.1%}")

    out.write_text(json.dumps({
        "model": hub_id, "domain": domain, "records": records, "summary": summary,
    }, ensure_ascii=False, indent=2))
    ckpt.unlink(missing_ok=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2-2b",
         sys.argv[2] if len(sys.argv) > 2 else "capitals")
