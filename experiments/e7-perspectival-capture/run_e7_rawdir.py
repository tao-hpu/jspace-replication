"""E7 direction-source control: repeat the single-layer profile with the
swap direction taken straight from the unembedding rows (d = normalize(
W_U[token])) instead of the lens transport (d = normalize(J_l^T W_U[token])).

Why: within open models the single-layer flip rate tracks coordinate drift
(Spearman -0.77..-0.83), but drift admits two readings. (1) Coordinate
rotation: at high-drift layers the lens direction is the *wrong basis* for
an intervention aimed at the readout, so writing there fails. (2) Timing
proxy: high-drift layers sit downstream of the entity->answer computation,
so *any* write is too late regardless of basis. This control separates
them: the raw W_U direction ignores the per-layer basis entirely. If flips
with the raw direction decay across depth the same way as with the lens
direction, drift is a clock; if the lens direction holds up where the raw
one dies (or vice versa at low drift), the basis itself is doing work.

Same machinery as run_e7_profile (staged generation, substring grading,
flip = ans_b is the floor-free metric). DirectionSwapHooks already accepts
arbitrary unit directions, so src/ is untouched.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python \
        experiments/e7-perspectival-capture/run_e7_rawdir.py [2-2b|1.7b|...] [capitals]
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
from src.interventions import DirectionSwapHooks  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import BAND_START_FRAC, MODELS, family  # noqa: E402

from run_e7 import DOMAINS, staged_gen  # noqa: E402


def raw_dirs(W: torch.Tensor, ta: int, tb: int, layers) -> dict:
    da = W[ta].float()
    db = W[tb].float()
    da, db = da / da.norm(), db / db.norm()
    return {l: (da, db) for l in layers}


def main(model_key: str = "2-2b", domain: str = "capitals") -> None:
    facts, template, out_tag = DOMAINS[domain]
    model_id, lens_file, _ = MODELS[model_key]
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
    read_layers = [band[len(band) // 2]]
    print(f"model={model_id} n_layers={n_layers} band={band[0]}..{band[-1]} ({len(band)} layers)")

    out = ROOT / "results" / f"e7_rawdir_{out_tag}{family(hub_id)}{model_key.replace('.', '')}.json"
    ckpt = out.with_suffix(".partial.json")

    # E7_RAWDIR_ANCHORS=1 runs only none/full: on closed-gate models the
    # per-layer arms are uninformative (lens flips are already zero), and
    # the full-band arm alone answers the basis question at dose.
    arms: dict[str, object] = {"none": None, "full": band}
    if not int(os.environ.get("E7_RAWDIR_ANCHORS", "0")):
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
            ctx = (None if layers is None
                   else DirectionSwapHooks(model.layers, raw_dirs(W, ta, tb, layers)))
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
        ckpt.write_text(json.dumps({
            "model": hub_id, "domain": domain, "partial": True,
            "records": records,
        }, ensure_ascii=False))
        print(f"{a:8s}->{b:8s} ok={int(rec['baseline_ok'])} "
              f"full_flip={int(rec['arms']['full']['ans_b'])}")
        okc = [r for r in records if r["baseline_ok"]]
        if len(okc) and len(okc) % 14 == 0:
            row = " ".join(f"{name}:{sum(r['arms'][name]['ans_b'] for r in okc)/len(okc)*100:.0f}"
                           for name in arms if name != "none")
            print(f"  [rawdir @ {len(okc)} ok] flip% by arm: {row}", flush=True)

    ok = [r for r in records if r["baseline_ok"]]
    n = len(ok)
    print(f"\nbaseline correct: {n}/{len(records)}")

    def rate(arm, key):
        return sum(r["arms"][arm][key] for r in ok) / n

    profile = []
    for l in band:
        arm = f"L{l}"
        if arm not in arms:
            continue
        profile.append({
            "layer": l, "frac_depth": l / n_layers,
            "flip": rate(arm, "ans_b"), "restate_swapped": rate(arm, "restate_b"),
        })
    summary = {
        "n": len(records), "n_ok": n, "n_layers": n_layers,
        "band": [band[0], band[-1]],
        "direction": "raw_wu",
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
