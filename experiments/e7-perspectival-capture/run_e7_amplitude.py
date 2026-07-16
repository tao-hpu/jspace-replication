"""A2: per-layer harvest amplitude of the swap edit, generation-free.

The swap at layer l applies h' = h - (h.d_a) d_a + (h.d_a) d_b, so the size
of the edit it can make is set by two purely geometric quantities measured on
the *clean* forward pass:

  amp   = |h . d_a|            the coefficient the swap harvests at layer l
  gap   = ||d_b - d_a||        how far the harvested coefficient is moved
                               (= sqrt(2 - 2 cos(d_a, d_b)) for unit dirs)

edit_norm = amp * gap is the exact norm of the per-position edit SwapHooks
would have applied; rel_edit = edit_norm / ||h|| is its size relative to the
ambient residual. No hooks, no generation: one prompt forward per item with
output_hidden_states=True, statistics over prompt positions.

Together with the drift axis (run_e7_drift.py: cos(J_l^T W_U[t], W_U[t]))
this gives the second factor of the capture ~ f(drift, amplitude) model; the
single-layer capture curve to regress against is run_e7_profile.py output.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e7-perspectival-capture/run_e7_amplitude.py [1.7b|4b|8b|14b|2-2b|2-9b] [capitals]
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
from src.interventions import token_direction  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import BAND_START_FRAC, MODELS, family  # noqa: E402

# side effect: registers the 8b/14b/2-9b/2-27b ladder entries in MODELS
from run_e7 import DOMAINS  # noqa: E402


@torch.no_grad()
def main(model_key: str = "1.7b", domain: str = "capitals") -> None:
    facts, template, out_tag = DOMAINS[domain]
    model_id, lens_file, _ = MODELS[model_key]
    # JSPACE_MODEL_DIR / JSPACE_LENS_DIR / JSPACE_DEVICE_MAP: same opt-in knobs
    # as run_e7 (local snapshot, local lens mirror, shard across cards for the
    # 27B). All unset = the previous single-device Hub path, unchanged.
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

    band = [l for l in lens.source_layers if l >= round(BAND_START_FRAC * model.n_layers)]
    n_layers = model.n_layers
    print(f"model={model_id} n_layers={n_layers} band={band[0]}..{band[-1]} ({len(band)} layers)")

    records = []
    for i, (a, _cap_a) in enumerate(facts):
        b, _cap_b = facts[(i + 1) % len(facts)]
        ta = tok.encode(" " + a, add_special_tokens=False)[0]
        tb = tok.encode(" " + b, add_special_tokens=False)[0]
        prompt = template.format(a=a)
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        out = hf(ids, output_hidden_states=True)
        rec = {"a": a, "b": b, "layers": {}}
        for l in band:
            # directions are built on the unembedding's card; under a sharded
            # model the layer's residual lives on another one. Move the
            # direction (a no-op when the devices already agree).
            h = out.hidden_states[l + 1][0].float()  # [seq, d], block-l output
            d_a = token_direction(lens, W, ta, l).to(h.device)
            d_b = token_direction(lens, W, tb, l).to(h.device)
            coeff = h @ d_a
            h_norm = h.norm(dim=-1)
            gap = float((d_b - d_a).norm())
            rec["layers"][str(l)] = {
                "amp_mean": float(coeff.abs().mean()),
                "amp_last": float(coeff[-1].abs()),
                "h_norm_mean": float(h_norm.mean()),
                "h_norm_last": float(h_norm[-1]),
                "gap": gap,
                "cos_ab": float(d_a @ d_b),
                "rel_edit_mean": float((coeff.abs() * gap / h_norm).mean()),
            }
        records.append(rec)
        mid = str(band[len(band) // 2])
        print(f"{a:12s}->{b:12s} amp@L{mid}={rec['layers'][mid]['amp_mean']:.2f} "
              f"gap={rec['layers'][mid]['gap']:.3f} "
              f"rel={rec['layers'][mid]['rel_edit_mean']:.4f}", flush=True)

    n = len(records)
    profile = []
    for l in band:
        key = str(l)
        mean = lambda f: sum(f(r["layers"][key]) for r in records) / n  # noqa: E731
        profile.append({
            "layer": l, "frac_depth": l / n_layers,
            "amp_mean": mean(lambda x: x["amp_mean"]),
            "amp_last": mean(lambda x: x["amp_last"]),
            "gap": mean(lambda x: x["gap"]),
            "cos_ab": mean(lambda x: x["cos_ab"]),
            "edit_norm_mean": mean(lambda x: x["amp_mean"] * x["gap"]),
            "rel_edit_mean": mean(lambda x: x["rel_edit_mean"]),
            "h_norm_mean": mean(lambda x: x["h_norm_mean"]),
        })

    print(f"\n{'layer':>5s} {'frac':>5s} {'amp':>8s} {'gap':>6s} {'edit':>8s} {'rel':>7s}")
    for p in profile:
        print(f"{p['layer']:5d} {p['frac_depth']:5.2f} {p['amp_mean']:8.2f} "
              f"{p['gap']:6.3f} {p['edit_norm_mean']:8.2f} {p['rel_edit_mean']:7.4f}")

    out_path = ROOT / "results" / f"e7_amplitude_{out_tag}{family(hub_id)}{model_key.replace('.', '')}.json"
    out_path.write_text(json.dumps({
        "model": hub_id, "domain": domain, "band": [band[0], band[-1]],
        "n_layers": n_layers, "records": records, "profile": profile,
    }, ensure_ascii=False, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1.7b",
         sys.argv[2] if len(sys.argv) > 2 else "capitals")
