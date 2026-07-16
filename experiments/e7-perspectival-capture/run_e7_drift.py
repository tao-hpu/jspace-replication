"""E7 coordinate-drift profile: how logit-like the J-space swap direction is,
per band layer, as a mechanistic account of where capture lives.

The swap intervention (src/interventions.py) edits along
    d_l(t) = normalize(J_l^T @ W_U[t]),
the source-space direction the Jacobian lens maps to token t at layer l. The
pure *output* coordinate of token t is just its unembedding row W_U[t] (the
logit-lens direction: logits = W_U @ h). Define per-layer coordinate drift

    drift_l(t) = cos( J_l^T W_U[t] , W_U[t] ).

High drift => at layer l the swap direction already coincides with t's
unembedding, so a swap there is a logit-space nudge (it moves the output but
need not rewrite internal computation). Low drift => the swap acts in a
distinct, pre-output ("workspace") basis. The paper's coordinate-drift claim
predicts drift rises with depth; the capture story predicts late-band swaps
stop capturing where drift is high. This script tests whether the two
families differ in *where* drift sets in at matched depth (Gemma-2-2B 26
layers vs Qwen3-1.7B 28 layers), which would explain why Gemma's late band
still captures while Qwen's does not.

No forward passes: pure matrix ops on the lens Jacobians and the unembedding.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e7-perspectival-capture/run_e7_drift.py [2-2b|1.7b|...]
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
sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import BAND_START_FRAC, MODELS, family  # noqa: E402
from run_e7 import DOMAINS  # noqa: E402


@torch.no_grad()
def main(model_key: str = "2-2b", domain: str = "capitals") -> None:
    facts, _template, out_tag = DOMAINS[domain]
    model_id, lens_file, _ = MODELS[model_key]
    # JSPACE_MODEL_DIR / JSPACE_LENS_DIR: load from a local snapshot / lens
    # mirror instead of the Hub, for boxes that cannot reach huggingface.co.
    # Unset = the public Hub ids, exactly as before.
    hub_id = model_id
    model_id = os.environ.get("JSPACE_MODEL_DIR", model_id)
    lens_repo = os.environ.get("JSPACE_LENS_DIR", "neuronpedia/jacobian-lens")
    # The math is a handful of d_model x d_model matmuls per layer in float32 on
    # the CPU; only the unembedding and the lens are ever touched. The float32
    # *load* is the trap at 27B scale: 52 GB of bf16 weights become ~110 GB of
    # fp32 and the OOM killer ends the process without a traceback. With
    # JSPACE_DEVICE_MAP set, load bf16 onto the GPUs instead and pull just the
    # unembedding back to CPU float32; the drift numbers are computed on the
    # same CPU/fp32 path either way.
    device_map = os.environ.get("JSPACE_DEVICE_MAP")
    if device_map:
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device_map).eval()
    else:
        hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(lens_repo, filename=lens_file)
    W = hf.get_output_embeddings().weight.detach().float().cpu()  # [vocab, d_model]

    band = [l for l in lens.source_layers if l >= round(BAND_START_FRAC * model.n_layers)]
    n_layers = model.n_layers
    print(f"model={model_id} n_layers={n_layers} band={band[0]}..{band[-1]} vocab={W.shape[0]}")

    # task-relevant tokens: the swap targets (first token of " "+entity) for A
    # and B sides of every item, deduped.
    ent_ids = set()
    for a, _ in facts:
        ent_ids.add(tok.encode(" " + a, add_special_tokens=False)[0])
    ent_ids = sorted(ent_ids)
    Wt = W[ent_ids]                       # [n_ent, d_model]
    Wt_n = Wt / Wt.norm(dim=1, keepdim=True)

    # vocab baseline: a fixed deterministic sample (every k-th row) so the
    # "generic token" drift curve has no RNG dependence.
    step = max(1, W.shape[0] // 4000)
    Wv = W[::step]
    Wv_n = Wv / Wv.norm(dim=1, keepdim=True)

    def drift_for(J: torch.Tensor, Wsub: torch.Tensor, Wsub_n: torch.Tensor) -> float:
        # d(t) = J^T W[t]; cosine with W[t], averaged over tokens t
        D = Wsub @ J                      # [n, d_model]  (rows: W[t]^T J = (J^T W[t])^T)
        D_n = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cos = (D_n * Wsub_n).sum(dim=1)   # per-token cosine
        return float(cos.mean())

    profile = []
    for l in band:
        J = lens.jacobians[l].float()     # [d_model, d_model]
        profile.append({
            "layer": l, "frac_depth": l / n_layers,
            "drift_entity": drift_for(J, Wt, Wt_n),
            "drift_vocab": drift_for(J, Wv, Wv_n),
        })

    print(f"\n{'layer':>5s} {'frac':>5s} {'drift_ent':>9s} {'drift_voc':>9s}")
    for p in profile:
        print(f"{p['layer']:5d} {p['frac_depth']:5.2f} {p['drift_entity']:9.3f} {p['drift_vocab']:9.3f}")

    out = ROOT / "results" / f"e7_drift_{out_tag}{family(hub_id)}{model_key.replace('.', '')}.json"
    out.write_text(json.dumps({
        "model": hub_id, "domain": domain, "n_layers": n_layers, "band": [band[0], band[-1]],
        "n_entity_tokens": len(ent_ids), "vocab_step": step, "profile": profile,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2-2b",
         sys.argv[2] if len(sys.argv) > 2 else "capitals")
