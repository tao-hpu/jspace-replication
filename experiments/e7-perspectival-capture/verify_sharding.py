"""Verify that a swap applied to a model sharded across GPUs is the same swap.

Under device_map="auto" the decoder blocks are split across cards while the
directions d_a, d_b are built from the unembedding, which lives on one card.
src/interventions.py relocates the directions to whatever device fires the hook.
That relocation is the only thing standing between a correct multi-GPU swap and
a silently wrong one, and "the answer flipped, so it must be fine" is not a
check: a swap that used wrong directions on the second card would still perturb
the model, just not in the way the paper claims.

Two checks, per band layer:

1. FUNCTIONAL (the one that matters): apply the single-layer swap and compare
   the actually-applied edit, D = h_swap - h_clean at the hooked block's output,
   against the analytic edit E = (h_clean . d_a)(d_b - d_a) computed from the
   canonical directions on the unembedding's card. If the hook used wrong
   directions on a far card, D and E disagree macroscopically. Both outputs are
   captured with recorder hooks registered *after* the swap hook, because
   transformers>=5 fills output_hidden_states from hooks registered at
   construction time, which fire first and see the pre-edit value.
   Pass: cos(D, E) >= 0.99 and ||D - E|| / ||E|| <= 0.15 (slack is bf16
   quantization of h, which is not small relative to a few-percent edit).

2. DIRECTION SANITY: the canonical d_a/d_b match a CPU recomputation to 1e-6.
   This catches wrong-layer Jacobians or wrong token rows. It is *not* a
   bit-identity check: CPU and CUDA float32 matmuls order their summations
   differently and legitimately disagree at ~5e-8.

Exits nonzero if any band layer fails either check, so it can gate a run.

Run (on the sharded box):
  HF_HUB_OFFLINE=1 JSPACE_DEVICE_MAP=auto \
  JSPACE_MODEL_DIR=/root/models/Qwen3.6-27B JSPACE_LENS_DIR=/root/jspace-lens \
  python experiments/e7-perspectival-capture/verify_sharding.py 3.6-27b
"""

from __future__ import annotations

import os
import pathlib
import sys

import torch
import transformers

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.interventions import SwapHooks, token_direction  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import BAND_START_FRAC, MODELS  # noqa: E402

from run_e7 import DOMAINS  # noqa: E402


class _Recorder:
    """Post-registered forward hook: sees the block's output after the swap
    hook (registration order = firing order)."""

    def __init__(self, block) -> None:
        self._block = block
        self._handle = None
        self.out: torch.Tensor | None = None

    def __enter__(self):
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            self.out = t.detach().float().clone()

        self._handle = self._block.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()


@torch.no_grad()
def main(model_key: str = "3.6-27b") -> None:
    facts, template, _ = DOMAINS["capitals"]
    model_id, lens_file, _ = MODELS[model_key]
    model_id = os.environ.get("JSPACE_MODEL_DIR", model_id)
    lens_repo = os.environ.get("JSPACE_LENS_DIR", "neuronpedia/jacobian-lens")
    device_map = os.environ.get("JSPACE_DEVICE_MAP", "auto")

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device_map).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(lens_repo, filename=lens_file)
    W = hf.get_output_embeddings().weight

    band = [l for l in lens.source_layers if l >= round(BAND_START_FRAC * model.n_layers)]
    a, _ = facts[0]
    b, _ = facts[1]
    ta = tok.encode(" " + a, add_special_tokens=False)[0]
    tb = tok.encode(" " + b, add_special_tokens=False)[0]
    ids = tok(template.format(a=a), return_tensors="pt").input_ids.to(hf.device)

    dev_of = {l: str(next(model.layers[l].parameters()).device) for l in band}
    devices = sorted(set(dev_of.values()))
    print(f"model={model_id}  n_layers={model.n_layers}  "
          f"band={band[0]}..{band[-1]} ({len(band)} layers)")
    print(f"unembedding W on {W.device}; band layers on: {devices}")
    for dev in devices:
        ls = sorted(l for l in band if dev_of[l] == dev)
        print(f"  {dev}: {len(ls)} band layers ({ls[0]}..{ls[-1]})")
    if len(devices) < 2:
        print("WARNING: the band sits on a single device; the cross-device "
              "path is not exercised and this check is vacuous.")
    print()

    W_cpu = W.detach().to("cpu")

    def ref_cpu(token_id: int, layer: int) -> torch.Tensor:
        u = W_cpu[token_id].float()
        J = lens.jacobians[layer].detach().to("cpu").float()
        d = J.T @ u
        return d / d.norm()

    # one clean forward, recording every band layer's output in a single pass
    recs = {l: _Recorder(model.layers[l]) for l in band}
    for r in recs.values():
        r.__enter__()
    hf(ids)
    for r in recs.values():
        r.__exit__()
    clean = {l: recs[l].out[0] for l in band}  # [seq, d] float32, per layer

    bad = []
    for l in band:
        # canonical directions, exactly as SwapHooks builds them (on W's card)
        d_a = token_direction(lens, W, ta, l)
        d_b = token_direction(lens, W, tb, l)

        # direction sanity vs CPU recomputation (tolerance, not bit-identity)
        ea = float((d_a.cpu() - ref_cpu(ta, l)).abs().max())
        eb = float((d_b.cpu() - ref_cpu(tb, l)).abs().max())
        if max(ea, eb) > 1e-6:
            bad.append((l, f"direction sanity: max dev {max(ea, eb):.2e} > 1e-6"))
            continue

        # functional check: the edit the sharded hook actually applied
        with SwapHooks(model.layers, lens, W, ta, tb, [l]):
            with _Recorder(model.layers[l]) as rec:
                hf(ids)
        h_c = clean[l]
        D = rec.out[0] - h_c                              # applied edit
        dev = h_c.device
        c = h_c @ d_a.to(dev).float()
        E = c.unsqueeze(-1) * (d_b - d_a).to(dev).float()  # intended edit

        cos = float((D * E).sum() / (D.norm() * E.norm()).clamp_min(1e-12))
        rel = float((D - E).norm() / E.norm().clamp_min(1e-12))
        status = "ok" if (cos >= 0.99 and rel <= 0.15) else "FAIL"
        print(f"  L{l:<3d} {dev_of[l]:>7s}  cos(D,E)={cos:.6f}  "
              f"||D-E||/||E||={rel:.4f}  {status}")
        if status == "FAIL":
            bad.append((l, f"functional: cos={cos:.4f} rel={rel:.3f}"))

    print()
    if bad:
        for l, why in bad:
            print(f"  FAIL L{l}: {why}")
        print(f"\nSHARDING CHECK FAILED on {len(bad)}/{len(band)} band layers")
        raise SystemExit(1)
    print(f"SHARDING CHECK PASSED: on all {len(band)} band layers across "
          f"{len(devices)} device(s), the edit actually applied by the sharded "
          f"hook matches the intended edit c*(d_b - d_a) built from the "
          f"unembedding's card (cos >= 0.99, residual <= 15% = bf16 noise), "
          f"and the canonical directions match a CPU recomputation to 1e-6.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3.6-27b")
