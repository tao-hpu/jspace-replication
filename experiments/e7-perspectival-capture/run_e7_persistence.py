"""A4: does a single-layer swap edit *persist* downstream, or is it healed away?

The writability gate (run_e7_profile.py) is measured with a 0/1 substring
verdict: on the Qwen ladder a single-layer swap captures the restatement 0.0%
of the time, at every band layer, while Gemma-2 captures from a single layer.
run_e7_amplitude.py already rules out the trivial reading of that zero -- the
Qwen band layers harvest a *comparable* relative edit magnitude, so it is not
that there is nothing to grab. But a 0/1 outcome cannot distinguish

  (i) the edit is applied and then dissipates before it reaches the readout,
      from
  (ii) the edit survives but never crosses the threshold the substring grader
       happens to use.

Those are different mechanisms and the paper's "irreducibly cumulative" wording
commits to (i). This script measures it directly, with a continuous readout and
no generation at all.

For each item and each injection layer l in the band, run two prompt forwards,
one clean and one with SwapHooks active at the single layer [l], and compare
the residual streams at every downstream layer l' >= l. Write

  D(l') = H_swap[l'] - H_clean[l']        the surviving perturbation, [seq, d]
  E(l') = c (x) ( d_b(l') - d_a(l') )     the swap edit expressed in layer l'
                                          coordinates, [seq, d], where
                                          c = H_clean[l] . d_a(l) is the
                                          coefficient actually harvested at the
                                          injection layer

and report three sequence-level (Frobenius) quantities:

  rel_pert(l')  = ||D||_F / ||H_clean[l']||_F
      how large the perturbation still is relative to the ambient residual.
      Decaying to ~0 => the network healed the edit (mechanism i).

  cos_align(l') = <D, E(l')> / (||D||_F ||E(l')||_F)
      does what survives still point along the swap's own semantics. Bounded in
      [-1, 1]. Falling while rel_pert holds => the perturbation persists but has
      rotated out of the swap direction.

  survive(l')   = <D, E(l')> / ||E(l')||_F^2
      the least-squares coefficient of the injected edit that is still present.
      Exactly 1.0 at l' = l by construction, so it doubles as a self-check.

Sequence-level rather than per-position, deliberately: the swap harvests almost
nothing at the final prompt position (``A:`` carries little coefficient along an
entity direction), so a per-position normalisation divides by ~0 there. The
edit lands at the entity position and reaches the answer position through
attention, which is exactly what rel_pert_last then tracks.

CAUTION, and the reason for ``_PostEditRecorder`` below: transformers>=5
collects ``output_hidden_states`` with its own forward hooks, registered at
model construction and therefore *before* any hook we add. PyTorch fires
forward hooks in registration order and an earlier hook sees the original
output, so the recorded hidden state at a hooked layer is the value from
*before* our intervention. Downstream layers are unaffected (they are not
hooked, so what is recorded there is the true propagated result), which means
the corruption is confined to exactly the injection layer, i.e. exactly the
self-check. We therefore read the injection layer through a recorder hook
registered after SwapHooks, so it fires last and sees the edited output; the
``survive`` self-check at l' = l must then come out at 1.0. No other script in
this repo is exposed to the trap: run_e7 only generates, and run_e7_amplitude
reads hidden states on hook-free forwards.

Predictions the gate story makes, and that this script can falsify:
  Qwen  (gate closed): rel_pert decays toward zero within a few layers of l.
  Gemma (gate open):   rel_pert survives to the final layer, cos_align stays > 0.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e7-perspectival-capture/run_e7_persistence.py [1.7b|4b|2-2b|...] [capitals]
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
from src.interventions import SwapHooks, token_direction  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import BAND_START_FRAC, MODELS, family  # noqa: E402

# side effect: registers the 8b/14b/2-9b/27b ladder entries in MODELS
from run_e7 import DOMAINS  # noqa: E402


class _PostEditRecorder:
    """Capture a block's output *after* the swap hook has rewritten it.

    Must be entered inside the SwapHooks context so that it registers second
    and therefore fires second (see the module docstring).
    """

    def __init__(self, block, ) -> None:
        self._block = block
        self._handle = None
        self.output: torch.Tensor | None = None

    def __enter__(self) -> "_PostEditRecorder":
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            self.output = t.detach().float().clone()

        self._handle = self._block.register_forward_hook(hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@torch.no_grad()
def main(model_key: str = "1.7b", domain: str = "capitals") -> None:
    facts, template, out_tag = DOMAINS[domain]
    model_id, lens_file, _ = MODELS[model_key]
    # same opt-in knobs as the rest of the E7 family; all unset = Hub, one device
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
    n_items = int(os.environ.get("E7_N", "0")) or len(facts)
    print(f"model={hub_id} n_layers={n_layers} band={band[0]}..{band[-1]} "
          f"({len(band)} layers) items={n_items}", flush=True)

    # Downstream readouts are taken at every lens source layer at or after the
    # injection layer. Using lens layers (not all layers) keeps d_a/d_b defined
    # by the same Jacobians the swap itself uses; layers between them have no
    # lens coordinate to project onto.
    read_at = {l: [lp for lp in lens.source_layers if lp >= l] for l in band}

    records = []
    for i, (a, _cap_a) in enumerate(facts[:n_items]):
        b, _cap_b = facts[(i + 1) % len(facts)]
        ta = tok.encode(" " + a, add_special_tokens=False)[0]
        tb = tok.encode(" " + b, add_special_tokens=False)[0]
        prompt = template.format(a=a)
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)

        clean = hf(ids, output_hidden_states=True).hidden_states
        rec = {"a": a, "b": b, "inject": {}}

        for l in band:
            # the recorder registers inside the SwapHooks context, so it fires
            # after the swap and sees the edited output of block l
            with SwapHooks(model.layers, lens, W, ta, tb, [l]):
                with _PostEditRecorder(model.layers[l]) as rec_l:
                    swapped = hf(ids, output_hidden_states=True).hidden_states
            post_edit = rec_l.output[0]                   # [seq, d], layer l

            # c: the coefficient the swap actually harvested at the injection
            # layer, per position. Signed, and near zero at the final prompt
            # position (see the module docstring).
            h_l = clean[l + 1][0].float()                 # [seq, d]
            c = h_l @ token_direction(lens, W, ta, l).to(h_l.device)  # [seq]

            curve = []
            for lp in read_at[l]:
                hc = clean[lp + 1][0].float()
                # hidden_states at the hooked layer is the pre-edit value; use
                # the recorder's tensor there instead
                hs = post_edit.to(hc.device) if lp == l else swapped[lp + 1][0].float()
                D = hs - hc                               # [seq, d]
                d_a_p = token_direction(lens, W, ta, lp).to(hc.device)
                d_b_p = token_direction(lens, W, tb, lp).to(hc.device)
                # the swap edit as it would look in layer lp's coordinates
                E = c.unsqueeze(-1) * (d_b_p - d_a_p)     # [seq, d]

                dn = D.norm().clamp_min(1e-12)
                en = E.norm().clamp_min(1e-12)
                dot = (D * E).sum()
                curve.append({
                    "layer": lp,
                    "rel_pert": float(D.norm() / hc.norm().clamp_min(1e-12)),
                    "rel_pert_last": float(
                        D[-1].norm() / hc[-1].norm().clamp_min(1e-12)),
                    "cos_align": float(dot / (dn * en)),
                    "survive": float(dot / en.pow(2)),
                })
            rec["inject"][str(l)] = {
                "c_abs_mean": float(c.abs().mean()),
                "c_abs_last": float(c[-1].abs()),
                "curve": curve,
            }

        records.append(rec)
        # progress line: inject at the first band layer, read at the last one
        first = rec["inject"][str(band[0])]["curve"]
        print(f"[{i+1}/{n_items}] {a:12s}->{b:12s} inject@L{band[0]}: "
              f"rel {first[0]['rel_pert']:.4f} -> {first[-1]['rel_pert']:.4f}  "
              f"survive {first[0]['survive']:.2f} -> {first[-1]['survive']:+.2f}  "
              f"cos {first[-1]['cos_align']:+.2f}", flush=True)

    n = len(records)

    # summary: for each injection layer, the survival curve averaged over items,
    # plus the scalars the paper quotes directly.
    profile = []
    for l in band:
        key = str(l)
        curve = []
        for j, lp in enumerate(read_at[l]):
            m = lambda f: sum(f(r["inject"][key]["curve"][j]) for r in records) / n  # noqa: E731
            curve.append({
                "layer": lp, "frac_depth": lp / n_layers,
                "rel_pert": m(lambda x: x["rel_pert"]),
                "rel_pert_last": m(lambda x: x["rel_pert_last"]),
                "cos_align": m(lambda x: x["cos_align"]),
                "survive": m(lambda x: x["survive"]),
            })
        head, tail = curve[0], curve[-1]
        profile.append({
            "inject_layer": l, "frac_depth": l / n_layers,
            # self-check: survive at the injection layer must be 1.0
            "survive_at_inject": head["survive"],
            "rel_pert_at_inject": head["rel_pert"],
            "rel_pert_at_final": tail["rel_pert"],
            # magnitude survival: how much of the perturbation is left at the
            # last lens layer, relative to the moment it was injected
            "mag_survival": tail["rel_pert"] / max(head["rel_pert"], 1e-12),
            "survive_at_final": tail["survive"],
            "cos_at_final": tail["cos_align"],
            "curve": curve,
        })

    # hard self-check: at the injection layer the surviving perturbation *is*
    # the injected edit, so survive must be 1.0. Anything else means the
    # readout is not seeing the post-edit state (see the module docstring) and
    # every downstream number would be silently mis-scaled.
    worst = max(abs(p["survive_at_inject"] - 1.0) for p in profile)
    if worst > 0.02:
        raise RuntimeError(
            f"self-check failed: survive at the injection layer deviates from "
            f"1.0 by up to {worst:.3f}; the injection-layer readout is not the "
            f"post-edit state. Refusing to write results.")
    print(f"\nself-check OK: survive@inject = 1.0 to within {worst:.4f}")

    print(f"\n{'inj':>4s} {'frac':>5s} {'chk':>5s} {'rel@inj':>8s} {'rel@end':>8s} "
          f"{'magsurv':>8s} {'surv@end':>9s} {'cos@end':>8s}")
    for p in profile:
        print(f"{p['inject_layer']:4d} {p['frac_depth']:5.2f} "
              f"{p['survive_at_inject']:5.2f} {p['rel_pert_at_inject']:8.4f} "
              f"{p['rel_pert_at_final']:8.4f} {p['mag_survival']:8.2f} "
              f"{p['survive_at_final']:+9.2f} {p['cos_at_final']:+8.2f}")

    out = ROOT / "results" / f"e7_persistence_{out_tag}{family(hub_id)}{model_key.replace('.', '')}.json"
    out.write_text(json.dumps({
        "model": hub_id, "domain": domain, "n_layers": n_layers,
        "band": [band[0], band[-1]], "lens_layers": list(lens.source_layers),
        "n": n, "records": records, "profile": profile,
    }, ensure_ascii=False, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1.7b",
         sys.argv[2] if len(sys.argv) > 2 else "capitals")
