"""E7 healing curve: direct measurement of how a single-layer swap write
decays through the downstream stack.

The sliding-window zone maps (2026-07-15) support a two-force model of
writability: early writes die to downstream healing, late writes die to
coordinate drift. So far healing has been a hidden variable inferred from
curve shapes (plateau / monotone decline / interior peak). This measures it
directly: inject the standard swap at one band layer l, then track the
induced residual perturbation Delta h(l') = h_swap(l') - h_clean(l') at
every layer l' >= l, on the prompt forward only (no generation).

Per (write layer l, read layer l'), averaged over items, at the last prompt
position (where the answer generation starts):
    norm      ||Delta h||
    rel       ||Delta h|| / ||h_clean||
    ab        Delta h . normalize(W_U[t_b] - W_U[t_a])  (final answer axis,
              signed; positive = pushes the readout toward the swapped-in b)
    pos_norm  mean over prompt positions of ||Delta h|| (the swap writes at
              all positions; this tracks healing at the written positions
              themselves, not just what reaches the readout slot)

Hook-order note: capture hooks are registered inside the SwapHooks context,
i.e. *after* the swap hook on the written block. Forward hooks chain (each
hook receives the previous hook's modified output), so the capture at the
written layer sees the post-swap value; the transformers v5
output_hidden_states pre-hook trap (see docs/replication-log.md) does not
apply here because we never read output_hidden_states.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python \
        experiments/e7-perspectival-capture/run_e7_healing.py [1.7b|4b|2-2b|...] [capitals]
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

from run_e7 import DOMAINS  # noqa: E402


class CaptureHooks:
    """Store every block's output hidden state (float32, positions x d)."""

    def __init__(self, blocks):
        self._blocks = blocks
        self._handles = []
        self.states: dict[int, torch.Tensor] = {}

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            self.states[idx] = t.detach().float()[0]  # [seq, d]
        return hook

    def __enter__(self):
        for i, blk in enumerate(self._blocks):
            self._handles.append(blk.register_forward_hook(self._make_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


def main(model_key: str = "1.7b", domain: str = "capitals") -> None:
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

    band = [l for l in lens.source_layers if l >= round(BAND_START_FRAC * model.n_layers)]
    n_layers = model.n_layers
    stride = int(os.environ.get("E7_HEAL_STRIDE", "1"))
    write_layers = band[::stride]
    limit = int(os.environ.get("E7_HEAL_ITEMS", "0"))
    if limit:
        facts = facts[:limit]
    print(f"model={model_id} n_layers={n_layers} band={band[0]}..{band[-1]} "
          f"write_layers={len(write_layers)}")

    out = ROOT / "results" / f"e7_healing_{out_tag}{family(hub_id)}{model_key.replace('.', '')}.json"

    # accumulators: sums over items, keyed (write layer, read layer)
    acc = {l: {lp: {"norm": 0.0, "rel": 0.0, "ab": 0.0, "ab_rel": 0.0,
                    "pos_norm": 0.0}
               for lp in range(n_layers)} for l in write_layers}
    # readout-level survival: change in the (logit_b - logit_a) gap at the
    # last prompt position, straight from the lm head (the quantity a flip
    # actually needs), per write layer
    dlogit = {l: 0.0 for l in write_layers}
    n_items = 0

    with torch.no_grad():
        for i, (a, _cap_a) in enumerate(facts):
            b, _cap_b = facts[(i + 1) % len(facts)]
            ta = tok.encode(" " + a, add_special_tokens=False)[0]
            tb = tok.encode(" " + b, add_special_tokens=False)[0]
            u_ab = (W[tb] - W[ta]).float()
            u_ab = u_ab / u_ab.norm()
            ids = tok(template.format(a=a), return_tensors="pt").input_ids.to(device)

            with CaptureHooks(model.layers) as cap:
                logits = hf(ids).logits[0, -1].float()
                clean = dict(cap.states)
            gap_clean = float(logits[tb] - logits[ta])

            for l in write_layers:
                ctx = SwapHooks(model.layers, lens, W, ta, tb, [l])
                with ctx, CaptureHooks(model.layers) as cap:
                    logits = hf(ids).logits[0, -1].float()
                dlogit[l] += float(logits[tb] - logits[ta]) - gap_clean
                for lp in range(l, n_layers):
                    delta = cap.states[lp] - clean[lp]
                    d_last = delta[-1]
                    rec = acc[l][lp]
                    clean_norm = float(clean[lp][-1].norm())
                    rec["norm"] += float(d_last.norm())
                    rec["rel"] += float(d_last.norm()) / clean_norm
                    rec["ab"] += float(d_last @ u_ab.to(d_last.device))
                    rec["ab_rel"] += float(d_last @ u_ab.to(d_last.device)) / clean_norm
                    rec["pos_norm"] += float(delta.norm(dim=-1).mean())
            n_items += 1
            print(f"{a:8s}->{b:8s} done ({n_items}/{len(facts)})", flush=True)

    curves = []
    for l in write_layers:
        rows = []
        for lp in range(l, n_layers):
            rec = acc[l][lp]
            rows.append({"read_layer": lp,
                         **{k: v / n_items for k, v in rec.items()}})
        inj = rows[0]
        final = rows[-1]
        curves.append({
            "write_layer": l,
            "frac_depth": l / n_layers,
            "inj_rel": inj["rel"], "inj_ab_rel": inj["ab_rel"],
            "final_rel": final["rel"], "final_ab_rel": final["ab_rel"],
            "surv_rel": (final["rel"] / inj["rel"]) if inj["rel"] else None,
            "dlogit_ab": dlogit[l] / n_items,
            "curve": rows,
        })

    print(f"\n{'write':>5s} {'frac':>5s} {'inj_rel':>8s} {'fin_rel':>8s} "
          f"{'surv_rel':>8s} {'dlogit_ab':>9s}")
    for c in curves:
        print(f"L{c['write_layer']:<4d} {c['frac_depth']:5.2f} {c['inj_rel']:8.3f} "
              f"{c['final_rel']:8.3f} "
              f"{(c['surv_rel'] if c['surv_rel'] is not None else float('nan')):8.3f} "
              f"{c['dlogit_ab']:9.3f}")

    out.write_text(json.dumps({
        "model": hub_id, "domain": domain,
        "summary": {"n_items": n_items, "n_layers": n_layers,
                    "band": [band[0], band[-1]],
                    "write_layers": write_layers, "curves": curves},
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1.7b",
         sys.argv[2] if len(sys.argv) > 2 else "capitals")
