"""E0 pipeline sanity: apply a pre-fitted Jacobian lens to GPT-2 124M.

For each probe prompt we track one *target token* — a concept the network
should hold internally before (or while) producing its answer — and report
its rank at the final position, per layer, under both transports:

  - J-lens   (use_jacobian=True):  unembed(J_l @ h)
  - logit lens (use_jacobian=False): unembed(h)

Success criterion (qualitative, walkthrough-level): the J-lens surfaces the
target at readable ranks in mid layers where the logit lens does not.

Run:  .venv/bin/python experiments/e0-sanity/run_e0.py
"""

from __future__ import annotations

import json
import pathlib

import torch
import transformers

import jlens

HF_LENS_REPO = "neuronpedia/jacobian-lens"
RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results"

# (name, prompt, target_token, why)
PROBES = [
    (
        "eiffel-paris",
        "The Eiffel Tower is located in the city of",
        " Paris",
        "direct fact completion; target should appear early and strengthen",
    ),
    (
        "boot-italy",
        "Fact: The currency used in the country shaped like a boot is",
        " Italy",
        "latent intermediate: 'Italy' never appears in the prompt (upstream README example)",
    ),
    (
        "ioi-mary",
        "When Mary and John went to the store, John gave a drink to",
        " Mary",
        "indirect-object identification; GPT-2 small is known to resolve this",
    ),
]


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    """1-indexed rank of token_id in a [vocab] logits vector."""
    return int((logits > logits[token_id]).sum().item()) + 1


def main(
    model_id: str = "gpt2",
    lens_file: str = "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt",
    out_name: str = "e0_gpt2.json",
) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"model={model_id}  device={device}")

    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)

    lens = jlens.JacobianLens.from_pretrained(HF_LENS_REPO, filename=lens_file)
    print(f"lens: layers={lens.source_layers}  d_model={lens.d_model}  n_prompts={lens.n_prompts}")

    report = {"model": model_id, "device": device, "lens_file": lens_file, "probes": []}
    for name, prompt, target, why in PROBES:
        # from_hf(force_bos=True) flips add_bos_token on the tokenizer, so
        # strip special tokens when encoding the bare target word.
        target_ids = tok.encode(target, add_special_tokens=False)
        assert len(target_ids) == 1, f"{target!r} is not a single token"
        tid = target_ids[0]

        jl, model_logits, _ = lens.apply(model, prompt, positions=[-1])
        ll, _, _ = lens.apply(model, prompt, positions=[-1], use_jacobian=False)

        rows = []
        for layer in sorted(jl):
            j_rank = token_rank(jl[layer][0], tid)
            l_rank = token_rank(ll[layer][0], tid)
            j_top = [tok.decode([t]) for t in jl[layer][0].topk(5).indices]
            rows.append({"layer": layer, "j_rank": j_rank, "logit_rank": l_rank, "j_top5": j_top})

        out_rank = token_rank(model_logits[0], tid)
        report["probes"].append(
            {"name": name, "prompt": prompt, "target": target, "why": why,
             "model_output_rank": out_rank, "layers": rows}
        )

        print(f"\n=== {name}  target={target!r}  (model output rank: {out_rank}) ===")
        print(f"{'layer':>5} {'J-rank':>8} {'logit-rank':>11}   J-lens top-5")
        for r in rows:
            print(f"{r['layer']:>5} {r['j_rank']:>8} {r['logit_rank']:>11}   {r['j_top5']}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / out_name
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main()
