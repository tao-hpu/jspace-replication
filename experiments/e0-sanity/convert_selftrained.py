"""Convert the self-trained nanoGPT-style GPT-2 124M checkpoint to HF format.

The source model (llm-from-scratch phase1) uses standard GPT-2 module names
but nn.Linear everywhere; HF's GPT2LMHeadModel uses Conv1D, whose weight is
the transpose of nn.Linear's. The vocab is padded to 50304 for training
efficiency and must be sliced back to the tokenizer's 50257.

Run:  .venv/bin/python experiments/e0-sanity/convert_selftrained.py
Sanity: prints wikitext perplexity (expect ~e^3.0≈20 for the val-3.02 ckpt)
and a short greedy sample.
"""

from __future__ import annotations

import os
import pathlib

import torch
import transformers

_ROOT = pathlib.Path(__file__).resolve().parents[2]
# The self-trained checkpoint lives in the sibling llm-from-scratch repo;
# override with SELFTRAINED_CKPT if your layout differs.
CKPT = pathlib.Path(
    os.environ.get(
        "SELFTRAINED_CKPT",
        _ROOT.parent / "llm-from-scratch" / "phase1-124m" / "ckpt10b" / "latest.pt",
    )
)
OUT = _ROOT / "out" / "selftrained-124m-hf"

TRANSPOSE_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)
VOCAB = 50257


def main() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    cfg, sd = ckpt["config"], ckpt["model"]
    # torch.compile / DDP prefixes, if any
    sd = { k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in sd.items() }
    print("source config:", cfg)

    hf_cfg = transformers.GPT2Config(
        vocab_size=VOCAB,
        n_positions=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        activation_function="gelu_new",
    )
    hf = transformers.GPT2LMHeadModel(hf_cfg)

    new_sd = {}
    for k, v in sd.items():
        if k.endswith("transformer.wte.weight") or k == "transformer.wte.weight":
            v = v[:VOCAB]
        if k == "lm_head.weight":
            continue  # tied to wte in HF
        if any(k.endswith(s) for s in TRANSPOSE_SUFFIXES):
            v = v.t().contiguous()
        new_sd[k] = v

    missing, unexpected = hf.load_state_dict(new_sd, strict=False)
    # attn.bias / masked_bias are non-persistent causal-mask buffers; lm_head is tied
    real_missing = [m for m in missing if "attn.bias" not in m and "masked_bias" not in m and m != "lm_head.weight"]
    assert not real_missing, f"missing: {real_missing}"
    assert not unexpected, f"unexpected: {unexpected}"

    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    hf.eval()

    # sanity 1: perplexity on a fixed wikitext-flavoured paragraph
    text = (
        "The city of Paris is the capital of France. It is known for the "
        "Eiffel Tower, the Louvre, and its long history as a centre of "
        "European art, science, and philosophy."
    )
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        loss = hf(ids, labels=ids).loss
    print(f"sanity loss on fixed paragraph: {loss.item():.3f} (ppl {loss.exp().item():.1f})")

    # sanity 2: greedy sample
    prompt_ids = tok("The meaning of life is", return_tensors="pt").input_ids
    sample = hf.generate(prompt_ids, max_new_tokens=25, do_sample=False)
    print("greedy sample:", tok.decode(sample[0]))

    OUT.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"saved HF model to {OUT}")


if __name__ == "__main__":
    main()
