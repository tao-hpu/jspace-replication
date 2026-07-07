"""Fit a Jacobian lens on the self-trained GPT-2 124M (HF-converted).

Mirrors the official recipe recorded in the gpt2-small lens config.yaml:
Salesforce/wikitext wikitext-103-raw-v1 train split, prompts truncated to
2000 chars / 128 tokens. The official fit early-stopped at 277 prompts;
we fit a fixed prompt budget (default 150; README says ~100 is usable)
with a resumable checkpoint, so the budget can be raised later without
starting over.

Run:   .venv/bin/python experiments/e0-sanity/fit_selftrained.py [n_prompts]
"""

from __future__ import annotations

import pathlib
import sys
import time

import datasets
import torch
import transformers

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "out" / "selftrained-124m-hf"
CKPT = ROOT / "out" / "selftrained_lens_ckpt.pt"
LENS_OUT = ROOT / "out" / "selftrained-124m_jacobian_lens.pt"


def wikitext_prompts(n: int, min_chars: int = 200, max_chars: int = 2000) -> list[str]:
    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    prompts = []
    for row in ds:
        t = row["text"].strip()
        if len(t) >= min_chars:
            prompts.append(t[:max_chars])
            if len(prompts) == n:
                break
    return prompts


def main(n_prompts: int = 150) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    model = jlens.from_hf(hf, tok)

    prompts = wikitext_prompts(n_prompts)
    print(f"device={device}  prompts={len(prompts)}")

    t0 = time.time()
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=32,
        max_seq_len=128,
        checkpoint_path=str(CKPT),
        checkpoint_every=5,
    )
    print(f"fit took {(time.time()-t0)/60:.1f} min")
    lens.save(str(LENS_OUT))
    print(f"saved {LENS_OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
