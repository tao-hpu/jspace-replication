"""B2: fit one Jacobian lens per Pythia training checkpoint.

Paper B asks *when* and *how fast* the low-dimensional, output-decoupled
J-space workspace forms during training. The Pythia suite exposes ~143 public
intermediate checkpoints per model as HF branches (revision="step{N}"), so we
can fit a lens at each step and read the developmental curves off the fitted
Jacobians (B3) with no extra training of our own.

Recipe mirrors the official gpt2/pythia lens fits (Salesforce/wikitext
wikitext-103-raw-v1 train split, prompts truncated to 2000 chars / 128 tokens);
each fit is resumable and each checkpoint's lens is skipped if already on disk,
so the full 154-step sweep on the 4090 can be grown incrementally.

Pilot (local MPS/CPU, pythia-70m is d=512/6L, minutes per checkpoint):
  .venv/bin/python experiments/e11-development/run_b2_fit.py
Full sweep (4090, September):
  .venv/bin/python experiments/e11-development/run_b2_fit.py --model EleutherAI/pythia-160m-deduped --full --n 466
"""

from __future__ import annotations

import argparse
import sys
import pathlib
import time

import datasets
import torch
import transformers

import jlens


def _local_root() -> str | None:
    """B_LOCAL_ROOT (any family) or the older PYTHIA_LOCAL_ROOT: a directory holding one
    <revision>/ folder per checkpoint (config.json + weights), plus an optional tokenizer/."""
    import os
    return os.environ.get("B_LOCAL_ROOT") or os.environ.get("PYTHIA_LOCAL_ROOT")


def _local_or_hub(model_id: str, revision: str) -> tuple[str, str | None]:
    """If <root>/<revision>/config.json exists, load from it and ignore the hub (the 4090's
    egress stalls on multi-GB hub downloads). Else hub as before. Local folders are named
    step{N} even when the hub branch carries a suffix (OLMo: step{N}-tokens{X}B)."""
    import os
    root = _local_root()
    if root and os.path.exists(os.path.join(root, revision, "config.json")):
        return os.path.join(root, revision), None
    return model_id, revision


def _tok_src(model_id: str) -> str:
    """Tokenizer is revision-invariant: <root>/tokenizer/ if staged locally, else the hub."""
    import os
    root = _local_root()
    if root and os.path.exists(os.path.join(root, "tokenizer", "tokenizer_config.json")):
        return os.path.join(root, "tokenizer")
    return model_id

ROOT = pathlib.Path(__file__).resolve().parents[2]
LENS_DIR = ROOT / "out" / "_b2_lenses"

# Log-spaced pilot: spans the phase change (induction heads ~2-4k steps) and
# the long tail. The full sweep swaps this for the whole 154-revision list.
PILOT_REVISIONS = ["step0", "step512", "step2000", "step8000", "step32000", "step143000"]

# The production developmental curve: log2 grid through the induction-head phase
# change (Olsson 2022; ~1k-4k steps), dense enough to resolve a transition, then
# coarse through the long tail. ~24 checkpoints -- the figure-grade set. Hot
# regions can be densified afterward by passing --revisions explicitly.
SWEEP_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000,
               6000, 9000, 13000, 23000, 43000, 73000, 103000, 143000]
SWEEP_REVISIONS = [f"step{s}" for s in SWEEP_STEPS]


def full_revisions() -> list[str]:
    """The 154 public Pythia checkpoints: log2 up to 512, then every 1000."""
    revs = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    revs += list(range(1000, 143000 + 1, 1000))
    return [f"step{s}" for s in revs]


def wikitext_prompts(n: int, min_chars: int = 200, max_chars: int = 2000,
                     offset: int = 0) -> list[str]:
    """First n qualifying prompts after skipping `offset` qualifying ones
    (offset=0 reproduces the paper's fits; 128/256 give disjoint samples)."""
    ds = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    prompts, skipped = [], 0
    for row in ds:
        t = row["text"].strip()
        if len(t) >= min_chars:
            if skipped < offset:
                skipped += 1
                continue
            prompts.append(t[:max_chars])
            if len(prompts) == n:
                break
    return prompts


def fit_one(model_id: str, revision: str, prompts: list[str], device: str,
            dim_batch: int = 32, suffix: str = "") -> None:
    tag = model_id.split("/")[-1]
    lens_out = LENS_DIR / f"{tag}_{revision}{suffix}.pt"
    if lens_out.exists():
        print(f"{tag} {revision}: cached, skipping")
        return
    _src, _rev = _local_or_hub(model_id, revision)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        _src, revision=_rev, dtype=torch.float32).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(_tok_src(model_id))  # revision-invariant
    model = jlens.from_hf(hf, tok)

    t0 = time.time()
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=dim_batch,  # column batching only; result is dim_batch-invariant
        max_seq_len=128,
        checkpoint_path=str(LENS_DIR / f".{tag}_{revision}{suffix}_ckpt.pt"),
        checkpoint_every=25,
    )
    lens.save(str(lens_out))
    print(f"{tag} {revision}: fit {len(prompts)} prompts in "
          f"{(time.time() - t0) / 60:.1f} min -> {lens_out.name}")
    del hf, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--n", type=int, default=128, help="prompt budget per fit")
    ap.add_argument("--sweep", action="store_true", help="~24-checkpoint figure set")
    ap.add_argument("--full", action="store_true", help="all 154 checkpoints")
    ap.add_argument("--revisions", nargs="*", help="explicit revision list override")
    ap.add_argument("--dim-batch", type=int, default=32,
                    help="Jacobian column batch; larger = faster on big GPUs")
    ap.add_argument("--prompt-offset", type=int, default=0,
                    help="skip this many qualifying WikiText prompts first (disjoint fit samples)")
    ap.add_argument("--suffix", default="",
                    help="appended to lens filename, e.g. _off128 (keeps paper fits intact)")
    args = ap.parse_args()

    LENS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    revisions = args.revisions or (
        full_revisions() if args.full else
        SWEEP_REVISIONS if args.sweep else PILOT_REVISIONS)
    prompts = wikitext_prompts(args.n, offset=args.prompt_offset)
    print(f"model={args.model} device={device} n_prompts={len(prompts)} "
          f"checkpoints={len(revisions)}")

    failed = []
    for rev in revisions:
        try:
            fit_one(args.model, rev, prompts, device, dim_batch=args.dim_batch,
                    suffix=args.suffix)
        except Exception as e:  # a bad revision must not kill the sweep
            print(f"FAILED {rev}: {e}")
            failed.append(rev)
    if failed:  # non-zero exit lets callers retry (e.g. smaller --dim-batch after OOM)
        sys.exit(1)


if __name__ == "__main__":
    main()
