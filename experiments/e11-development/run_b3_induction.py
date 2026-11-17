"""B3 follow-up: native induction scores for the Pythia checkpoints we study.

Section 3.2 anchors answer-readability onset (steps 2,000--6,000) to the
induction-head formation window. Olsson et al. (2022) established that window
in a different model family, so the anchor was borrowed; this script measures
it natively on the same public checkpoints the paper analyzes.

Standard prefix-matching score: feed a batch of sequences consisting of a
random token block repeated twice (length T + T). For a query position i in
the second block, the previous occurrence of token s[i] is at i-T, so an
induction head attends to i-T+1. Score of a head = mean attention mass on
that target over all second-block positions; score of a checkpoint = max over
heads (top-3 mean and per-layer max also recorded).

Run (GPU box):
  python experiments/e11-development/run_b3_induction.py --model EleutherAI/pythia-1.4b-deduped
Out:  results/b3_induction_{tag}.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch
import transformers

ROOT = pathlib.Path(__file__).resolve().parents[2]
STEPS = [128, 256, 512, 1000, 2000, 3000, 4000, 6000, 9000, 13000, 23000, 143000]
T = 64          # unique-block length; sequence length is 2T
BATCH = 8
SEED = 0


def _local_root():
    import os
    return os.environ.get("B_LOCAL_ROOT") or os.environ.get("PYTHIA_LOCAL_ROOT")


def _local_or_hub(model_id: str, revision: str):
    """<root>/<revision>/config.json present -> load locally, else hub (see run_b2_fit)."""
    import os
    root = _local_root()
    if root and os.path.exists(os.path.join(root, revision, "config.json")):
        return os.path.join(root, revision), None
    return model_id, revision


def _tok_src(model_id: str) -> str:
    import os
    root = _local_root()
    if root and os.path.exists(os.path.join(root, "tokenizer", "tokenizer_config.json")):
        return os.path.join(root, "tokenizer")
    return model_id


@torch.no_grad()
def induction_scores(model_id: str, revision: str, seqs: torch.Tensor) -> dict:
    _src, _rev = _local_or_hub(model_id, revision)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        _src, revision=_rev, dtype=torch.float32,
        attn_implementation="eager").cuda().eval()
    out = hf(seqs.cuda(), output_attentions=True)
    n_layers = len(out.attentions)
    per_head = []
    for li, att in enumerate(out.attentions):     # att: [B, H, 2T, 2T]
        att = att.float()
        q = torch.arange(T, 2 * T, device=att.device)
        tgt = q - T + 1
        sc = att[:, :, q, tgt].mean(dim=(0, 2))   # [H]
        for h, v in enumerate(sc.tolist()):
            per_head.append({"layer": li, "head": h, "score": v})
    del hf, out
    torch.cuda.empty_cache()
    scores = sorted((x["score"] for x in per_head), reverse=True)
    best = max(per_head, key=lambda x: x["score"])
    layer_max = {}
    for x in per_head:
        layer_max[x["layer"]] = max(layer_max.get(x["layer"], 0.0), x["score"])
    n_heads = len(per_head)
    return {"max_score": scores[0], "top3_mean": sum(scores[:3]) / 3,
            "best_head": {"layer": best["layer"], "head": best["head"]},
            "layer_max": [layer_max[l] for l in range(n_layers)],
            # report the distribution, not just the max
            "n_heads": n_heads,
            "frac_above": {str(t): sum(v > t for v in scores) / n_heads for t in (0.1, 0.2, 0.3, 0.5)},
            "n_above": {str(t): sum(v > t for v in scores) for t in (0.1, 0.2, 0.3, 0.5)},
            "top10_scores": scores[:10],
            "heads": per_head}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--out-suffix", default="_heads", help="output filename suffix (keeps paper file intact)")
    ap.add_argument("--steps", type=int, nargs="*", help="override the checkpoint list (e.g. OLMo has no step<1000)")
    args = ap.parse_args()
    steps = args.steps or STEPS
    tag = args.model.split("/")[-1]

    tok = transformers.AutoTokenizer.from_pretrained(_tok_src(args.model))
    g = torch.Generator().manual_seed(SEED)
    # sample mid-frequency vocab ids, identical across checkpoints and models
    block = torch.randint(1000, min(40000, tok.vocab_size), (BATCH, T), generator=g)
    seqs = torch.cat([block, block], dim=1)       # [B, 2T]

    rows = []
    for step in steps:
        rev = f"step{step}"
        r = {"step": step} | induction_scores(args.model, rev, seqs)
        rows.append(r)
        print(f"  step {step:>7d}  max={r['max_score']:.3f} top3={r['top3_mean']:.3f} "
              f"best=L{r['best_head']['layer']}H{r['best_head']['head']}", flush=True)

    dst = ROOT / "results" / f"b3_induction_{tag}{args.out_suffix}.json"
    dst.write_text(json.dumps({"model": args.model, "T": T, "batch": BATCH,
                               "seed": SEED, "rows": rows}, indent=1))
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
