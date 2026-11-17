"""B3 (third curve): probe-rank of task tokens in the J-lens readout vs step.

The eff_dim and drift curves (run_b3_curves.py) are pure geometry of the fitted
Jacobian. This third curve anchors that geometry to *capability*: at each Pythia
checkpoint, how readable is the task answer -- and the multi-hop intermediate --
from the J-space workspace?

For each probe-swap item we run the model on the prompt and read the J-lens
logits at the last token via lens.apply(...)[0][layer]. Two targets, matching
the workspace claim:

  answer_rank:  rank of the final-answer token in the readout at the DEEPEST
                source layer (closest to output). Low = answer readable.
  inter_rank:   rank of the multi-hop INTERMEDIATE token (e.g. "Brazil" for an
                Amazon->Portuguese item), taken as the BEST (min) rank over all
                source layers -- the intermediate is never the output and may
                surface in any band, so a fixed layer would understate it. This
                is the fair "is the bridge legible in J-space anywhere" test,
                the signature workspace claim. Tracked only on multihop items.

Rank of token t = #{tokens with higher logit}. We report the MEDIAN rank over
items (robust to the long tail) and top-10 hit rate, per checkpoint.

Run:  .venv/bin/python experiments/e11-development/run_b3_probe_rank.py [--model ...]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics

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
DATA = ROOT / "third_party/jacobian-lens/data/experiments/probe-swap.json"


def step_of(name: str) -> int:
    return int(re.match(r"step(\d+)", name).group(1))


def first_token(tok, word: str) -> int | None:
    enc = tok.encode(" " + word, add_special_tokens=False)
    return enc[0] if enc else None


def rank_of(logits: torch.Tensor, tok_id: int) -> int:
    return int((logits > logits[tok_id]).sum())


@torch.no_grad()
def analyze(model_id: str, revision: str, tok, items: list[dict],
            suffix: str = "", foil_shift: int = 0, n_foil: int = 0) -> dict:
    """foil_shift=k>0: score each multihop item against the intermediate of the
    k-th next multihop item (a wrong-but-plausible entity) -- a calibrated null
    for best-over-layers that takes the same max over the same layers."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _src, _rev = _local_or_hub(model_id, revision)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        _src, revision=_rev, dtype=torch.float32).to(dev).eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(
        str(LENS_DIR / f"{model_id.split('/')[-1]}_{revision}{suffix}.pt"))
    multi = [it for it in items if it["category"] == "multihop"]
    foil_of = {id(it): multi[(j + foil_shift) % len(multi)]["intermediate"]
               for j, it in enumerate(multi)}
    layers = lens.source_layers
    deep = layers[-1]  # answer read here; intermediate uses best over all layers

    ans_ranks, inter_ranks = [], []
    foil_ranks = {k: [] for k in range(1, n_foil + 1)}   # one readout, many targets
    for j_it, it in enumerate(items):
        readout, _, _ = lens.apply(model, it["prompt"], positions=[-1])
        a_id = first_token(tok, it["answer"])
        if a_id is not None:
            ans_ranks.append(rank_of(readout[deep][0], a_id))
        if it["category"] == "multihop":
            target = foil_of[id(it)] if foil_shift else it["intermediate"]
            i_id = first_token(tok, target)
            if i_id is not None:  # best (min) rank over all source layers
                inter_ranks.append(min(rank_of(readout[l][0], i_id) for l in layers))
            if n_foil:
                j = multi.index(it)
                for k in range(1, n_foil + 1):
                    f_id = first_token(tok, multi[(j + k) % len(multi)]["intermediate"])
                    if f_id is not None:
                        foil_ranks[k].append(min(rank_of(readout[l][0], f_id) for l in layers))
    del hf, model

    def summ(ranks: list[int]) -> dict:
        if not ranks:
            return {"n": 0}
        return {"n": len(ranks), "median_rank": statistics.median(ranks),
                "mean_log_rank": sum(math.log(r + 1) for r in ranks) / len(ranks),
                "top10_rate": sum(r < 10 for r in ranks) / len(ranks),
                # raw per-item ranks: lets downstream bootstrap CIs over items
                # without re-running the forward passes
                "ranks": ranks}

    out = {"step": step_of(revision), "deep_layer": int(deep),
           "inter_method": "best-over-layers", "lens_suffix": suffix,
           "foil_shift": foil_shift,
           "answer": summ(ans_ranks), "intermediate": summ(inter_ranks)}
    if n_foil:
        pooled = [r for k in foil_ranks for r in foil_ranks[k]]
        out["intermediate_foil"] = {"n": len(pooled), "shifts": n_foil,
            "median_rank": statistics.median(pooled) if pooled else None,
            "top10_rate": (sum(x < 10 for x in pooled) / len(pooled)) if pooled else None,
            "per_shift": {str(k): summ(v) for k, v in foil_ranks.items()}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--suffix", default="", help="lens filename suffix, e.g. _off128")
    ap.add_argument("--revisions", nargs="*", help="only these revisions (default: all fitted)")
    ap.add_argument("--foil", type=int, default=0,
                    help="N>0: also score N foil rotations (calibrated null) and pool them")
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    tok = transformers.AutoTokenizer.from_pretrained(_tok_src(args.model))
    items = json.load(open(DATA))["items"]
    n_multi = sum(it["category"] == "multihop" for it in items)
    print(f"model={args.model} items={len(items)} (multihop={n_multi})")

    revs = sorted((re.match(r"step\d+", p.name[len(tag) + 1:]).group(0)
                   for p in LENS_DIR.glob(f"{tag}_step*{args.suffix}.pt")
                   if p.name.endswith(f"{args.suffix}.pt")
                   and (args.suffix or re.fullmatch(rf"{tag}_step\d+\.pt", p.name))),
                  key=step_of)
    if args.revisions:
        revs = [r for r in revs if r in set(args.revisions)]
    rows = []
    for rev in revs:
        r = analyze(args.model, rev, tok, items, suffix=args.suffix, n_foil=args.foil)
        rows.append(r)
        a, i = r["answer"], r["intermediate"]
        print(f"  step {r['step']:>7d}  answer: med_rank={a['median_rank']:>6.0f} "
              f"top10={a['top10_rate']:.2f}  |  intermediate: med_rank="
              f"{i.get('median_rank', float('nan')):>6.0f} top10={i.get('top10_rate', 0):.2f}"
              + (f"  |  FOIL med={r['intermediate_foil']['median_rank']:.0f} top10={r['intermediate_foil']['top10_rate']:.2f}" if args.foil else ""))

    dst = ROOT / "results" / f"b3_probe_rank_{tag}{args.suffix}{'_foil' if args.foil else ''}.json"
    dst.write_text(json.dumps({"model": args.model, "rows": rows}, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
