"""B3: read the workspace developmental curves off the per-checkpoint lenses.

Given the lenses fitted in B2 (out/_b2_lenses/{tag}_step{N}.pt), plot two
output-referenced geometry observables against training step. Both are pure
linear algebra on (fitted Jacobian J_l, that checkpoint's own unembedding W_U)
-- no forward passes -- and both are, per the 2026-07-22 prior-art search, the
novel developmental observables (devinterp measures SLT quantities; Tigges
tracks behavioral circuits; neither measures these):

  1. eff_dim(step): participation ratio of the J-transported entity-direction
     cone at a mid layer, d_t = normalize(J_l^T W_U[t]). Low eff_dim = a
     formed low-dimensional workspace. Raw-unembedding cone reported as control
     (is the cone from J, or already narrow in embedding space?).

  2. drift(step, layer): cos(J_l^T W_U[t], W_U[t]) over a vocab sample, per
     layer. Late-rising drift is the coordinate-decoupling signature; B1 found
     pythia's final checkpoint sits at drift~0 everywhere -- this asks whether
     that was true from step 0 or developed.

The heavier probe-rank(step) curve (forward passes on IOI/Eiffel probes per
checkpoint) is a separate script; these two anchor the pipeline end-to-end.

Run:  .venv/bin/python experiments/e11-development/run_b3_curves.py [--model ...]
"""

from __future__ import annotations

import argparse
import json
import pathlib

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


def _pr(mat: torch.Tensor) -> float:
    """Participation-ratio effective dimension of a [n, d] set of rows."""
    s = torch.linalg.svdvals(mat)
    return float((s.square().sum() ** 2) / (s.pow(4).sum()))


def cone_stats(dirs: torch.Tensor) -> dict:
    """dirs: [n, d] unit vectors. PR eff-dim (raw) + an anisotropy-corrected
    eff-dim. Trained LM unembeddings are strongly anisotropic (a large shared
    mean direction; Gao et al. 2019), so the *raw* PR of normalized directions
    collapses toward 1 just from that common component and does NOT measure
    workspace dimensionality. eff_dim_centered removes the shared mean first,
    so it reports how many dimensions the directions actually spread across."""
    G = dirs @ dirs.T
    n = G.shape[0]
    off = G[~torch.eye(n, dtype=torch.bool)]
    centered = dirs - dirs.mean(0, keepdim=True)
    return {"mean_cos": float(off.mean()), "abs_mean_cos": float(off.abs().mean()),
            "eff_dim": _pr(dirs), "eff_dim_centered": _pr(centered), "n_words": n}


def entity_token_ids(tok) -> list[int]:
    entities = sorted({it[k] for it in json.load(open(DATA))["items"]
                       for k in ("intermediate", "swap_to", "answer", "swap_answer")})
    ids = []
    for w in entities:
        enc = tok.encode(" " + w, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    return sorted(set(ids))


def step_of(rev: str) -> int:
    return int(rev.replace("step", "").replace(".pt", ""))


@torch.no_grad()
def analyze(model_id: str, revision: str, tok, entity_ids: list[int], suffix: str = "") -> dict:
    lens = jlens.JacobianLens.from_pretrained(str(LENS_DIR / f"{model_id.split('/')[-1]}_{revision}{suffix}.pt"))
    _src, _rev = _local_or_hub(model_id, revision)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        _src, revision=_rev, dtype=torch.float32)
    W = hf.get_output_embeddings().weight.float()          # [vocab, d]
    vocab, d = W.shape
    n_layers = hf.config.num_hidden_layers
    del hf

    # eff_dim at the mid layer (matches scout_cone_geometry convention)
    layers = lens.source_layers
    mid = layers[len(layers) // 2]
    Jm = lens.jacobians[mid].float()
    U = W[entity_ids]
    raw = cone_stats(torch.nn.functional.normalize(U, dim=1))
    jl = cone_stats(torch.nn.functional.normalize(U @ Jm, dim=1))

    # per-layer drift over a uniform vocab sample
    vocab_ids = list(range(0, vocab, max(1, vocab // 512)))[:512]
    Uv = W[vocab_ids]
    drift = []
    for l in layers:
        J = lens.jacobians[l].float()
        cos = torch.nn.functional.cosine_similarity(Uv @ J, Uv, dim=1)
        drift.append({"layer": int(l), "frac_depth": l / n_layers,
                      "drift_vocab": float(cos.mean())})
    mean_drift = sum(x["drift_vocab"] for x in drift) / len(drift)
    deep = [x["drift_vocab"] for x in drift if x["frac_depth"] >= 0.66]

    return {"step": step_of(revision), "mid_layer": int(mid), "n_layers": n_layers,
            "eff_dim_jl": jl["eff_dim"], "eff_dim_raw": raw["eff_dim"],
            "eff_dim_jl_centered": jl["eff_dim_centered"],
            "eff_dim_raw_centered": raw["eff_dim_centered"],
            "cone_cos_jl": jl["mean_cos"], "n_entities": jl["n_words"],
            "mean_drift": mean_drift,
            "deep_drift": (sum(deep) / len(deep)) if deep else None,
            "drift_profile": drift}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    ap.add_argument("--suffix", default="", help="lens filename suffix, e.g. _off128 (prompt-offset refits)")
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    tok = transformers.AutoTokenizer.from_pretrained(_tok_src(args.model))
    entity_ids = entity_token_ids(tok)

    import re as _re
    revs = sorted((_re.match(r"step\d+", p.name[len(tag) + 1:]).group(0) + ".pt"
                   for p in LENS_DIR.glob(f"{tag}_step*{args.suffix}.pt")
                   if p.name.endswith(f"{args.suffix}.pt")
                   and (args.suffix or _re.fullmatch(rf"{tag}_step\d+\.pt", p.name))),
                  key=step_of)
    print(f"model={args.model} entities={len(entity_ids)} checkpoints={len(revs)}")

    rows = []
    for rev in revs:
        r = analyze(args.model, rev.replace(".pt", ""), tok, entity_ids, suffix=args.suffix)
        rows.append(r)
        print(f"  step {r['step']:>7d}  eff_dim(J)={r['eff_dim_jl']:5.1f} "
              f"(raw {r['eff_dim_raw']:5.1f}) | centered J={r['eff_dim_jl_centered']:5.1f} "
              f"raw={r['eff_dim_raw_centered']:5.1f}  cone_cos={r['cone_cos_jl']:+.3f}  "
              f"mean_drift={r['mean_drift']:+.3f}")

    dst = ROOT / "results" / f"b3_curves_{tag}{args.suffix}.json"
    dst.write_text(json.dumps({"model": args.model, "rows": rows}, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
