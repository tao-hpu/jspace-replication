"""Tensor-level comparison of the public Pythia-1.4B-deduped step16 branch against step143000.

Paper B (Appendix A) reports that the step16 branch of EleutherAI/pythia-1.4b-deduped yields
observables at endpoint values. This script checks the weights directly: for every tensor in
model.safetensors it reports whether step16 and the reference revision are bitwise identical,
and otherwise the max absolute difference and the relative Frobenius distance. The neighbours
step8 and step32 are compared too, as a control for what a genuine early checkpoint looks like.

CPU only; each revision is ~2.8 GB. Needs HF_HUB_DISABLE_XET=1 if the files are not cached.

Result (2026-11-17, run on cached Hub files): all 292 tensors of step16 are elementwise
identical to step143000; the two files still differ as files (different SHA-256), so only the
tensor contents match. Every tensor differs from step8 and step32.

Run:  HF_HUB_DISABLE_XET=1 python experiments/e11-development/compare_step16_weights.py \
          [--out results/b3_step16_weight_check.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO = "EleutherAI/pythia-1.4b-deduped"


def load(rev: str):
    path = hf_hub_download(REPO, "model.safetensors", revision=rev)
    return path, safe_open(path, framework="pt")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(a, b) -> dict:
    keys_a, keys_b = set(a.keys()), set(b.keys())
    shared = sorted(keys_a & keys_b)
    n_identical, n_total_elems, rows = 0, 0, []
    for k in shared:
        ta, tb = a.get_tensor(k), b.get_tensor(k)
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            rows.append({"tensor": k, "status": "shape_or_dtype_mismatch"})
            continue
        n_total_elems += ta.numel()
        if torch.equal(ta, tb):  # exact equality of every element
            n_identical += 1
            continue
        fa, fb = ta.float(), tb.float()
        diff = (fa - fb)
        rows.append({
            "tensor": k,
            "max_abs_diff": float(diff.abs().max()),
            "rel_frobenius": float(diff.norm() / (fb.norm() + 1e-12)),
            "frac_elems_equal": float((ta == tb).float().mean()),
        })
    return {
        "n_tensors_shared": len(shared),
        "only_in_a": sorted(keys_a - keys_b),
        "only_in_b": sorted(keys_b - keys_a),
        "n_tensors_bitwise_identical": n_identical,
        "n_params_compared": n_total_elems,
        "differing_tensors": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/b3_step16_weight_check.json")
    args = ap.parse_args()
    revs = ["step16", "step143000", "step8", "step32"]
    files = {r: load(r) for r in revs}
    out = {"repo": REPO, "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET")}
    out["file_sha256"] = {r: file_sha256(p) for r, (p, _) in files.items()}
    out["dtypes"] = {r: str(f.get_tensor(next(iter(f.keys()))).dtype) for r, (_, f) in files.items()}
    for a, b in [("step16", "step143000"), ("step16", "step8"), ("step16", "step32"),
                 ("step8", "step143000")]:
        res = compare(files[a][1], files[b][1])
        diffs = res["differing_tensors"]
        res["summary"] = {
            "max_rel_frobenius": max((d.get("rel_frobenius", 0.0) for d in diffs), default=0.0),
            "median_rel_frobenius": float(torch.tensor(
                [d.get("rel_frobenius", 0.0) for d in diffs] or [0.0]).median()),
        }
        out[f"{a}_vs_{b}"] = res
        print(a, "vs", b, "identical tensors:", res["n_tensors_bitwise_identical"], "/",
              res["n_tensors_shared"], res["summary"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("written:", args.out)


if __name__ == "__main__":
    main()
