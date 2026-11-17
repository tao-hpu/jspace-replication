"""Compare Jacobian-lens fits made on identical weights (W4 of internal review round 2).

The public pythia-1.4b-deduped `step16` branch holds weights elementwise identical
to `step143000` (results/b3_step16_weight_check.json), yet the two lenses fitted
from them give slightly different lens-derived observables. This script compares
the saved lens tensors directly, and optionally a fresh refit of step143000, to
tell apart "same setup, nondeterministic arithmetic" from "different setup"
(prompt count, source layers, d_model are stored in each lens file).

Run (on the machine that holds out/_b2_lenses/, CPU is enough):
  python experiments/e11-development/compare_step16_lenses.py \
      --a out/_b2_lenses/pythia-1.4b-deduped_step143000.pt \
      --b out/_b2_lenses/pythia-1.4b-deduped_step16.pt \
      [--c out/_b2_lenses/pythia-1.4b-deduped_step143000_refit.pt] \
      --out results/b3_step16_lens_check.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import torch


def load(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "J": {int(k): v.float() for k, v in ck["J"].items()},
        "meta": {"n_prompts": ck.get("n_prompts"),
                 "source_layers": list(ck.get("source_layers") or sorted(ck["J"])),
                 "d_model": ck.get("d_model"),
                 "stored_dtype": str(next(iter(ck["J"].values())).dtype)},
    }


def compare(a: dict, b: dict) -> dict:
    layers = sorted(set(a["J"]) & set(b["J"]))
    per_layer = []
    n_equal = n_total = 0
    for l in layers:
        A, B = a["J"][l], b["J"][l]
        d = (A - B)
        eq = int((A == B).sum())
        n_equal += eq
        n_total += A.numel()
        per_layer.append({
            "layer": l,
            "bitwise_identical": bool(eq == A.numel()),
            "frac_elems_equal": eq / A.numel(),
            "max_abs_diff": float(d.abs().max()),
            "rel_frobenius": float(d.norm() / A.norm()),
        })
    rel = [r["rel_frobenius"] for r in per_layer]
    return {
        "meta_a": a["meta"], "meta_b": b["meta"],
        "meta_equal": a["meta"] == b["meta"],
        "n_layers_compared": len(layers),
        "n_layers_bitwise_identical": sum(r["bitwise_identical"] for r in per_layer),
        "frac_elems_equal_overall": n_equal / max(n_total, 1),
        "rel_frobenius_max": max(rel), "rel_frobenius_median": sorted(rel)[len(rel) // 2],
        "per_layer": per_layer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--c", default=None, help="optional fresh refit to compare with both")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lenses = {"a": load(args.a), "b": load(args.b)}
    names = {"a": pathlib.Path(args.a).name, "b": pathlib.Path(args.b).name}
    if args.c:
        lenses["c"] = load(args.c)
        names["c"] = pathlib.Path(args.c).name
    out = {"files": names, "pairs": {}}
    for x, y in itertools.combinations(sorted(lenses), 2):
        out["pairs"][f"{x}_vs_{y}"] = compare(lenses[x], lenses[y])
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    for k, v in out["pairs"].items():
        print(k, "meta_equal", v["meta_equal"], "identical_layers",
              v["n_layers_bitwise_identical"], "/", v["n_layers_compared"],
              "frac_equal %.4f" % v["frac_elems_equal_overall"],
              "relF max %.3e median %.3e" % (v["rel_frobenius_max"], v["rel_frobenius_median"]))


if __name__ == "__main__":
    main()
