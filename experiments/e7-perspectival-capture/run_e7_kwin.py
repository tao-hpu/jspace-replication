"""E7 minimum write width: capture as a function of *how many* contiguous
band layers the swap spans.

Motivation (2026-07-15 single-layer audit): on the flip metric the
single-layer gate is binary - open on Gemma-2-2B/9B and Qwen3.5-9B, closed on
every dense Qwen3 and on Gemma-2-27B - yet every closed model still captures
at the half-band dose (dense Qwen3-1.7B: half1 restate 66.7% vs single-layer
0%). So "closed" cannot mean unwritable; it must mean a single layer's write
does not survive the healing of the layers downstream. That makes the gate a
continuous quantity: the minimum contiguous window width k at which capture
appears. This script measures it.

Arms: none and full-band anchors, plus early-anchored windows band[:k] for
k in 2, 3, 4, 6, 8, 12 (k=1 is already measured by run_e7_profile.py; the
half-band dose is the run_e7 half1 arm), plus one late-anchored window of
width 6 starting at the band midpoint as a position control (the drift law
predicts widening a high-drift window buys nothing).

Same machinery as run_e7_profile (shared KV cache, hook active on the prompt
forward only, substring grading). Read flips first: the restate grade carries
the incidental-mention floor documented in the replication log.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python \
        experiments/e7-perspectival-capture/run_e7_kwin.py [1.7b|2-2b|...] [capitals]
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
from src.interventions import DirectionSwapHooks, SwapHooks  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import BAND_START_FRAC, MODELS, family  # noqa: E402

from run_e7 import DOMAINS, staged_gen  # noqa: E402

EARLY_K = (2, 3, 4, 6, 8, 12)
LATE_K = (6,)


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
    n_answer = int(os.environ.get("E7_NANS", "12"))

    band = [l for l in lens.source_layers if l >= round(BAND_START_FRAC * model.n_layers)]
    n_layers = model.n_layers
    read_layers = [band[len(band) // 2]]  # unused for grading; staged_gen needs a target layer
    print(f"model={model_id} n_layers={n_layers} band={band[0]}..{band[-1]} ({len(band)} layers)")

    # E7_DIR=rand replaces the lens swap directions in every window (and the
    # full-band anchor) with seeded random unit directions via the same
    # amplitude-harvesting transfer as run_e7's randdir arm: the null that
    # width-k coordination is not direction-generic. E7_SEED as in run_e7.
    dir_mode = os.environ.get("E7_DIR", "lens")
    seed = int(os.environ.get("E7_SEED", "1"))
    rand_dirs = None
    if dir_mode == "rand":
        gen = torch.Generator().manual_seed(seed)
        d_model = hf.config.hidden_size
        rand_dirs = {}
        for l in band:
            ra = torch.randn(d_model, generator=gen)
            rb = torch.randn(d_model, generator=gen)
            rand_dirs[l] = ((ra / ra.norm()).to(hf.device), (rb / rb.norm()).to(hf.device))

    stem = (f"e7_slide{os.environ['E7_SLIDE_K']}" if os.environ.get("E7_SLIDE_K")
            else "e7_kwin")
    if dir_mode == "rand":
        stem += "_randdir" + ("" if seed == 1 else f"{seed}")
    out = ROOT / "results" / f"{stem}_{out_tag}{family(hub_id)}{model_key.replace('.', '')}.json"
    ckpt = out.with_suffix(".partial.json")

    # arms: anchors, then either the width sweep (default) or, with
    # E7_SLIDE_K set, a fixed-width window slid across the band (stride
    # E7_SLIDE_STRIDE, default 2) - the zone map that de-confounds width
    # from position (see the 2026-07-15 Qwen3-4B log entry).
    arms: dict[str, object] = {"none": None, "full": band}
    slide_k = int(os.environ.get("E7_SLIDE_K", "0"))
    if slide_k:
        stride = int(os.environ.get("E7_SLIDE_STRIDE", "2"))
        for i in range(0, len(band) - slide_k + 1, stride):
            arms[f"s{band[i]}"] = band[i:i + slide_k]
    else:
        mid = len(band) // 2
        for k in EARLY_K:
            if k <= len(band):
                arms[f"e{k}"] = band[:k]
        for k in LATE_K:
            if mid + k <= len(band):
                arms[f"l{k}"] = band[mid:mid + k]
    print("arms:", {name: (f"L{ls[0]}..L{ls[-1]}" if ls else "-")
                    for name, ls in arms.items()})

    records = []
    for i, (a, cap_a) in enumerate(facts):
        b, cap_b = facts[(i + 1) % len(facts)]
        ta = tok.encode(" " + a, add_special_tokens=False)[0]
        tb = tok.encode(" " + b, add_special_tokens=False)[0]
        prompt = template.format(a=a)
        rec = {"a": a, "b": b, "arms": {}}
        for arm, layers in arms.items():
            if layers is None:
                ctx = None
            elif rand_dirs is not None:
                ctx = DirectionSwapHooks(model.layers, {l: rand_dirs[l] for l in layers})
            else:
                ctx = SwapHooks(model.layers, lens, W, ta, tb, layers)
            ans, restate, _report, _ranks, _rl = staged_gen(
                hf, tok, lens, W, prompt, device, (ta, tb), read_layers,
                swap_ctx=ctx, n_answer=n_answer, n_report=1)
            rec["arms"][arm] = {
                "ans_b": cap_b.lower() in ans.lower(),
                "ans_a": cap_a.lower() in ans.lower(),
                "restate_b": b.lower() in restate.lower(),
                "restate_a": a.lower() in restate.lower(),
            }
        rec["baseline_ok"] = rec["arms"]["none"]["ans_a"]
        records.append(rec)
        ckpt.write_text(json.dumps({
            "model": hub_id, "domain": domain, "partial": True,
            "records": records,
        }, ensure_ascii=False))
        wnames = [w for w in arms if w not in ("none", "full")]
        print(f"{a:8s}->{b:8s} ok={int(rec['baseline_ok'])} "
              f"full_flip={int(rec['arms']['full']['ans_b'])} "
              f"{wnames[0]}_flip={int(rec['arms'][wnames[0]]['ans_b'])} "
              f"{wnames[-1]}_flip={int(rec['arms'][wnames[-1]]['ans_b'])}")
        okc = [r for r in records if r["baseline_ok"]]
        if len(okc) and len(okc) % 14 == 0:
            row = " ".join(
                f"{name}:{sum(r['arms'][name]['ans_b'] for r in okc)/len(okc)*100:.0f}"
                for name in arms if name != "none")
            print(f"  [kwin @ {len(okc)} ok] flip% by window: {row}", flush=True)

    ok = [r for r in records if r["baseline_ok"]]
    n = len(ok)
    print(f"\nbaseline correct: {n}/{len(records)}")

    def rate(arm, key):
        return sum(r["arms"][arm][key] for r in ok) / n

    windows = []
    for name, layers in arms.items():
        if name == "none":
            continue
        windows.append({
            "arm": name,
            "layers": [layers[0], layers[-1]],
            "k": len(layers),
            "anchor": ("full" if name == "full"
                       else "slide" if name.startswith("s")
                       else "early" if name.startswith("e") else "late"),
            "flip": rate(name, "ans_b"),
            "restate_swapped": rate(name, "restate_b"),
        })
    summary = {
        "n": len(records), "n_ok": n, "n_layers": n_layers,
        "band": [band[0], band[-1]],
        "none": {"flip": rate("none", "ans_b"), "restate_swapped": rate("none", "restate_b")},
        "windows": windows,
    }

    print(f"\n{'arm':>6s} {'k':>3s} {'layers':>9s} {'flip':>6s} {'restate':>8s}")
    for w in windows:
        print(f"{w['arm']:>6s} {w['k']:3d} L{w['layers'][0]:>2d}..L{w['layers'][1]:<3d}"
              f" {w['flip']:6.1%} {w['restate_swapped']:8.1%}")

    out.write_text(json.dumps({
        "model": hub_id, "domain": domain, "direction": dir_mode,
        **({"seed": seed} if dir_mode == "rand" else {}),
        "records": records, "summary": summary,
    }, ensure_ascii=False, indent=2))
    ckpt.unlink(missing_ok=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1.7b",
         sys.argv[2] if len(sys.argv) > 2 else "capitals")
