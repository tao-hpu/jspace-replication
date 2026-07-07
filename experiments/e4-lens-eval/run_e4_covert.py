"""E4b: covert-content audit — is anything in the lens that is not already
in the mouth?

E3b showed the poetry "plan" evaporates under mouth-exclusion: every lens
hit was also (or soon) in the model's own next-token distribution. This run
generalizes that decomposition to all six official eval sets. For every
intermediate at the designated readout position, alongside the lens rank we
record the rank in the model's *actual output distribution* at the same
position, and classify each lens hit (rank < 10) as:

  shadow  the mouth also ranks it high (mouth rank < 10): the readout
          restates output plausibility;
  covert  the mouth does not (mouth rank >= 10; strict variant >= 100):
          content readable inside that the mouth is not about to say —
          the only kind of hit that supports a workspace distinct from
          the decoder's convergence.

The permutation control from E4 is kept: its covert rate is the false-
positive floor any genuine covert content must clear.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e4-lens-eval/run_e4_covert.py [1.7b|4b]
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch
import transformers

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import MODELS  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e3-poetry"))
from run_e3 import min_rank, newline_position  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e4-lens-eval"))
from run_e4 import CTRL_OFFSET, EVAL_DIR, SETS, token_ids  # noqa: E402


def best_rank(lens_logits, pos_idx: int, ids: list[int]) -> int:
    return min(min_rank(lens_logits, pos_idx, t)[0] for t in ids)


def mouth_rank(m_logits: torch.Tensor, pos_idx: int, ids: list[int]) -> int:
    v = m_logits[pos_idx]
    return min(int((v > v[t]).sum()) for t in ids)


def main(model_key: str = "4b") -> None:
    model_id, lens_file, _ = MODELS[model_key]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens", filename=lens_file)
    print(f"model={model_id} layers={lens.source_layers[0]}..{lens.source_layers[-1]}")

    all_sets = {}
    hdr = f"{'set':14s} {'n_int':>5} {'lens@10':>8} {'shadow':>7} {'covert':>7} {'strict':>7} {'ctrl_cov':>8}"
    print("\n" + hdr)
    for slug in SETS:
        items = json.load(open(EVAL_DIR / f"lens-eval-{slug}.json"))["items"]
        ids_per_item = [[token_ids(tok, slug, w) for w in it["intermediates"]] for it in items]

        rows = []
        for i, it in enumerate(items):
            prompt = it["prompt"].rstrip()
            pos = newline_position(tok, prompt) if slug == "poetry" else -1
            j_logits, m_logits, _ = lens.apply(model, prompt, positions=[pos], use_jacobian=True)
            for ids in ids_per_item[i]:
                rows.append({
                    "item": it["name"], "kind": "target",
                    "j": best_rank(j_logits, 0, ids), "m": mouth_rank(m_logits, 0, ids),
                })
            for ids in ids_per_item[(i + CTRL_OFFSET) % len(items)]:
                rows.append({
                    "item": it["name"], "kind": "control",
                    "j": best_rank(j_logits, 0, ids), "m": mouth_rank(m_logits, 0, ids),
                })

        tgt = [r for r in rows if r["kind"] == "target"]
        ctl = [r for r in rows if r["kind"] == "control"]
        frac = lambda rs, f: sum(f(r) for r in rs) / len(rs)  # noqa: E731
        stats = {
            "n_intermediates": len(tgt),
            "lens10": frac(tgt, lambda r: r["j"] < 10),
            "shadow10": frac(tgt, lambda r: r["j"] < 10 and r["m"] < 10),
            "covert10": frac(tgt, lambda r: r["j"] < 10 and r["m"] >= 10),
            "covert_strict": frac(tgt, lambda r: r["j"] < 10 and r["m"] >= 100),
            "ctrl_covert10": frac(ctl, lambda r: r["j"] < 10 and r["m"] >= 10),
            "ctrl_covert_strict": frac(ctl, lambda r: r["j"] < 10 and r["m"] >= 100),
        }
        all_sets[slug] = {"stats": stats, "rows": rows}
        print(f"{slug:14s} {stats['n_intermediates']:>5} {stats['lens10']:>8.1%} "
              f"{stats['shadow10']:>7.1%} {stats['covert10']:>7.1%} "
              f"{stats['covert_strict']:>7.1%} {stats['ctrl_covert10']:>8.1%}")

    out = ROOT / "results" / f"e4b_covert_qwen{model_key.replace('.', '')}.json"
    out.write_text(json.dumps({"model": model_id, "sets": all_sets}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "4b")
