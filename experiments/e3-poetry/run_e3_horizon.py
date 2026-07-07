"""E3b: when does the rhyme word become lens-readable? (planning horizon)

E3 read only the official position (end of line 1) and found nothing. The
obvious rejoinder: maybe the "plan" emerges later, somewhere inside line 2.
If the rhyme word only becomes readable at the token immediately before it
is emitted, the observable picture is next-word prediction, not planning.

Per item, read the J-lens (and logit lens) at *every* line-2 prompt
position, from the end-of-line-1 newline to the final prompt token. Align
positions by distance-to-go d (d=1 is the last token before the rhyme word)
and aggregate pass@10 and median rank per d, with the same fixed-offset
permutation control as E3.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e3-poetry/run_e3_horizon.py [1.7b|4b]
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

DATA = ROOT / "third_party/jacobian-lens/data/evaluations/lens-eval-poetry.json"
MAX_D = 12  # deepest distance-to-go reported


def main(model_key: str = "4b") -> None:
    model_id, lens_file, _ = MODELS[model_key]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens", filename=lens_file)
    print(f"model={model_id} layers={lens.source_layers[0]}..{lens.source_layers[-1]}")

    items = json.load(open(DATA))["items"]
    targets = [tok.encode(" " + it["intermediates"][0].strip(), add_special_tokens=False)[0]
               for it in items]

    records = []
    for i, it in enumerate(items):
        prompt = it["prompt"].rstrip()
        n_tok = len(tok(prompt).input_ids)
        nl = newline_position(tok, prompt)
        positions = list(range(nl, n_tok))  # newline .. final prompt token
        tid, ctrl_tid = targets[i], targets[(i + 7) % len(items)]

        j_logits, m_logits, _ = lens.apply(model, prompt, positions=positions, use_jacobian=True)
        l_logits, _, _ = lens.apply(model, prompt, positions=positions, use_jacobian=False)

        by_d = []
        for idx, pos in enumerate(positions):
            d = n_tok - pos  # 1 = last token before the rhyme word
            if d > MAX_D:
                continue
            v = m_logits[idx]
            by_d.append({
                "d": d,
                "j_rank": min_rank(j_logits, idx, tid)[0],
                "ctrl_rank": min_rank(j_logits, idx, ctrl_tid)[0],
                "l_rank": min_rank(l_logits, idx, tid)[0],
                # rank in the model's actual next-token distribution here:
                # a lens hit with a high m_rank is genuine anticipation, a
                # lens hit with a low m_rank is mere local plausibility.
                "m_rank": int((v > v[tid]).sum()),
            })
        records.append({"name": it["name"], "target": it["intermediates"][0], "by_d": by_d})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(items)}")

    print(f"\n{'d':>3} {'n':>4}  {'J pass@10':>10} {'ctrl':>7} {'logit':>7} {'model':>7} {'antic':>7}   {'J median':>9}")
    summary = {}
    for d in range(1, MAX_D + 1):
        rows = [x for r in records for x in r["by_d"] if x["d"] == d]
        if not rows:
            continue
        stats = {
            "n": len(rows),
            "j_pass10": sum(x["j_rank"] < 10 for x in rows) / len(rows),
            "ctrl_pass10": sum(x["ctrl_rank"] < 10 for x in rows) / len(rows),
            "l_pass10": sum(x["l_rank"] < 10 for x in rows) / len(rows),
            "m_pass10": sum(x["m_rank"] < 10 for x in rows) / len(rows),
            # anticipation signature: lens sees it, the mouth does not
            "j_anticip10": sum(x["j_rank"] < 10 and x["m_rank"] >= 10 for x in rows) / len(rows),
            "j_median": sorted(x["j_rank"] for x in rows)[len(rows) // 2],
        }
        summary[d] = stats
        print(f"{d:>3} {stats['n']:>4}  {stats['j_pass10']:>10.1%} {stats['ctrl_pass10']:>7.1%} "
              f"{stats['l_pass10']:>7.1%} {stats['m_pass10']:>7.1%} {stats['j_anticip10']:>7.1%}   "
              f"{stats['j_median']:>9}")

    out = ROOT / "results" / f"e3b_horizon_qwen{model_key.replace('.', '')}.json"
    out.write_text(json.dumps({"model": model_id, "records": records,
                               "summary": {str(k): v for k, v in summary.items()}}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "4b")
