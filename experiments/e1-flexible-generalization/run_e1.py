"""E1: multi-fact editing (C1) on Qwen3-1.7B with the official prompt set.

Baseline: every (category, func, arg) completion, greedy next token graded
against ``funcs[*].answers[arg]``.
Swap: for every ordered arg pair (A, B) within a category and every func,
swap A's lens direction for B's across the band at all prompt positions,
then grade the greedy next token against B's answer.

Band: source layers from ~28% of depth to the last fitted layer, matching
the fraction used by the Neuronpedia demo on Qwen3.6-27B (18..63 of 64).

Run:  .venv/bin/python experiments/e1-flexible-generalization/run_e1.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import torch
import transformers

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.interventions import SwapHooks  # noqa: E402

MODEL_ID = "Qwen/Qwen3-1.7B"
LENS_FILE = "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"
DATA = ROOT / "third_party/jacobian-lens/data/experiments/flexible-generalization.json"
BAND_START_FRAC = 18 / 64

MODELS = {
    "1.7b": ("Qwen/Qwen3-1.7B", "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt", "e1_qwen17b.json"),
    "4b": ("Qwen/Qwen3-4B", "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt", "e1_qwen4b.json"),
    # second model family (Gemma) for cross-family replication; pre-fitted lens
    # ships in the same neuronpedia/jacobian-lens repo, so no self-fit is needed.
    "2-2b": ("google/gemma-2-2b", "gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt", "e1_gemma2-2b.json"),
}


def family(model_id: str) -> str:
    """Output-filename family prefix, so cross-family results are not mislabeled.
    Qwen keeps the historical ``qwen`` stem; new families get their own."""
    m = model_id.lower()
    if "qwen" in m:
        return "qwen"
    if "gemma" in m:
        return "gemma"
    return m.split("/")[-1]


def first_token_id(tok, word: str) -> int:
    """Token id of the leading-space form of a word (answers/args are graded
    on their first token, per the dataset README's next-token grading)."""
    return tok.encode(" " + word.strip(), add_special_tokens=False)[0]


@torch.no_grad()
def greedy_gen(hf, tok, prompt: str, device: str, n_new: int = 6, swap_ctx=None):
    """Greedy continuation of ``n_new`` tokens.

    If ``swap_ctx`` is given, it is active only for the prompt forward pass;
    continuation steps reuse the (already-swapped) KV cache without further
    intervention — matching the dataset convention of swapping at prompt
    positions and then sampling.

    Returns (first_token_id, text).
    """
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    if swap_ctx is not None:
        with swap_ctx:
            out = hf(ids, use_cache=True)
    else:
        out = hf(ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[0, -1].argmax().reshape(1, 1)
    first = int(nxt.item())
    gen = [first]
    for _ in range(n_new - 1):
        out = hf(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().reshape(1, 1)
        gen.append(int(nxt.item()))
    return first, tok.decode(gen)


def text_match(text: str, answer: str) -> bool:
    """Lenient grading for small models: the stripped continuation starts
    with the answer, ignoring case and leading punctuation/whitespace
    (they often emit a filler token before the content word)."""
    clean = text.strip().lstrip("\"'`.,:;!?-— ").lower()
    return clean.startswith(answer.strip().lower())


def main(model_key: str = "1.7b", band_override: tuple[int, int] | None = None) -> None:
    model_id, lens_file, out_name = MODELS[model_key]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens", filename=lens_file)
    W = hf.get_output_embeddings().weight

    n_layers = model.n_layers
    if band_override is not None:
        lo, hi = band_override
        band = [l for l in lens.source_layers if lo <= l <= hi]
        out_name = out_name.replace(".json", f"_band{lo}-{hi}.json")
    else:
        band = [l for l in lens.source_layers if l >= round(BAND_START_FRAC * n_layers)]
    print(f"device={device} n_layers={n_layers} band={band[0]}..{band[-1]}")

    cats = json.load(open(DATA))["categories"]

    # ---- baseline ----
    baseline = {}  # (cat, func, arg) -> correct?
    records = []
    for cat in cats:
        for fn in cat["funcs"]:
            for arg in cat["args"]:
                prompt = fn["template"].format(arg=arg)
                first, text = greedy_gen(hf, tok, prompt, device)
                strict = first == first_token_id(tok, fn["answers"][arg])
                correct = text_match(text, fn["answers"][arg])
                baseline[(cat["name"], fn["name"], arg)] = correct
                records.append({"kind": "baseline", "cat": cat["name"], "func": fn["name"],
                                "arg": arg, "text": text, "correct": correct, "strict": strict})
    n_ok = sum(r["correct"] for r in records)
    print(f"baseline: {n_ok}/{len(records)} correct")

    # ---- swaps ----
    t0 = time.time()
    swap_records = []
    for cat in cats:
        arg_tok = {a: tok.encode(" " + a, add_special_tokens=False) for a in cat["args"]}
        for a, ids in arg_tok.items():
            if len(ids) != 1:
                print(f"NOTE: arg {a!r} is {len(ids)} tokens; using first")
        for fn in cat["funcs"]:
            for a in cat["args"]:
                for b in cat["args"]:
                    if a == b:
                        continue
                    prompt = fn["template"].format(arg=a)
                    ctx = SwapHooks(model.layers, lens, W, arg_tok[a][0], arg_tok[b][0], band)
                    _, text = greedy_gen(hf, tok, prompt, device, swap_ctx=ctx)
                    swap_records.append({
                        "kind": "swap", "cat": cat["name"], "func": fn["name"],
                        "a": a, "b": b, "text": text,
                        "hit_b": text_match(text, fn["answers"][b]),
                        "stayed_a": text_match(text, fn["answers"][a]),
                        "baseline_ok": baseline[(cat["name"], fn["name"], a)]
                                        and baseline[(cat["name"], fn["name"], b)],
                    })
    print(f"swaps: {len(swap_records)} trials in {time.time()-t0:.0f}s")

    # ---- summary ----
    def rate(rs, key):
        return sum(r[key] for r in rs) / len(rs) if rs else float("nan")

    all_r, ok_r = swap_records, [r for r in swap_records if r["baseline_ok"]]
    print(f"\nswap→new-answer rate: {rate(all_r,'hit_b'):.1%} overall "
          f"({rate(ok_r,'hit_b'):.1%} on baseline-correct pairs, n={len(ok_r)})")
    print(f"stayed-at-old rate:   {rate(all_r,'stayed_a'):.1%} overall "
          f"({rate(ok_r,'stayed_a'):.1%} on baseline-correct pairs)")
    print("\nper category (baseline-correct pairs):")
    for cat in cats:
        rs = [r for r in ok_r if r["cat"] == cat["name"]]
        print(f"  {cat['name']:>12}: hit_b {rate(rs,'hit_b'):.1%}  stayed_a {rate(rs,'stayed_a'):.1%}  (n={len(rs)})")

    out = ROOT / "results" / out_name
    out.write_text(json.dumps({"model": model_id, "band": [band[0], band[-1]],
                               "baseline": records, "swaps": swap_records}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "1.7b"
    band = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else None
    main(key, band)
