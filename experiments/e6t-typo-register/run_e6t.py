"""E6t: causal test of the covert typo register (second register instance).

E4b found the intended form of a corrupted word survives mouth-exclusion
(16-37% covert) — the other register family besides language identity. E6
showed the language register is causally load-bearing. This experiment asks
the same question for the typo register, with the same recipe:

- Register axis from content-matched (clean, corrupted) sentence pairs,
  topics and typo words disjoint from the test items (asserted at runtime).
  Because the pair shares a prefix, activations are identical before the
  typo (causal attention), so residuals are averaged from the first
  diverging token position onward: d = normalize(mu_typo - mu_clean),
  gap = |mu_typo - mu_clean| per band layer.
- Test items: the official lens-eval-typo set (96 items; prompt ends with
  the corrupted word, `intermediates[0]` is the intended form).
- Interventions are gap translations (h' = h + alpha * gap * d), with an
  amplitude-matched random-direction control, exactly as in E6.

Two readouts per item:

  correction (generative, on corrupted prompts): append an elicitation
    suffix and grade whether the intended word appears (fixed) or the
    corrupted surface form is echoed. Arms: baseline / erase@alpha
    (shift toward clean) / random@alpha. If the register carries the
    intended form, erasing it should impair correction even though the
    corrupted surface is still in context. Compared by exact McNemar on
    "baseline corrected but arm did not".

  report (single forward, both prompts): a yes/no spelling-mistake probe;
    the yes-no logit margin under baseline / shift / random, erase on
    corrupted prompts and inject (toward typo) on clean prompts. Same
    greedy-margin caveat as E7's self-report.

covert flag: intended word's first token outside the mouth top-100 at the
end of the bare corrupted prompt (E4b strict criterion, unchanged).

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e6t-typo-register/run_e6t.py [1.7b|4b] [alpha ...]
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import torch
import transformers

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.interventions import DirectionShiftHooks  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import BAND_START_FRAC, MODELS, family, greedy_gen, text_match  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e2-probe-swap"))
from run_e2 import mcnemar_exact_p  # noqa: E402

COVERT_RANK = 100
EVAL_SET = ROOT / "third_party" / "jacobian-lens" / "data" / "evaluations" / "lens-eval-typo.json"

# One-shot demo lifts 1.7B baseline correction from 4/8 to 7/8 (format pilot,
# baseline-only). The demo word "blue" is asserted disjoint from eval targets.
# The demo's own typo sits in context during interventions; the shift applies
# to it too, identically across erase/random arms.
CORRECTION_SHOT = ('The sky above the mountains was bright blu.\n'
                   'The last word is misspelled. Spelled correctly, it is "blue".\n\n')
CORRECTION_SUFFIX = '.\nThe last word is misspelled. Spelled correctly, it is "'
REPORT_SUFFIX = ".\nQuestion: Does the sentence above contain a spelling mistake? Answer:"

# (clean, corrupted) estimation pairs. Typos sit mid-sentence so the
# register state has positions to live in after the corrupted word; the
# intended words are asserted disjoint from the eval set's targets.
TYPO_PAIRS = [
    ("He will definitely come to the party tonight", "He will definately come to the party tonight"),
    ("You will receive the package early next week", "You will recieve the package early next week"),
    ("Please separate the recycling from the trash", "Please seperate the recycling from the trash"),
    ("We are leaving tomorrow morning for the coast", "We are leaving tommorow morning for the coast"),
    ("I believe the story he told us yesterday", "I beleive the story he told us yesterday"),
    ("My best friend moved to another city in spring", "My best freind moved to another city in spring"),
    ("That movie was really weird but a lot of fun", "That movie was really wierd but a lot of fun"),
    ("The bakery on the corner sells delicious cinnamon rolls", "The bakery on the corner sells delicius cinnamon rolls"),
    ("Mark the date on your calendar right away", "Mark the date on your calender right away"),
    ("It is necessary to bring your own water bottle", "It is neccessary to bring your own water bottle"),
    ("The accident occurred near the old stone bridge", "The accident occured near the old stone bridge"),
    ("Do not embarrass your sister in front of the guests", "Do not embarass your sister in front of the guests"),
    ("They will probably arrive right after lunch", "They will probaly arrive right after lunch"),
    ("The beginning of the film was quite slow", "The begining of the film was quite slow"),
    ("Our neighbors are throwing a barbecue on Sunday", "Our nieghbors are throwing a barbecue on Sunday"),
    ("The restaurant serves excellent grilled salmon", "The restaraunt serves excellent grilled salmon"),
]


def load_items():
    data = json.load(open(EVAL_SET))
    items = data if isinstance(data, list) else list(data.values())[0]
    out = []
    for it in items:
        prompt = it["prompt"]
        intended = it["intermediates"][0]
        corrupted = prompt.split()[-1]
        clean = prompt[: prompt.rfind(corrupted)] + intended
        out.append({"prompt": prompt, "clean": clean,
                    "corrupted": corrupted, "intended": intended})
    return out


@torch.no_grad()
def register_axis(hf, tok, band, device):
    """Contrast axis per band layer from TYPO_PAIRS, residuals averaged
    from the first diverging token position onward."""
    means = {"clean": None, "typo": None}
    for clean, typo in TYPO_PAIRS:
        ids_c = tok(clean, return_tensors="pt").input_ids[0]
        ids_t = tok(typo, return_tensors="pt").input_ids[0]
        k = next(i for i in range(min(len(ids_c), len(ids_t))) if ids_c[i] != ids_t[i])
        assert k >= 1, f"pair diverges at position 0: {typo!r}"
        for key, sent in (("clean", clean), ("typo", typo)):
            ids = tok(sent, return_tensors="pt").to(device)
            hs = hf(**ids, output_hidden_states=True).hidden_states
            act = torch.stack([hs[l + 1][0, k:].float().mean(0) for l in band])
            means[key] = act if means[key] is None else means[key] + act
    mu_c = means["clean"] / len(TYPO_PAIRS)
    mu_t = means["typo"] / len(TYPO_PAIRS)
    axis, gap = {}, {}
    for j, l in enumerate(band):
        diff = mu_t[j] - mu_c[j]
        axis[l] = diff / diff.norm()
        gap[l] = float(diff.norm())
    return axis, gap


@torch.no_grad()
def mouth_rank(hf, tok, prompt, word_ids, device):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    v = hf(ids).logits[0, -1].float()
    return int((v > v[word_ids[0]]).sum())


@torch.no_grad()
def yes_margin(hf, tok, prompt, yes_id, no_id, device, ctx=None):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    if ctx is None:
        v = hf(ids).logits[0, -1].float()
    else:
        with ctx:
            v = hf(ids).logits[0, -1].float()
    return float(v[yes_id] - v[no_id])


def main(model_key: str = "1.7b", alphas: list[float] | None = None) -> None:
    alphas = alphas or [0.125, 0.25, 0.5, 1.0]
    items = load_items()
    # disjointness: no estimation intended word may appear among eval targets
    eval_targets = {it["intended"].lower() for it in items}
    est_intended = set()
    for clean, typo in TYPO_PAIRS:
        cw, tw = clean.split(), typo.split()
        diff_words = [c for c, t in zip(cw, tw) if c != t]
        assert len(diff_words) == 1, f"pair must differ in exactly one word: {typo!r}"
        est_intended.add(diff_words[0].lower())
    overlap = est_intended & eval_targets
    assert not overlap, f"estimation words overlap eval targets: {overlap}"
    assert "blue" not in eval_targets, "one-shot demo word collides with eval targets"

    model_id, _, _ = MODELS[model_key]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    band = list(range(round(BAND_START_FRAC * model.n_layers), model.n_layers - 1))
    print(f"model={model_id} band={band[0]}..{band[-1]} alphas={alphas} items={len(items)}")

    axis, gap = register_axis(hf, tok, band, device)
    gaps = " ".join(f"L{l}:{gap[l]:.1f}" for l in band[::4])
    print(f"typo axis from {len(TYPO_PAIRS)} pairs; gaps {gaps}")

    gen = torch.Generator().manual_seed(0)
    d_model = axis[band[0]].numel()
    rand = {}
    for l in band:
        r = torch.randn(d_model, generator=gen)
        rand[l] = (r / r.norm()).to(device)

    yes_id = tok.encode(" yes", add_special_tokens=False)[0]
    no_id = tok.encode(" no", add_special_tokens=False)[0]

    records = []
    for it in items:
        intended_ids = tok.encode(" " + it["intended"], add_special_tokens=False)
        covert = mouth_rank(hf, tok, it["prompt"], intended_ids, device) >= COVERT_RANK

        def grade(text):
            # word-boundary match: "succes" must not fire inside "success"
            echo = re.search(rf"\b{re.escape(it['corrupted'])}\b", text, re.IGNORECASE)
            return {"text": text,
                    "fixed": text_match(text, it["intended"]),
                    "echo": bool(echo)}

        rec = {**{k: it[k] for k in ("prompt", "clean", "corrupted", "intended")},
               "covert": covert, "correction": {}, "report": {}}

        cor_prompt = CORRECTION_SHOT + it["prompt"] + CORRECTION_SUFFIX
        _, base_text = greedy_gen(hf, tok, cor_prompt, device, n_new=8)
        rec["correction"]["baseline"] = grade(base_text)
        rec["report"]["typo_baseline"] = yes_margin(hf, tok, it["prompt"] + REPORT_SUFFIX, yes_id, no_id, device)
        rec["report"]["clean_baseline"] = yes_margin(hf, tok, it["clean"] + REPORT_SUFFIX, yes_id, no_id, device)

        for alpha in alphas:
            erase = {l: -alpha * gap[l] * axis[l] for l in band}
            inject = {l: alpha * gap[l] * axis[l] for l in band}
            rnd = {l: alpha * gap[l] * rand[l] for l in band}
            for name, shifts in ((f"erase@{alpha:g}", erase), (f"random@{alpha:g}", rnd)):
                ctx = DirectionShiftHooks(model.layers, shifts)
                _, text = greedy_gen(hf, tok, cor_prompt, device, n_new=8, swap_ctx=ctx)
                rec["correction"][name] = grade(text)
            rec["report"][f"typo_erase@{alpha:g}"] = yes_margin(
                hf, tok, it["prompt"] + REPORT_SUFFIX, yes_id, no_id, device,
                DirectionShiftHooks(model.layers, erase))
            rec["report"][f"typo_random@{alpha:g}"] = yes_margin(
                hf, tok, it["prompt"] + REPORT_SUFFIX, yes_id, no_id, device,
                DirectionShiftHooks(model.layers, rnd))
            rec["report"][f"clean_inject@{alpha:g}"] = yes_margin(
                hf, tok, it["clean"] + REPORT_SUFFIX, yes_id, no_id, device,
                DirectionShiftHooks(model.layers, inject))
            rec["report"][f"clean_random@{alpha:g}"] = yes_margin(
                hf, tok, it["clean"] + REPORT_SUFFIX, yes_id, no_id, device,
                DirectionShiftHooks(model.layers, rnd))
        records.append(rec)
        e = rec["correction"][f"erase@{alphas[len(alphas) // 2]:g}"]
        print(f"{it['intended']:12s} covert={int(covert)} base={rec['correction']['baseline']['fixed']:d} "
              f"erase[mid]={e['fixed']:d} echo={e['echo']:d} text={e['text'][:20]!r}")

    ok = [r for r in records if r["correction"]["baseline"]["fixed"]]
    covert_ok = [r for r in ok if r["covert"]]

    def rate(rs, arm):
        return sum(r["correction"][arm]["fixed"] for r in rs) / len(rs) if rs else float("nan")

    def mean(rs, key):
        return sum(r["report"][key] for r in rs) / len(rs) if rs else float("nan")

    print(f"\nbaseline corrects: {len(ok)}/{len(records)}  (covert among them: {len(covert_ok)})")
    summary = {"n_ok": len(ok), "n_covert_ok": len(covert_ok), "per_alpha": {}}
    for alpha in alphas:
        er, rn = f"erase@{alpha:g}", f"random@{alpha:g}"
        kill_er = sum(not r["correction"][er]["fixed"] and r["correction"][rn]["fixed"] for r in ok)
        kill_rn = sum(not r["correction"][rn]["fixed"] and r["correction"][er]["fixed"] for r in ok)
        p = mcnemar_exact_p(kill_er, kill_rn)
        d_typo = mean(records, f"typo_erase@{alpha:g}") - mean(records, "typo_baseline")
        d_typo_r = mean(records, f"typo_random@{alpha:g}") - mean(records, "typo_baseline")
        d_clean = mean(records, f"clean_inject@{alpha:g}") - mean(records, "clean_baseline")
        d_clean_r = mean(records, f"clean_random@{alpha:g}") - mean(records, "clean_baseline")
        print(f"\nalpha={alpha:g}: correction erase={rate(ok, er):.1%} random={rate(ok, rn):.1%} "
              f"(baseline 100% by construction) covert-erase={rate(covert_ok, er):.1%} "
              f"only-erase-killed {kill_er} vs only-random-killed {kill_rn}, McNemar p={p:.4f}")
        print(f"  report Δmargin: typo-erase {d_typo:+.2f} (rand {d_typo_r:+.2f})  "
              f"clean-inject {d_clean:+.2f} (rand {d_clean_r:+.2f})")
        summary["per_alpha"][f"{alpha:g}"] = {
            "correction_erase": rate(ok, er), "correction_random": rate(ok, rn),
            "correction_covert_erase": rate(covert_ok, er),
            "p_erase_vs_random": p,
            "report_typo_erase_delta": d_typo, "report_typo_random_delta": d_typo_r,
            "report_clean_inject_delta": d_clean, "report_clean_random_delta": d_clean_r,
        }

    out = ROOT / "results" / f"e6t_typo_register_{family(model_id)}{model_key.replace('.', '')}.json"
    out.write_text(json.dumps({
        "model": model_id, "band": [band[0], band[-1]], "alphas": alphas,
        "n_pairs": len(TYPO_PAIRS), "covert_rank": COVERT_RANK,
        "gap": {str(l): gap[l] for l in band},
        "records": records, "summary": summary,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "1.7b"
    alphas = [float(a) for a in sys.argv[2:]] or None
    main(key, alphas)
