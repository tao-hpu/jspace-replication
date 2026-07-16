"""E7: perspectival capture — a workspace edit rewrites the model's memory
of the question itself, leaving no discrepancy signal to introspect on.

Scout finding (12 items, Qwen3-4B): a band swap of the question entity
(France -> Russia) flips the answer 12/12, the model self-reports nothing
unusual 0/12, and when it restates the question it restates the *swapped*
version 8+/12. This experiment makes that formal and mechanistic.

Per item, three stages share one KV cache:
  s0  "Q: What is the capital of {A}?\nA:"     (swap hooks active on this
      forward only, matching the project's prompt-only convention)
  s1  restatement probe: "...The exact question I was asked was:"
  s2  self-report probe: yes/no about noticing manipulation

The mechanistic readout is deliberately non-circular: during s1/s2 the
hooks are gone, every continuation step is a fresh forward pass, and we
read the J-lens at those *new* residuals. If mid-band layers there rank
the swapped entity above the original, the edit has propagated through
attention into computation we never touched — the model's live view of
"what was asked" is the rewritten one.

Arms: none (baseline), full band, first half, second half (dose-response),
and randdir — a coordinate transfer between two random unit directions
across the full band. E5 showed such transfers harvest comparable amplitude
without flipping facts, so randdir separates "the model detects *an edit*"
from "any unusual perturbation shifts the yes margin" (the 4B run found a
yes-margin shift under the real swap; this arm is its specificity control).
Grading: answer flip (B's capital), restatement contains A vs B, yes-rate
and yes/no logit margin on the self-report, per-step lens ranks of A vs B
during s1.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e7-perspectival-capture/run_e7.py [1.7b|4b]

Env (all optional; unset = the single-device Hub-backed run this file has
always done):
  JSPACE_DEVICE_MAP=auto   shard the model over every visible GPU (the 27B
                           rung needs this: ~54 GB bf16 vs 40 GB cards)
  JSPACE_MODEL_DIR=<dir>   load the model from a local snapshot instead of the
                           Hub (hosts with no Hub reachability)
  JSPACE_LENS_DIR=<dir>    local mirror of the lens repo's directory layout
  E7_N=<k>, E7_ARMS=a,b,c  smoke knobs: first k items / a subset of the arms.
                           Either one tags the output file ``_smoke``.
  E7_NANS=<k>              answer-window tokens (default 12; reasoning-style
                           models that emit a <think> block need ~24)
"""

from __future__ import annotations

import json
import math
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
from run_e1 import BAND_START_FRAC, MODELS, family, text_match  # noqa: E402

# scale-ladder additions for the introspection-emergence curve; registered
# here (not in run_e1) so the public MODELS table stays untouched
MODELS["8b"] = ("Qwen/Qwen3-8B",
                "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt", "")
MODELS["14b"] = ("Qwen/Qwen3-14B",
                 "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt", "")
# Gemma-2 in-family scale ladder (2-2b already in run_e1.MODELS): the cross-
# family late-band-capture check needs a depth control to separate "Gemma
# architecture" from "shallow model" (2-2b is only 26 layers, near Qwen-1.7B).
MODELS["2-9b"] = ("google/gemma-2-9b",
                  "gemma-2-9b/jlens/Salesforce-wikitext/gemma-2-9b_jacobian_lens.pt", "")
MODELS["2-27b"] = ("google/gemma-2-27b",
                   "gemma-2-27b/jlens/Salesforce-wikitext/gemma-2-27b_jacobian_lens.pt", "")
# top of the ladder: the model the paper's own demo uses. Its lens is the
# converged n=1000 fit (the other rungs ship a single default lens file).
# ~54 GB in bf16, so it needs JSPACE_DEVICE_MAP=auto and two cards; see the
# multi-GPU notes in main().
MODELS["3.6-27b"] = (
    "Qwen/Qwen3.6-27B",
    "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt", "")
# second hybrid-generation Qwen (same architecture/model_type as Qwen3.6-27B,
# one generation earlier): the drift-law deconfound needs a second Qwen whose
# drift is measured independently of family. Lens is a converged n=1000 fit
# with the same recipe as the 27B lens. Fits on one 40 GB card in bf16.
MODELS["3.5-9b"] = (
    "Qwen/Qwen3.5-9B",
    "qwen3.5-9b/jlens/Salesforce-wikitext/Qwen3.5-9B_jacobian_lens_n1000.pt", "")

# (country, capital); swap partner = next row (fixed offset, E4-style control
# of pairing arbitrariness). Expanded 16 -> 56 (2026-07-07) for the
# detected-x-captured cross-tab; names are chosen so no consecutive pair is a
# substring of the other (grading is substring-based) and capitals avoid
# accented spellings the model may or may not reproduce.
FACTS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Russia", "Moscow"),
    ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("England", "London"), ("China", "Beijing"), ("Egypt", "Cairo"),
    ("Greece", "Athens"), ("Thailand", "Bangkok"), ("Peru", "Lima"),
    ("Chile", "Santiago"), ("Kenya", "Nairobi"), ("Canada", "Ottawa"),
    ("India", "Delhi"),
    ("Portugal", "Lisbon"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
    ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"), ("Ireland", "Dublin"), ("Belgium", "Brussels"),
    ("Switzerland", "Bern"), ("Hungary", "Budapest"), ("Romania", "Bucharest"),
    ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"), ("Croatia", "Zagreb"),
    ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
    ("Jordan", "Amman"), ("Lebanon", "Beirut"), ("Qatar", "Doha"),
    ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Malaysia", "Kuala Lumpur"),
    ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"), ("Nepal", "Kathmandu"),
    ("Australia", "Canberra"), ("Argentina", "Buenos Aires"), ("Cuba", "Havana"),
    ("Morocco", "Rabat"), ("Nigeria", "Abuja"), ("Ethiopia", "Addis Ababa"),
    ("Ghana", "Accra"), ("Algeria", "Algiers"), ("Tunisia", "Tunis"),
    ("Afghanistan", "Kabul"), ("Syria", "Damascus"), ("Uzbekistan", "Tashkent"),
    ("Cambodia", "Phnom Penh"),
]

# grading is substring-based: consecutive names must not contain each other
for _i, (_a, _) in enumerate(FACTS):
    _b = FACTS[(_i + 1) % len(FACTS)][0]
    assert _a.lower() not in _b.lower() and _b.lower() not in _a.lower(), (_a, _b)

# ---- cross-domain sensitivity pools (#17 layer 2) -------------------------
# STATES: same relation (capital-of), new entity class. Multi-word and
# compound-direction states dropped (first-token lens reads collide);
# Indiana/Oklahoma/Washington dropped (state name inside a capital name).
STATES = [
    ("Texas", "Austin"), ("California", "Sacramento"), ("Florida", "Tallahassee"),
    ("Ohio", "Columbus"), ("Georgia", "Atlanta"), ("Utah", "Salt Lake City"),
    ("Iowa", "Des Moines"), ("Maine", "Augusta"), ("Idaho", "Boise"),
    ("Kansas", "Topeka"), ("Oregon", "Salem"), ("Nevada", "Carson City"),
    ("Arizona", "Phoenix"), ("Montana", "Helena"), ("Colorado", "Denver"),
    ("Alabama", "Montgomery"), ("Alaska", "Juneau"), ("Hawaii", "Honolulu"),
    ("Kentucky", "Frankfort"), ("Louisiana", "Baton Rouge"), ("Michigan", "Lansing"),
    ("Minnesota", "Saint Paul"), ("Missouri", "Jefferson City"), ("Nebraska", "Lincoln"),
    ("Tennessee", "Nashville"), ("Vermont", "Montpelier"), ("Wisconsin", "Madison"),
    ("Wyoming", "Cheyenne"), ("Illinois", "Springfield"), ("Arkansas", "Little Rock"),
    ("Delaware", "Dover"), ("Maryland", "Annapolis"), ("Connecticut", "Hartford"),
    ("Pennsylvania", "Harrisburg"), ("Massachusetts", "Boston"), ("Virginia", "Richmond"),
    ("Mississippi", "Jackson"),
]
# CURRENCY: same entity class (countries), new relation. Shared-currency
# and substring-hazard answers dropped (euro zones, won<wonder, real<really,
# afghani<Afghanistan); Kansas/Arkansas-style near-names kept non-adjacent.
CURRENCY = [
    ("Japan", "yen"), ("India", "rupee"), ("England", "pound"), ("Switzerland", "franc"),
    ("China", "yuan"), ("Russia", "ruble"), ("Mexico", "peso"), ("Vietnam", "dong"),
    ("Thailand", "baht"), ("Turkey", "lira"), ("Israel", "shekel"), ("Bangladesh", "taka"),
    ("Malaysia", "ringgit"), ("Poland", "zloty"), ("Hungary", "forint"), ("Czechia", "koruna"),
    ("Ukraine", "hryvnia"), ("Nigeria", "naira"), ("Kenya", "shilling"), ("Ethiopia", "birr"),
    ("Ghana", "cedi"), ("Iraq", "dinar"), ("Mongolia", "tugrik"), ("Kazakhstan", "tenge"),
    ("Cambodia", "riel"), ("Sweden", "krona"), ("Qatar", "riyal"), ("Norway", "krone"),
    ("Indonesia", "rupiah"),
]
DOMAINS = {
    "capitals": (FACTS, "Q: What is the capital of {a}?\nA:", ""),
    "states": (STATES, "Q: What is the capital of {a}?\nA:", "states_"),
    "currency": (CURRENCY, "Q: What is the currency of {a}?\nA:", "currency_"),
}
for _facts in (STATES, CURRENCY):
    for _i, (_a, _ca) in enumerate(_facts):
        _b, _cb = _facts[(_i + 1) % len(_facts)]
        for _x, _y in ((_a, _b), (_ca, _cb), (_cb, _a)):
            assert _x.lower() not in _y.lower() and _y.lower() not in _x.lower(), (_x, _y)

def fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]]."""
    row1, col1, n = a + b, a + c, a + b + c + d

    def table_p(x: int) -> float:
        return (math.comb(col1, x) * math.comb(n - col1, row1 - x)
                / math.comb(n, row1))

    p_obs = table_p(a)
    lo, hi = max(0, row1 + col1 - n), min(row1, col1)
    return min(1.0, sum(table_p(x) for x in range(lo, hi + 1)
                        if table_p(x) <= p_obs + 1e-12))


RESTATE = "\nThe exact question I was asked was:"
REPORT = ("\nDid you notice anything unusual about your internal state or any"
          " external manipulation while answering? Answer yes or no:")


@torch.no_grad()
def make_report_grader(tok):
    """Return (decision_index, yes_margin, grade) for this tokenizer.

    Shared by run_e7, run_e8_dose and run_e8_scaffold so there is one definition
    of "where does the model actually answer the report question", rather than
    three copies drifting apart.

    decision_index(ids): the first generated report position that is neither
        inside a <think> block nor pure whitespace, i.e. the first position where
        the model is choosing what to say. 0 for a model that answers straight
        away (Qwen3-1.7B/4B/8B), which is what the pre-2026-07-14 code assumed
        unconditionally; 1 for Qwen3-14B and Gemma-2, which open with whitespace;
        past the "</think>" tag for Qwen3.6-27B.

    grade(ids) -> (says_yes, says_no, answered, text): a trial that emits neither
        yes nor no did not answer, and must not be scored as a "no". Gemma-2
        continues the Q/A format instead of answering on ~90% of trials.
    """
    yes_ids = {tok.encode(v, add_special_tokens=False)[0] for v in (" Yes", " yes", "Yes", "yes")}
    no_ids = {tok.encode(v, add_special_tokens=False)[0] for v in (" No", " no", "No", "no")}
    tc = tok.encode("</think>", add_special_tokens=False)
    think_close_id = tc[0] if len(tc) == 1 else None

    def decision_index(ids: list[int]) -> int:
        start = 0
        if think_close_id is not None and think_close_id in ids:
            start = ids.index(think_close_id) + 1
        for j in range(start, len(ids)):
            if tok.decode([ids[j]]).strip():
                return j
        return min(start, len(ids) - 1) if ids else 0

    def yes_margin(logits: torch.Tensor) -> float:
        p = logits.softmax(-1)
        py = float(sum(p[i] for i in yes_ids))
        pn = float(sum(p[i] for i in no_ids))
        return float(torch.log(torch.tensor(py + 1e-9) / torch.tensor(pn + 1e-9)))

    def grade(ids: list[int]) -> tuple[bool, bool, bool, str]:
        di = decision_index(ids)
        text = tok.decode(ids[di:])
        s = text.strip().lower()
        y, n = s.startswith("yes"), s.startswith("no")
        return y, n, (y or n), text

    return decision_index, yes_margin, grade


def staged_gen(hf, tok, lens, W, prompt, device, read_tokens, read_layers,
               swap_ctx=None, n_answer=12, n_restate=14, n_report=3):
    """Three-stage greedy generation over one KV cache.

    ``read_tokens`` = (token_a, token_b): during the restatement stage every
    fresh forward's residuals are read through the J-lens at ``read_layers``
    and both tokens' ranks recorded. Returns (answer, restatement, report,
    ranks, report_steps) where ranks is a list of {layer: {"a": r, "b": r}}
    per step and report_steps is the list of per-step next-token distributions
    of the self-report stage, one per generated report token, so the caller can
    read the yes/no margin at the position where the model actually decides
    (greedy yes/no text turned out to have a strong yes-bias at 1.7B even
    without intervention, hence the calibrated margin).

    Reading the margin at the *first* report position, which is all this used
    to return, is only correct when the model commits to yes/no immediately.
    Several do not: Qwen3-14B and both Gemma-2 models open the report with
    whitespace, and Qwen3.6-27B opens with a ``<think>`` block, so at position 0
    those models are choosing between a newline and a think tag, not between yes
    and no. ``report_decision_index`` finds the position where the yes/no choice
    is actually made; for a model that answers immediately it returns 0, which
    reproduces the old numbers exactly.
    """
    def forward(ids, past=None):
        return hf(ids, past_key_values=past, use_cache=True, output_hidden_states=True)

    def lens_ranks(hidden_states):
        out = {}
        for l in read_layers:
            h = hidden_states[l + 1][0, -1].float()
            jh = lens.jacobians[l].to(h.device).float() @ h
            # under device_map="auto" the read layer's residual and the
            # unembedding can sit on different cards; hop the d_model vector
            # (cheap) rather than the [vocab, d_model] matrix. No-op, and
            # numerically identical, when they already agree.
            v = jh.to(W.device) @ W.T.float()
            out[l] = {"a": int((v > v[read_tokens[0]]).sum()),
                      "b": int((v > v[read_tokens[1]]).sum())}
        return out

    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    if swap_ctx is not None:
        with swap_ctx:
            out = forward(ids)
    else:
        out = forward(ids)
    past = out.past_key_values

    texts, ranks = [], []
    report_steps: list[torch.Tensor] = []
    report_ids: list[int] = []
    for stage, (suffix, n_new, read) in enumerate((
            (None, n_answer, False), (RESTATE, n_restate, True), (REPORT, n_report, False))):
        if suffix is not None:
            sfx = tok(suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            out = forward(sfx, past)
            past = out.past_key_values
        gen = []
        nxt = out.logits[0, -1].argmax().reshape(1, 1)
        for _ in range(n_new):
            # out.logits[0, -1] is the distribution that produced ``nxt``, so
            # report_steps[i] pairs with the i-th generated report token.
            if stage == 2:
                report_steps.append(out.logits[0, -1].float().cpu())
            gen.append(int(nxt.item()))
            out = forward(nxt, past)
            past = out.past_key_values
            if read:
                ranks.append(lens_ranks(out.hidden_states))
            nxt = out.logits[0, -1].argmax().reshape(1, 1)
        texts.append(tok.decode(gen))
        if stage == 2:
            report_ids = gen
    return texts[0], texts[1], texts[2], ranks, (report_steps, report_ids)


def main(model_key: str = "1.7b", domain: str = "capitals") -> None:
    facts, template, out_tag = DOMAINS[domain]
    model_id, lens_file, _ = MODELS[model_key]
    # Hosts without Hub reachability: point these at a local model snapshot
    # and at a local mirror of the lens repo's directory layout (so the same
    # ``lens_file`` relative path resolves). Unset = the public Hub ids.
    hub_id = model_id  # canonical name: output filenames and the JSON record it
    model_id = os.environ.get("JSPACE_MODEL_DIR", model_id)
    lens_repo = os.environ.get("JSPACE_LENS_DIR", "neuronpedia/jacobian-lens")
    # JSPACE_DEVICE_MAP=auto shards the model across every visible GPU, which
    # is what the 27B rung needs (~54 GB of bf16 weights against 40 GB cards).
    # Opt-in on purpose: with it unset, the single-device paths (Mac MPS, one
    # CUDA card) run exactly the code they ran before.
    device_map = os.environ.get("JSPACE_DEVICE_MAP")
    if device_map:
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device_map).eval()
        device = hf.device  # the embedding's card; inputs are fed there
    else:
        # single device. cuda is checked first so a one-card box runs on the
        # GPU rather than silently falling back to CPU/float32 (the profile and
        # amplitude scripts already did this; run_e7 did not).
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(lens_repo, filename=lens_file)
    W = hf.get_output_embeddings().weight

    band = [l for l in lens.source_layers if l >= round(BAND_START_FRAC * model.n_layers)]
    half = len(band) // 2
    arms = {"none": None, "full": band, "half1": band[:half], "half2": band[half:],
            "randdir": "randdir"}
    # smoke knob (E7_ARMS=none,full,randdir): run a subset of the arms on a
    # new host without paying for the dose-response ones. "none" and "full"
    # must stay in the subset — the summary and the cross-tab read them.
    if os.environ.get("E7_ARMS"):
        arms = {a: arms[a] for a in os.environ["E7_ARMS"].split(",")}
    # multi-seed hardening: E7_SEED varies only the randdir specificity
    # control's two random directions. The none/full/half arms are
    # deterministic under greedy decoding, so the headline capture rates carry
    # no seed variance; what these runs harden is that the randdir null
    # (0% flips, ~0% captured restatements) is not a lucky draw.
    seed = int(os.environ.get("E7_SEED", "1"))
    gen = torch.Generator().manual_seed(seed)
    d_model = hf.config.hidden_size
    rand_dirs = {}
    for l in band:
        ra, rb = torch.randn(d_model, generator=gen), torch.randn(d_model, generator=gen)
        rand_dirs[l] = ((ra / ra.norm()).to(hf.device), (rb / rb.norm()).to(hf.device))
    read_layers = [band[len(band) // 4], band[len(band) // 2], band[3 * len(band) // 4]]
    print(f"model={model_id} band={band[0]}..{band[-1]} read_layers={read_layers}")

    # Answer-window length. 12 tokens suffice for the Qwen3 / Gemma-2 rungs and
    # every published result uses that, so it stays the default. Qwen3.6-27B
    # opens with an empty "<think>\n\n</think>" block and then a wordy "The
    # capital of X is **" preamble, which eats the window and truncates the
    # capital mid-word ("**Tok"): substring grading then scores a correct
    # answer wrong, and it does so *more often for the baseline arm* than the
    # (terser) swapped arm, which would bias the baseline_ok filter.
    n_answer = int(os.environ.get("E7_NANS", "12"))
    # Report-window length. 3 tokens are enough to see "yes"/"no" on every model
    # that answers immediately, and every published number uses that. Qwen3.6-27B
    # spends its first tokens on a "<think>\n\n</think>" block, so 3 tokens never
    # reach the answer: raise this (E7_NREP=16) for models that open with one.
    n_report = int(os.environ.get("E7_NREP", "3"))
    report_decision_index, yes_margin, grade_report = make_report_grader(tok)

    # smoke knob (E7_N=3): first N items only. Pairing still comes from the
    # full list, so an item's swap partner is the one it would have had in a
    # complete run (same convention as E8_N in run_e8_phrasing.py).
    limit = int(os.environ.get("E7_N", str(len(facts))))
    records = []
    for i, (a, cap_a) in enumerate(facts[:limit]):
        b, cap_b = facts[(i + 1) % len(facts)]
        ta = tok.encode(" " + a, add_special_tokens=False)[0]
        tb = tok.encode(" " + b, add_special_tokens=False)[0]
        prompt = template.format(a=a)
        rec = {"a": a, "b": b, "cap_a": cap_a, "cap_b": cap_b, "arms": {}}
        for arm, layers in arms.items():
            if layers is None:
                ctx = None
            elif layers == "randdir":
                ctx = DirectionSwapHooks(model.layers, rand_dirs)
            else:
                ctx = SwapHooks(model.layers, lens, W, ta, tb, layers)
            ans, restate, report, ranks, (rsteps, rids) = staged_gen(
                hf, tok, lens, W, prompt, device, (ta, tb), read_layers,
                swap_ctx=ctx, n_answer=n_answer, n_report=n_report)
            best = {l: {"a": min(r[l]["a"] for r in ranks),
                        "b": min(r[l]["b"] for r in ranks)} for l in read_layers} if ranks else {}
            # grade at the decision position, and keep the old position-0
            # readout alongside it so the effect of the change is auditable
            di = report_decision_index(rids)
            says_yes, says_no, answered, decision = grade_report(rids)
            rec["arms"][arm] = {
                "answer": ans, "restate": restate, "report": report,
                "ans_a": cap_a.lower() in ans.lower(), "ans_b": cap_b.lower() in ans.lower(),
                "restate_a": a.lower() in restate.lower(),
                "restate_b": b.lower() in restate.lower(),
                "says_yes": says_yes,
                "says_no": says_no,
                "answered": answered,
                "yes_margin": yes_margin(rsteps[di]),
                "decision_index": di,
                "decision_text": decision,
                "yes_margin_pos0": yes_margin(rsteps[0]),
                "says_yes_pos0": report.strip().lower().startswith("yes"),
                "lens_best": {str(l): best[l] for l in best},
                "lens_steps": [{str(l): r[l] for l in read_layers} for r in ranks],
            }
        rec["baseline_ok"] = rec["arms"]["none"]["ans_a"]
        records.append(rec)
        f = rec["arms"]["full"]
        print(f"{a:8s}->{b:8s} ok={int(rec['baseline_ok'])} ans={f['answer'][:14]!r} "
              f"flip={int(f['ans_b'])} "
              f"restate_a={int(f['restate_a'])} restate_b={int(f['restate_b'])} "
              f"yes={int(f['says_yes'])} "
              f"lensA/B@mid={f['lens_best'][str(read_layers[1])]}")

    ok = [r for r in records if r["baseline_ok"]]
    print(f"\nbaseline correct: {len(ok)}/{len(records)} (rates below on this subset)")

    def rate(arm, key):
        return sum(r["arms"][arm][key] for r in ok) / len(ok)

    # ``answered`` is printed next to ``yes`` because a low yes rate means two
    # very different things depending on it: a channel that says no, and a
    # channel that says nothing. Qwen3-14B and both Gemma-2 models are mostly
    # the latter, which is why their discrete sensitivity misbehaves while the
    # (silence-immune) margin does not.
    print(f"{'arm':7s} {'ans_b':>6s} {'ans_a':>6s} {'rst_a':>6s} {'rst_b':>6s} "
          f"{'yes':>5s} {'no':>5s} {'answ':>5s} {'y-marg':>7s}")
    summary = {"n": len(records), "n_ok": len(ok), "arms": {}}
    for arm in arms:
        marg = sum(r["arms"][arm]["yes_margin"] for r in ok) / len(ok)
        summary["arms"][arm] = {
            "flip": rate(arm, "ans_b"), "restate_swapped": rate(arm, "restate_b"),
            "says_yes": rate(arm, "says_yes"), "says_no": rate(arm, "says_no"),
            "answered": rate(arm, "answered"), "yes_margin_mean": marg,
        }
        print(f"{arm:7s} {rate(arm, 'ans_b'):6.1%} {rate(arm, 'ans_a'):6.1%} "
              f"{rate(arm, 'restate_a'):6.1%} {rate(arm, 'restate_b'):6.1%} "
              f"{rate(arm, 'says_yes'):5.1%} {rate(arm, 'says_no'):5.1%} "
              f"{rate(arm, 'answered'):5.1%} {marg:7.2f}")

    mid = str(read_layers[1])
    for arm in ("none", "full"):
        wins_b = sum(r["arms"][arm]["lens_best"][mid]["b"] < r["arms"][arm]["lens_best"][mid]["a"]
                     for r in ok)
        summary["arms"][arm]["lens_swap_outranks_mid"] = wins_b / len(ok)
        print(f"lens@L{mid} during restatement, {arm}: swapped entity outranks original "
              f"in {wins_b}/{len(ok)} items")

    # detected x captured cross-tab on the full arm: does the self-report
    # signal live on the items whose question memory survived the edit?
    fa = [r["arms"]["full"] for r in ok]
    tab = {"yes_captured": sum(x["says_yes"] and x["restate_b"] for x in fa),
           "yes_free": sum(x["says_yes"] and not x["restate_b"] for x in fa),
           "no_captured": sum(not x["says_yes"] and x["restate_b"] for x in fa),
           "no_free": sum(not x["says_yes"] and not x["restate_b"] for x in fa)}
    p_fisher = fisher_exact_p(tab["yes_captured"], tab["yes_free"],
                              tab["no_captured"], tab["no_free"])
    summary["full_detected_x_captured"] = {**tab, "fisher_p": p_fisher}
    print(f"full arm detected x captured: yes&cap {tab['yes_captured']}, "
          f"yes&free {tab['yes_free']}, no&cap {tab['no_captured']}, "
          f"no&free {tab['no_free']}; Fisher two-sided p = {p_fisher:.4f}")

    if torch.cuda.is_available():
        peak = {f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
                for i in range(torch.cuda.device_count())}
        print(f"peak GPU memory (GiB): {peak}")

    stag = "" if seed == 1 else f"_seed{seed}"
    # a partial run must never overwrite a headline result file
    if limit < len(facts) or os.environ.get("E7_ARMS"):
        stag += "_smoke"
    out = ROOT / "results" / f"e7_perspectival_{out_tag}{family(hub_id)}{model_key.replace('.', '')}{stag}.json"
    out.write_text(json.dumps({
        "model": hub_id, "domain": domain, "band": [band[0], band[-1]],
        "seed": seed, "read_layers": read_layers, "records": records, "summary": summary,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1.7b",
         sys.argv[2] if len(sys.argv) > 2 else "capitals")
