# Replication Log

Newest first. Every experiment gets an entry the day it runs, failures included.
Template:

```
## YYYY-MM-DD  E{n} one-line takeaway
- Model / lens config:
- What was run:
- Results (concrete numbers; figures in results/figures/):
- Verdict: replicated / partially replicated / not replicated / not tested
- Agreement with the external review:
- Next:
```

---

## 2026-07-07 E1 second pass: band sweep + Qwen3-4B; the swap boundary is function type, not category

- **Band sweep (1.7B, countries hit_b / stayed_a)**: all-layers 0–26:
  91.7%/4.2%; early-mid **4–13 alone: 91.7%/4.2%** (full effect); default
  8–26: 87.5%; late 14–26: 79.2%; tail 20–26: 70.8%. The editable argument
  representation lives in the early-mid layers; late-only intervention
  degrades. The first_letter immunity (below) is 0% under every band.
- **Qwen3-4B** (band 10–34 by the same fraction, baseline 35/64): countries
  97.2% hit / 0% stayed (n=36) — the headline effect strengthens with scale.
- **Per-function aggregation across both models** (baseline-correct pairs):
  - Associative lookups replicate at or near ceiling at 4B: capital 92%,
    continent 100%, currency 100%, language 100%, all 0% stayed.
  - `numbers/first_letter`: 0% hit / **100% stayed** at both scales — the
    model reports the *original* argument's first letter. Orthographic
    properties appear to read from the actual token, untouched by the
    lens-space swap. (This corrects the first-pass entry, which attributed
    the numbers result to double/square; those mostly failed baseline.)
  - `months/next_month` (4B): 0% hit / 0% stayed — an *echo* failure mode:
    swap February→April and the model answers "April", verbalizing the
    injected concept instead of computing its successor (May).
  - `animals/legs` (4B): 0% hit.
- Verdict refinement: the swap rewrites what a concept is associatively
  linked to, but (a) surface-form-derived properties bypass it entirely
  (first_letter), and (b) functions that must *operate on* the concept
  either ignore it or collapse to echoing it (next_month). Three distinct
  failure signatures — stayed / echo / broken — worth their own figure.
- Method notes: the zsh no-word-splitting gotcha silently reran the default
  band five times before the sweep (caught by identical outputs); month
  names are single tokens in Qwen3, ruling out tokenization as the
  next_month explanation.
- Next: E2 (probe-swap + final-token control) — today's function-type split
  sharpens its stakes: if swap effects are echo-like, the control may win.

## 2026-07-07 E1 first pass (Qwen3-1.7B): fact-editing replicates strongly for associative facts, fails completely for computed ones

- Config: Qwen3-1.7B (bf16, MPS), pre-fitted wikitext lens, band = layers
  8–26 (28%→end, matching the Neuronpedia demo fraction on 27B). Swap
  implemented in `src/interventions.py` (upstream ships no intervention code):
  direction = normalize(J_l^T @ W_U[token]), coordinate transfer
  h' = h − (h·dA)dA + (h·dA)dB at all prompt positions; hooks active for the
  prompt forward only, continuation greedy-decoded from the swapped KV cache.
- Grading: official next-token grading is too brittle at 1.7B (numeric answers
  arrive after a filler token), so grading is greedy 6-token continuation,
  prefix-match after stripping punctuation. Strict next-token result kept in
  the records. Baseline 24/64 overall (the base model is simply weak on many
  of these completions); headline numbers below are on baseline-correct pairs.
- Results (192 ordered swap trials; n = baseline-correct pairs):

  | category | swap→new answer | stayed at old | n |
  |---|---|---|---|
  | countries | **87.5%** | 4.2% | 24 |
  | months | 66.7% | 16.7% | 6 |
  | animals | 25.0% | 0% | 8 |
  | numbers | **0%** | **100%** | 12 |

  (Correction, same day: the numbers pairs that survived baseline were the
  `first_letter` function, not double/square as first written — see the
  per-function entry above.)

- Verdict: C1 **replicated** for associative facts at 1.7B — France→China
  flips capital/language/continent/currency answers with our own independent
  swap implementation. The clean split is the interesting part: swapping the
  input representation propagates through associative lookup but not through
  computed functions (double/square), where the output sticks to the original
  argument 100% of the time.
- Agreement with the external review: convergent — Nanda failed to replicate
  the mental-arithmetic experiments; here the same boundary shows up inside a
  single experiment as an associative-vs-computed split.
- Next: repeat on Qwen3-4B (stronger baseline should widen n), band-range
  sweep, and a bilingual probe rider (candidate-3 scouting).

## 2026-07-07 E0 complete: the gpt2 anomaly is checkpoint-specific, not scale- or lens-specific

- What was run: completed the 2×2 control. (a) Converted the self-trained
  GPT-2 124M (nanoGPT-style ckpt, fineweb-edu 10B tokens, val 3.02) to HF
  format (`convert_selftrained.py`; sanity: ppl 8.3 on a fixed paragraph,
  coherent greedy sample). (b) Fitted lenses with the official wikitext recipe
  (150 prompts, ~9 min each on MPS — backward pass works fine) on both the
  self-trained model and the official `gpt2`. (c) Re-ran the 3 probes.
- Results (target rank, J-lens vs logit lens):

  | model | lens | J-lens advantage? |
  |---|---|---|
  | official gpt2 | official (277 prompts) | ✗ inverted at all layers |
  | official gpt2 | ours (150 prompts) | ✗ same pattern (eiffel L8: 314 vs 11) |
  | self-trained 124M | ours (150 prompts) | ✓ clear (eiffel L9: **1** vs 28; ioi L9: **5** vs 277) |
  | pythia-70m | official | ✓ clear (see previous entry) |

- Detail worth keeping: on the self-trained model, ioi-mary's J-lens top-1 at
  L9 is ` her` — the gender pronoun surfaces one layer before the name
  ` Mary` (L10 rank 1). Concept-before-surface, at 124M.
- Verdict: the paper's "J-lens recovers content where the logit lens cannot"
  is **replicated on 2 of 3 tiny checkpoints**; the exception (openai gpt2)
  is a property of that checkpoint, not of the lens fit, the architecture, or
  the 124M scale (the self-trained model shares the architecture and size).
- Hypotheses for the gpt2 exception (untested): (1) its residual basis is
  unusually well aligned with output space — the logit lens was originally
  demonstrated on GPT-2, leaving the J-transport little to correct and its
  noise net-negative; (2) glitch-token directions (`ModLoader`, ` enthusi`, …)
  dominate its average Jacobian. Both are checkable later; parked.
- Next: E1 on qwen3-1.7b (pre-fitted lens, official flexible-generalization
  set). M1 (pipeline) is effectively closed 6 days early.

## 2026-07-07 E0 pipeline works on MPS; J-lens vs logit lens diverges between the two smallest models

- Model / lens config: GPT-2 124M (`gpt2`) and Pythia-70m-deduped, each with its
  pre-fitted wikitext lens from `neuronpedia/jacobian-lens` (gpt2 lens:
  layers 0–10, 277 fit prompts). Device: MPS; apply() runs in seconds.
- What was run: `experiments/e0-sanity/run_e0.py` — 3 probes (eiffel-paris,
  boot-italy, ioi-mary), target-token rank per layer under J-lens vs logit
  lens at the final position. Results in `results/e0_gpt2.json`,
  `results/e0_pythia70m.json`.
- Results:
  - **gpt2-small: J-lens does NOT beat the logit lens on target rank at any
    layer for any probe** (e.g. eiffel-paris L8: J-rank 232 vs logit 11).
    J-lens late-layer top-5 shows the right semantic *category* (a city
    cluster: Cologne/Amsterdam/Hamburg) without ranking the target better.
    Early-layer J readouts are dominated by GPT-2's known glitch tokens
    (`ModLoader`, ` enthusi`, …).
  - **pythia-70m: the J-lens advantage is clear.** eiffel-paris from L2 on
    (J 128 vs logit 3745; L3 J 25 vs 327); ioi-mary hits rank 1 at L3 under
    the J-lens while the logit lens is at 313, with a coherent drink-scene
    cluster (laughter/liquor/milk/whiskey) at L2.
  - boot-italy: both models fail the task itself (output rank 2108 / 1021) —
    too small for the latent-Italy inference; probe kept for larger models.
- Verdict: pipeline sanity **passed**; the paper's "J-lens recovers content
  where the logit lens cannot" claim is **partially replicated** at tiny
  scale — present on pythia-70m, absent/inverted on gpt2-small over these
  3 probes.
- Agreement with the external review: n/a (review did not test tiny models).
- Next: more probes + a proper aggregate metric before reading anything into
  the gpt2/pythia divergence; fit a lens on the self-trained 124M; then E1 on
  qwen3-1.7b.

## 2026-07-07 API access verified: remote swap intervention works (mini-E1)

- Captured the playground's request schema (`POST
  https://www.neuronpedia.org/api/lens/prompt`, `x-api-key` auth, NDJSON
  stream). Body: `modelId`, `chat[]`, `type: ["JACOBIAN_LENS"|"LOGIT_LENS"]`,
  `topN`, `steerTokens`/`swapToken`/`steerLayers`/`steerStrength`. Response
  streams per-token lines with per-layer `top_tokens` readouts plus the
  completion.
- Mini fact-editing test on Qwen3.6-27B (temperature 0, layers 18–63):
  - Baseline "What is the capital of France? Answer in one word." → **Paris**
  - Same prompt with J-space swap ` France`→` China` → **Beijing**
  - A clean remote causal intervention in ~4 s; large-model qualitative arm of
    E1 can run entirely over this API.
- Incidental C5 evidence: on function tokens (e.g. `</think>`) the early-layer
  readout is vocabulary junk (incl. NSFW web tokens) — a vivid example of the
  lens producing uninterpretable readouts outside the workspace band.

## 2026-07-07 Recon: pre-fitted lenses exist; hands-on session on Neuronpedia

- Found `huggingface.co/neuronpedia/jacobian-lens` (MIT): 35+ pre-fitted
  lenses incl. `gpt2-small`, `qwen3-1.7b`, `qwen3-4b`, `gemma-3-1b`,
  `pythia-70m-deduped`. Fitting is now only needed for self-trained weights;
  E0 scope reduced accordingly.
- Ran the Neuronpedia interactive playground (Qwen3.6-27B, live compute):
  - Verbal-report demo: swapping `tennis`→`rugby` in the J-space (layers
    18–63) flips the sampled answer Tennis→Rugby.
  - Jacobian vs logit mode on the same prompt: `tennis` hits 162 across the
    mid-layer band under the J-lens vs 63 confined to the last layers under
    the logit lens, whose mid-layer readout is fragment noise (`_a`, `S`,
    `ing`, digits). Live illustration of the paper's coordinate-drift claim.
  - Own prompt ("Think of an animal that lives in the desert"): model answers
    *Camel*; the J-space top readout is dominated by task-frame words
    (answer/animal/desert in English and Chinese), with the answer itself
    lower down — 骆驼 95 hits (mid layers), camel-variants ~200 combined
    (later layers). Two takeaways: (1) readout ranking ≠ answer saliency,
    consistent with the false-positive concern (C5); (2) the bilingual
    ordering (Chinese concept token before English surface forms) is worth a
    systematic look on Qwen models.

## 2026-07-07 Project start

- Cloned official `anthropics/jacobian-lens` (Apache 2.0); confirmed it ships
  all experiment prompt sets and an end-to-end walkthrough notebook. Pinned
  commit recorded in `third_party/PINNED_COMMIT.txt`.
- Built the claims inventory (C1–C6) and milestones (M1 07-13 / M2 07-20 /
  M3 07-27).
- No experiments run yet.
