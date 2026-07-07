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

## 2026-07-07 E3b planning horizon: the rhyme word's "emergence curve" is next-word plausibility ramping up, not a plan

- Motivation: E3 read only the official end-of-line-1 position. The natural
  rejoinder is that a plan might emerge later, inside line 2. So: read every
  line-2 position, align by distance-to-go d (d=1 = the token right before
  the rhyme word), and decompose each lens hit against the model's *actual
  next-token distribution* at that position. Code:
  `experiments/e3-poetry/run_e3_horizon.py`.
- Decomposition: a lens hit whose word is also in the mouth's top-10
  ("model" column) is mere local plausibility; a hit outside the mouth's
  top-10 ("antic") is candidate anticipation; the strict version requires
  outside the mouth's top-100.
- Qwen3-4B pass@10 by distance-to-go:

  | d | J-lens | control | mouth | antic (mouth≥10) | strict antic (mouth≥100) |
  |---|---|---|---|---|---|
  | 1 | 94.9% | 4.1% | 96.9% | 1.0% | 0.0% |
  | 2 | 57.1% | 1.0% | 54.1% | 8.2% | 0.0% |
  | 3 | 36.7% | 4.1% | 19.4% | 18.4% | **3.1%** |
  | 4–7 | 9–14% | 0% | 4–11% | 3–7% | 0–2.1% |
  | ≥10 | 0% | 0% | 0% | 0% | 0% |

  Qwen3-1.7B is the same shape (strict antic peaks at 6.1%, control 3.1%).
- Reading: the graded tail that looks like anticipation at a loose threshold
  is almost entirely the rhyme word *climbing into the model's own output
  distribution* as the line converges. Once the mouth's rank-100 band is
  excluded, genuine "in the lens but not in the mouth" readouts fall to the
  lens false-positive base rate measured in E4 (~3–4%). E3's 0% at the
  newline is explained: line 2 is ~10+ tokens and the curve is zero there.
- Verdict: at this scale the observable picture is **next-word prediction
  with a soft 2–3 token convergence ramp, and no mid-layer plan
  distinguishable from lens noise at any distance**. This sharpens E3 from
  "nothing at the official position" to "nothing anywhere in the line that
  survives the mouth-exclusion control".

## 2026-07-07 E4 lens quality (C5): low sensitivity, measurable false positives, and no consistent J-lens advantage over the logit lens

- Design: all six official `lens-eval-*.json` sets, at each set's designated
  readout position (README conventions; single position, min-over-layers
  rank, pass@k averaged over items; order-ops with the documented synonym
  expansion). Two additions per item: a **permutation control** (another
  item's intermediates scored identically at the same position — any hit is
  a false positive on same-distribution but absent content) and the
  **vanilla logit lens** on the same footing. Code:
  `experiments/e4-lens-eval/run_e4.py`.
- Results, pass@10 (J-lens true / J control / logit true / logit control):

  | set | Qwen3-4B | Qwen3-1.7B |
  |---|---|---|
  | multihop | 41.4 / 9.5 / 40.9 / 11.1 | 24.2 / 9.0 / 32.3 / 11.1 |
  | multilingual | 35.7 / 3.7 / 44.4 / 6.3 | 26.6 / 1.6 / 25.2 / 1.9 |
  | poetry | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
  | order-ops | 14.5 / 8.2 / 22.7 / 19.1 | 7.3 / **7.3** / 13.6 / 9.1 |
  | association | 1.0 / 0 / 2.0 / 0 | 2.0 / 0 / 0 / 0 |
  | typo | 26.0 / 0 / **69.8** / 0 | **55.2** / 0 / 43.8 / 0 |

- Four takeaways:
  1. **Sensitivity is low across the board.** The best set tops out at ~41%
     pass@10; `association` (the evocative "grief vignette" reading) is ~0
     at both scales; poetry is 0 (consistent with E3).
  2. **False positives are real and quantified.** The multihop control hits
     ~9–11% at rank ≤ 10 — the criterion fires about one time in ten on
     content that is not there. On order-ops the apparent signal is mostly
     or entirely base rate: 14.5% vs 8.2% control at 4B, and **7.3% vs
     7.3% at 1.7B — exactly zero signal above chance** for numbers and
     operations.
  3. **No consistent J-lens advantage over the 2020 logit lens.** Multihop:
     logit wins or ties at both scales. Multilingual: logit wins at 4B,
     ties at 1.7B. Typo flips with scale (logit dominates at 4B 69.8 vs
     26.0; J wins at 1.7B). On these open models the headline "J-lens reads
     concepts the logit lens cannot" does not show up in its own evaluation
     distributions.
  4. Our readout is a *single* position. The paper's workspace-band "hit"
     criterion scans layers × many positions, a much larger
     multiple-comparison surface — so these control rates are a **lower
     bound** on that criterion's false-positive rate.
- Verdict: **C5 concern confirmed with numbers** at this scale. The lens is
  a usable but noisy instrument: strong only where the target is imminent
  or surface-aligned, near chance on operations, blind to evoked concepts,
  and not consistently better than its 2020 baseline.
- Caveats: neuronpedia lens fits (wikitext prompts) may differ from the
  paper's internal fits; frontier-model behavior may differ; pass@k on
  single tokens under-credits multi-token concept readouts.

## 2026-07-07 E3 rhyme planning: total negative at both scales — the models rhyme without any lens-readable plan

- Design: official 98-couplet `lens-eval-poetry.json`; the planning claim is
  that the line-2 rhyme word is lens-readable at the *end of line 1* (last
  newline token) before any of line 2 exists. Per item, at that position:
  J-lens min-over-layers rank of the rhyme word (official pass@k), a
  permutation control (another item's rhyme word, same position), the
  logit-lens rank, plus two sanity checks — greedy continuation (does the
  model rhyme as intended?) and the readout at the final prompt token
  (imminent word, should be visible). Code: `experiments/e3-poetry/run_e3.py`.
- Results:

  | model | rhymes as intended | newline pass@10 (J / logit / control) | final-token pass@1 (J) | sign test target<control |
  |---|---|---|---|---|
  | Qwen3-4B | 78/98 | **0% / 0% / 0%** | 78.6% | 59/98, p=0.054 |
  | Qwen3-1.7B | 67/98 | **0% / 0% / 0%** | 64.3% | 56/98, p=0.19 |

  Median newline rank of the target: 3658 (4B) / 6140 (1.7B) — three orders
  of magnitude from readable. Restricting to items the model actually rhymes
  as intended changes nothing (pass@10 stays 0%).
- Both sanity checks pass: the models complete the couplets with the intended
  rhyme word at 68–80%, and the final-token readout matches that rate almost
  exactly (78.6% vs 79.6% baseline at 4B), so the pipeline reads what is
  there. There is simply nothing to read at the newline: even the most
  charitable paired test finds at best a marginal, non-significant tendency
  for the target to outrank an arbitrary rhyme word.
- Verdict: **not replicated at this scale**, in full agreement with the
  external review. The models rhyme correctly *without* a lens-readable plan
  — either the plan exists in a non-verbalizable form, or the rhyme word is
  chosen on the fly at the final position. Note the negative is not
  lens-specific (logit lens also reads nothing), so this is not a J-lens
  fitting artifact.
- Scale caveat as usual: the paper's planning result is on a frontier model;
  1.7B/4B may genuinely not plan ahead. What this establishes is that the
  claim does not transfer down to open models even qualitatively.
- Next: E4 lens false-positive quantification on the six evaluation sets.

## 2026-07-07 E2p follow-up: probe-family directions do not rescue the intermediate swap — the direction-choice caveat narrows

- Design: rerun both E2 arms with the direction source swapped from
  Jacobian-lens to **mass-mean probe** directions —
  `normalize(mean(entity) − grand mean)` over final-token residuals from 12
  neutral templates (208 entities), the closed-form stand-in for a trained
  linear probe. Mechanism, band, positions, grading, and items identical to
  E2; baselines reused from the E2 run (same greedy decoding). Code:
  `experiments/e2-probe-swap/run_e2_probe.py`.
- Results (baseline-correct items; lens numbers from E2 for comparison):

  | model | n | A': intermediate | B': answer ctrl | A' vs B' p | A lens→probe p | lens was A / B |
  |---|---|---|---|---|---|---|
  | Qwen3-4B | 48 | 20.8% | 29.2% | 0.50 | **0.0043** (4 gained, 18 lost) | 50.0% / 85.4% |
  | Qwen3-1.7B | 29 | 31.0% | 20.7% | 0.51 | 0.42 (5 gained, 9 lost) | 44.8% / 82.8% |

- Both arms collapse under probe-family directions; the A-vs-B gap vanishes
  because *both* interventions get weak, not because A catches up. At 4B the
  intermediate swap is significantly *worse* along probe directions than
  along lens directions. The mass-mean operator is also visibly cruder: it
  breaks generation outright (degenerate repetition/garbage) in 18.8% (A') /
  10.4% (B') of 4B items vs **0% for lens directions in every E2 arm**
  (1.7B: 0% / 3.4%).
- Verdict: **the E2 direction-choice caveat narrows substantially.** A's
  weakness in E2 cannot be attributed to lens directions handicapping the
  intermediate swap — moving toward the probe family makes A weaker, not
  stronger. Within every operator we can implement, the lens direction is
  the cleanest and strongest, and under it the answer-token control still
  significantly beats the intermediate swap.
- Remaining gap, stated plainly: the official experiment uses *trained*
  linear probes, which are likely cleaner than the mass-mean stand-in (the
  broken-generation rate shows the stand-in carries non-identity context
  components). A trained-probe rerun could still behave differently; this
  entry closes the "was the lens direction unfair to A" branch, not the
  "would official probes do better" branch.
- Side observation echoing E1: in `amazon-language`, A' continues " the
  Mexican" — the intermediate did transfer (Brazil→Mexico) but the second
  hop never ran; the model verbalizes the injected concept instead of
  computing on it.

## 2026-07-07 E2 adjudication: the answer-token control significantly BEATS the intermediate swap at both scales

- Design: official 90 two-hop items; per item three greedy continuations —
  baseline, arm A (swap intermediate→swap_to across all fitted layers at all
  prompt positions), arm B (control: swap answer→swap_answer, identical
  mechanism/band/positions). Hit = continuation matches swap_answer. Paired
  exact McNemar on baseline-correct items. Code:
  `experiments/e2-probe-swap/run_e2.py`.
- Results (baseline-correct items):

  | model | n | A: intermediate swap | B: answer control | discordant A:B | p |
  |---|---|---|---|---|---|
  | Qwen3-4B | 48 | 50.0% | **85.4%** | 3:20 | 0.0005 |
  | Qwen3-1.7B | 29 | 44.8% | **82.8%** | 2:13 | 0.0074 |

- Verdict: **the review's critique is confirmed and sharpened** at this
  scale. The headline "rewrite an intermediate thought" intervention is not
  merely indistinguishable from trivial answer substitution — it is
  significantly *weaker* than it. Whatever arm A achieves (≈half the items),
  a mechanism with direct output leverage does far more reliably; nothing in
  this data requires the "edited thought propagates through the second hop"
  interpretation.
- Honest caveats, prominently: (1) the official experiment swaps along
  *linear-probe* directions; both our arms use Jacobian-lens directions. The
  A-vs-B comparison is internally fair, but A's absolute weakness could
  partly reflect the direction choice — training intermediate-entity probes
  is the natural follow-up before quoting this against the paper. (2) Models
  here are 1.7B/4B; the paper's results are on a frontier model. Scale could
  rescue A.
- This was the project's central open question (candidate 1): outcome is the
  sharp-negative branch, pending the probe-direction follow-up. (Follow-up
  ran same day — see the E2p entry above; the caveat narrows.)

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
