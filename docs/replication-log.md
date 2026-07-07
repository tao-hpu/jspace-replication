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
