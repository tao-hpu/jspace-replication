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
