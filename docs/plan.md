# Experiment Plan

Created 2026-07-07.

## Models and hardware

| Role | Model | Platform | Rationale |
|------|-------|----------|-----------|
| E0 sanity | GPT-2 124M | Mac (MPS) | Cheapest full pipeline check: fit + apply + slice page |
| Main | Qwen instruct, 1.5–3B | Mac (MPS), cloud GPU if needed | Official examples use Qwen; lowest adaptation cost |
| Robustness (optional) | Llama-3.2-1B or a second family | cloud GPU | Rule out single-family artifacts; cut if time-boxed |

First check Hugging Face for officially pre-fitted lenses (the upstream README's
`from_pretrained("org/lens-repo")` placeholder); otherwise fit our own
(~100 prompts is usable per upstream).

## Milestones

| Date | Milestone | Done means |
|------|-----------|------------|
| 2026-07-13 | **M1 pipeline** | uv env works; walkthrough.ipynb produces a slice page on a small Qwen; a lens fitted on GPT-2 |
| 2026-07-20 | **M2 core results** | E1 complete with figures; E2 (swap + final-token control) has first numbers |
| 2026-07-27 | **M3 replication freeze** | E2/E3 done; every claim in the inventory has a verdict grade |

## Experiments

- **E0 sanity** (`experiments/e0-sanity/`): fit + apply on GPT-2 124M,
  qualitative slice-page check at walkthrough level.
- **E1 multi-fact editing** (`experiments/e1-flexible-generalization/`):
  official prompt set, 4 args × 4 function templates per category; swap an
  argument's lens representation, grade the next token against the new
  argument's answer. Expected: replicates (already did in review).
- **E2 thought-swap + control** (`experiments/e2-probe-swap/`): official 90
  two-hop prompts, plus our final-token substitution baseline. If the
  intermediate-entity swap does not beat the baseline by a clear margin, the
  "rewriting an intermediate thought" interpretation does not hold.
- **E3 rhyme planning** (`experiments/e3-poetry/`): quantify rhyme-word hit
  rate in the band vs a random-word control. Expected: weak or absent.
- **E4 lens false positives** (`experiments/e4-lens-eval/`): run the six
  official lens-eval sets, report false-positive and miss rates.

## Risks

- `transformers>=5.5` + MPS compatibility: fitting needs backward passes and
  MPS may miss operators. Fallback: GPT-2 locally, Qwen on a cloud GPU.
- The paper is receiving enormous attention and fast-follow work will appear;
  we prioritize the highest-stakes claims (C1, C2) first and keep the log
  honest rather than racing for coverage.
- A base (non-instruct) GPT-2 cannot run the conversational experiments;
  E0 validates lens quality only.
