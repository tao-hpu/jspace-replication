# jspace-replication

An independent, third-party replication of the core experiments in
[**Verbalizable Representations Form a Global Workspace in Language Models**](https://transformer-circuits.pub/2026/workspace/index.html)
(Anthropic, 2026) on small open-weights models, with systematic bookkeeping of
which claims hold up and which do not.

## Why another replication

The paper shipped with one external review ([Neel Nanda's](https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper)),
and his results were mixed: the rhyme-planning and mental-arithmetic experiments
failed to replicate, the multi-hop probe-swap effect was weak, and the Jacobian
lens produces many false positives, while the multi-fact editing result
(swap *France*→*China*, and capital / language / continent / currency answers
all change together) replicated cleanly.

This project re-runs the highest-stakes claims from scratch and adds one control
the public review lacked: a direct final-token substitution baseline for the
probe-swap experiment (see `experiments/e2-probe-swap/`). Negative and partial
results are reported with the same care as positive ones.

## Upstream resources

- Paper: https://transformer-circuits.pub/2026/workspace/index.html
- Official companion code: https://github.com/anthropics/jacobian-lens
  (Apache 2.0; cloned at the commit recorded in `third_party/PINNED_COMMIT.txt`,
  includes all experiment prompt sets under `data/`)
- Anthropic research page: https://www.anthropic.com/research/global-workspace

## Layout

```
docs/
  claims-inventory.md   paper claims → official prompt set → review verdict → our priority
  plan.md               experiment plan, model/hardware choices, milestones
  replication-log.md    running log; failures recorded the same day as successes
src/                    shared utilities (MPS adaptation, plotting)
notebooks/              exploration only; canonical results live in experiments/
experiments/            one directory per experiment: script + config + README
results/                figures and metric JSONs
third_party/
  jacobian-lens/        official reference implementation (never modified; wrap in src/)
```

## Setup

The official code uses `uv` and requires `transformers>=5.5`:

```bash
git clone --depth 1 https://github.com/anthropics/jacobian-lens third_party/jacobian-lens
uv venv && source .venv/bin/activate
uv pip install -e third_party/jacobian-lens
uv pip install jupyter matplotlib einops
```

Target models are small enough for a laptop: GPT-2 124M for pipeline sanity,
Qwen 1.5–3B instruct for the main experiments (the official examples use Qwen).
Per the upstream README, ~100 fitting prompts already give a usable lens
(the paper uses 1000 × 128 tokens).

## Ground rules

- Every result distinguishes three sources: the paper's claim, the external
  review's verdict, and our measurement. They are never blended.
- Every experiment gets a log entry the day it runs, including failures.
- `third_party/` is never modified; adaptations live in `src/`.
