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

The project grew past replication into an audit and reframing, written up as a
paper (arXiv link forthcoming). The additions, all in `experiments/` with
results JSONs in `results/`:

- **Mouth-exclusion audit** (`e4-lens-eval/run_e4_covert.py`): every lens hit
  scored against the model's own next-token distribution; covert content
  survives almost exclusively for *context registers* (language identity,
  intended form of a typo), not content plans.
- **Causal register control** (`e6-covert-register/`, `e6t-typo-register/`):
  measured-gap translations of a language axis and a typo axis, with
  amplitude-matched random controls and dose curves.
- **Perspectival capture** (`e7-perspectival-capture/`): a mid-band entity
  swap rewrites the model's restatement of the question itself, stably across
  a 1.7B-14B ladder, while self-report about the edit changes shape at every
  scale.
- **Transport-cone geometry** (`results/cone_*.json`): raw vs J-transported
  effective dimensionality across the model ladder.
- **Statistics** (`stats/run_bootstrap.py`, `sensitivity/reanalyze.py`):
  bootstrap CIs for every headline rate and a prompt-set sensitivity
  reanalysis.

## The self-trained 124M control

The transport-cone ladder includes one model we trained ourselves: a 124M
GPT-2 reproduction (standard nanoGPT recipe), the same checkpoint released at
[tao-hpu/llm-from-scratch](https://github.com/tao-hpu/llm-from-scratch)
(weights on Hugging Face). It was trained for that unrelated reproduction, not
for this paper, so it is a pre-existing, publicly checkable artifact rather than
a hand-picked point.

It is the load-bearing counterexample in the geometry section: unlike every
other model, its transported directions are *more* isotropic than its raw ones
(effective dimensionality 23.4 → 31.2), which is what proves the collapse
elsewhere is a property of the fitted lens, not a mathematical necessity of the
transport. Its Jacobian lens is the only one we fit ourselves; it can be refit
from the public weights with `experiments/e0-sanity/fit_selftrained.py`, and the
fitted lenses are also downloadable directly from the
[`selftrained-124m-lens-v1` release](https://github.com/tao-hpu/jspace-replication/releases/tag/selftrained-124m-lens-v1)
(both the published `n150_mps` lens and a converged `n1000_cuda` refit).
The non-collapse is robust: transported effective dimensionality stays ~31
(31.2 → 31.1) across fit scale (150 → 1000 prompts) and backend (MPS → CUDA).

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
Qwen3 1.7B-4B for the register experiments and up to 14B for the capture ladder (128 GB unified memory covers 14B bf16).
Per the upstream README, ~100 fitting prompts already give a usable lens
(the paper uses 1000 × 128 tokens).

## Ground rules

- Every result distinguishes three sources: the paper's claim, the external
  review's verdict, and our measurement. They are never blended.
- Every experiment gets a log entry the day it runs, including failures.
- `third_party/` is never modified; adaptations live in `src/`.

## License

Our own code and result files are released under the MIT License (see `LICENSE`).
Upstream artifacts keep their own terms: the official jacobian-lens code is Apache
2.0, Qwen3 weights are Apache 2.0, and Gemma-2 weights are under the Gemma Terms of
Use.
