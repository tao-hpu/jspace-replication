# Claims Inventory

Three-way comparison: the paper's claim, the external review's verdict
(Nanda, [LessWrong](https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper)),
and this project's replication priority. Official prompt sets live in
`third_party/jacobian-lens/data/`; construction and scoring rules are in
`data/experiments/README.md` upstream (slugs below refer to those files).

| # | Claim | Official prompt set | Review verdict | Priority | Notes |
|---|-------|--------------------|----------------|----------|-------|
| C1 | Shared fact representations: swapping one concept (France→China) flips every dependent answer (capital / language / continent / currency) at once | `flexible-generalization.json` | ✅ replicated; strongest result in the paper | **P0** | Positive anchor; run first |
| C2 | Thought-swap: intermediate representations in two-hop reasoning can be rewritten (spider→ant) | `probe-swap.json` | ⚠️ weak; close to substituting the final answer token directly | **P0** | We add the missing control: a final-token substitution baseline. This is the one place this project goes beyond the public review |
| C3 | Rhyme planning: rhyme words are held in the workspace ahead of time during poetry | `lens-eval-poetry.json` | ❌ failed to replicate | **P1** | A quantified negative result is still a result |
| C4 | Mental arithmetic completes internally and is lens-readable | `directed-modulation.json` (math_problems), `dual-task.json` (arithmetic) | ❌ failed to replicate | P2 | Same "failed in review" family as C3 |
| C5 | J-lens reliability: the lens reads out internal concepts | `data/evaluations/lens-eval-*.json` | ⚠️ heavy false positives; "won't reliably flag everything important". The paper itself calls the method "undoubtedly imperfect" | P1 | Quantify false-positive rate with the official eval sets |
| C6 | The structure constitutes a *global workspace* (ignition, capacity, selectivity experiments) | `ignition.json`, `capacity.json`, `selectivity-*.json`, `top-down-summoning.json` | Doubtful; the reviewer considered this the least important, least confident analogy | Not planned | Discussed, not compute-funded |

## Scoring conventions (inherited from upstream)

- **Workspace band**: the contiguous mid-network layer range; results aggregate
  over the band, not per-layer.
- **Hit**: target token at lens rank 1 at any (layer, position) in the band.
- **Swap**: clamp a lens coordinate, replacing one token's direction with
  another's at every band layer at the specified positions, then sample.

## Verdict scale for our results

Each measured claim is graded **replicated** / **partially replicated**
(direction agrees, magnitude clearly weaker) / **not replicated** / **not
tested**. Any disagreement with the external review gets its own note on
plausible causes (model family, scale, hyperparameters).

## Replication verdicts (graded 2026-07-07)

All experiments on Qwen3-1.7B and Qwen3-4B with neuronpedia pre-fitted
lenses; every number below is traceable to a dated entry in
`docs/replication-log.md` and a JSON in `results/`. Standing caveat for
every row: the paper's results are on a frontier model; these verdicts
establish what does and does not transfer to small open models.

| # | Our verdict | Evidence | vs. the review |
|---|-------------|----------|----------------|
| C1 | **Replicated, and strengthens with scale** | E1: countries swap 97.2% hit / 0% stayed at 4B; early-mid band (4–13 of 28) alone carries the full effect. Boundary mapped: only associative links move; operations on concepts fail with three distinct signatures (stayed / echo / broken) | Agrees, and adds the band localization and the failure taxonomy |
| C2 | **Not replicated (interpretation unsupported)** | E2: the answer-token control significantly *beats* the intermediate swap (4B 85.4% vs 50.0%, p=0.0005; 1.7B p=0.0074). E2p: probe-family directions do not rescue it. E5: a non-answer source flips the answer as well as the answer itself (77.1% vs 85.4%, p=0.29), and flips are predicted by direction cosine, not source identity | Sharper than the review: not merely "close to" answer substitution — significantly weaker than it, and non-specific |
| C3 | **Not replicated** | E3: rhyme word never in lens top-10 at the official readout position (pass@10 = 0% both scales; positive sanity checks pass). E3b: no anticipation above the lens false-positive base rate anywhere in line 2 once next-word plausibility is excluded | Agrees, with the emergence-curve decomposition added |
| C4 | **Not replicated** (lens-readability route; official intervention sets not run) | E4 order-ops: number/operation readout at or near base rate (1.7B: exactly 0 above chance; 4B: 14.5% vs 8.2% control) | Agrees |
| C5 | **Concern confirmed and quantified** | E4: best-set sensitivity 41% pass@10, association ~0; permutation control fires at 9–11% on multihop; and the J-lens shows **no consistent advantage over the vanilla logit lens** on its own eval sets (typo at 4B: 26.0% vs 69.8%) | Confirms the false-positive concern with numbers; the J-vs-logit null is beyond the review |
| C6 | **Not tested** (by design) | — | — |
