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
