# E2 — Thought-swap with a proper control (C2, P0)

Official `probe-swap.json`: 90 two-hop factual prompts; swap the intermediate
bridge entity's representation (spider→ant) across the band at every prompt
position, and check whether the final answer follows.

**What this project adds.** The external review noted the effect is "close to
substituting the final answer token directly" but published no formal control.
We add that baseline: substitute the final-answer token's direction directly
(no intermediate rewriting) and compare hit rates. If the intermediate swap
does not beat this baseline by a clear margin, the "rewriting an intermediate
thought" interpretation does not hold.

Milestones M2/M3. Metrics: swap hit rate vs baseline hit rate, per relation
category, with paired comparison across the 90 items.
