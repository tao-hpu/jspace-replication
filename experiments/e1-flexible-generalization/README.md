# E1 — Multi-fact editing (C1, P0)

Official `flexible-generalization.json`: each category pairs 4 argument values
with 4 function templates. Baseline grades the greedy next token against the
correct answer; the swap test replaces one argument's lens representation with
another's at every prompt position across the band, then grades against the
*new* argument's answer (France→China should flip capital / language /
continent / currency together).

Expected: replicates — this was the strongest result in both the paper and the
external review.

Milestone M2. Metrics: baseline accuracy, post-swap accuracy for the new
argument, broken down by category.
