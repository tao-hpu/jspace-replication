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

## 2026-07-16 E7 mechanism campaign synthesis (no new runs): the write-zone cutoff is a composite - whichever comes first of the drift wall (~0.5-0.8 window-mean) and the task's consumption deadline - and it fits all nine models

- Scope: synthesis over the 07-15/16 zone-map / deadline / healing /
  raw-direction / drift entries below (nine models: dense Qwen3 1.7B/4B/8B/14B,
  hybrid Qwen3.5-9B, Qwen3.6-27B, Gemma-2-2B/9B/27B). Merge table:
  `analyze_e7_driftlaw.py` -> `results/e7_driftlaw_table.json`; figures:
  `make_driftlaw_figures.py` -> `e7_mechanism` / `fig_writable_zone` /
  `fig_deadline` (in `results/figures/`).
- Instrument -> result-file map (the entries below describe the runs; file
  stems for traceability): zone maps = `e7_slide6_*` (incl. `_states` /
  `_currency` domain and `_randdir` control variants), deadlines =
  `e7_deadline_*`, healing = `e7_healing_*`, raw-direction controls =
  `e7_rawdir_*`, minimum write width = `e7_kwin_*`, single-layer profiles =
  `e7_profile_*`, drift = `e7_drift_*`, gate test = `e7_gatetest_*`,
  persistence = `e7_persistence_*`.
- The reading that survives the campaign:
  1. Zone maps (width-6 windows, stride 2): EVERY model is writable somewhere.
     The open/closed single-layer taxonomy was a dose artifact (Qwen3.6-27B:
     3.6% single-layer peak vs 87.5% at the L36 window, 28-layer plateau >70%);
     matched-dose random-direction windows flip 0% everywhere (Qwen3-8B,
     Gemma-2-9B), so width-6 writability is direction-specific.
  2. The cutoff belongs to the task, ordinally: within models the tail dies in
     domain order (states < capitals < currency); the intervention-free commit
     layer rank-correlates with the zone cutoff at rho=0.86 (n=7), and the
     deadline sits after the cutoff in every model.
  3. Drift is a within-model clock with a common wall: six of the seven models
     that place a cutoff land it at window-mean drift ~0.5-0.8. The exception,
     Gemma-2-27B (flip dead by drift 0.21), is also the earliest-deadline model.
  4. Across families drift orders capture only coarsely (family residual): no
     high-drift dense-Qwen late band captures (<=17.9% at 0.76-0.79) and
     lowest-drift Gemma-2-27B captures late at 55.4%, but hybrid Qwen3.5-9B
     captures 62.7% at dense-range drift 0.72 - not a per-layer threshold.
- Retired readings, falsifying data on record: the open/closed gate taxonomy
  (dose artifact) and the universal cross-model drift threshold (killed by
  Gemma-2-27B, its sixth data point). Pre-registered predictions: the
  deadline-ordering prediction hit; the g27 healing-onset prediction missed.
- Verdict: composite cutoff = min(drift wall, task deadline) fits 9/9 but only
  Gemma-2-27B separates the two terms, so the paper flags it as a candidate
  reading, not a result. Paper updated: Section 6 drift paragraph rewritten to
  the nine-model version, new Section 7 (zone geometry, five instruments), new
  contributions bullet, appendix tables/figures regenerated.
- Next: a second model whose deadline precedes its drift wall would separate
  the composite's two terms; dose-aware healing measurements would close the
  open onset mechanism.

## 2026-07-16 Backfill: earlier runs that predate the entries below (multi-seed hardening; single-layer profiles, persistence, gate test, amplitude, report-grader fix)

- Context: these runs happened while the mechanism section was being drafted
  and were logged only in working notes at the time; recorded here
  retroactively so every public result file has a log entry.
- Multi-seed hardening (07-08, commit adabc38, `aggregate_multiseed.py`,
  `results/*_seed*.json`): re-estimating the E6 register axis on bootstrap
  resamples of the 16 parallel pairs (five seeds) leaves the operating-point
  full-flip at 56.2 +/- 1.6% (1.7B; canonical 55.6%) and 59.5 +/- 2.4% (4B;
  canonical 64.6%), with the amplitude-matched random control at exactly 0% on
  every seed at every dose. E7 capture is deterministic under greedy decoding,
  so only the randdir null is seed-averaged: 0% flips and 0% captured
  restatements across three seeds at both scales.
- Single-layer capture profiles (07-13, `run_e7_profile.py` ->
  `results/e7_profile_*.json`, six models): dense Qwen 1.7B/4B/8B single-layer
  restatement capture is 0% at EVERY band layer (vs 71/59/70% full-band);
  Gemma-2-2B peaks at 32.1% (L14); Gemma-2-9B holds a 21-43% mid-band plateau
  (peak L18) plus an 11% late tail; Qwen3-14B is near-zero with a 2-5% flicker
  at L33-38 (high-drift late layers, suspected output-surface leakage).
  Methodological warning: single-layer ablations alone would misread Qwen as
  "the swap does not work".
- Drift, 14B (07-13, `results/e7_drift_qwen14b.json`): monotone 0.17 -> 0.86,
  ~0.81 at fractional depth 0.8 (near 1.7B's 0.83, far above 4B's 0.66),
  matching 14B's 8.9% late-band capture.
- Persistence (07-14, `run_e7_persistence.py`, generation-free, three models):
  the predicted "healing away" of single-layer Qwen edits is FALSIFIED.
  Injected at any band layer the edit survives to the last layer (survival
  coefficient 0.72-1.18, same order as Gemma) and its relative magnitude grows
  up to 10.5x; what differs is alignment (final-layer cosine +0.05 on
  Qwen3-1.7B vs +0.20 on Gemma-2-9B at matched depth). The edit lives; it is
  not read out.
- Gate test (07-14, `run_e7_gatetest.py`, alpha ladder {1,2,4,8,16},
  Qwen3-1.7B L17/L8 + Gemma-2-2B L16 control): the "closed" single-layer gate
  is QUANTITATIVE, not structural. L17: capture 0% at alpha=1 -> 33.9% at 4 ->
  62.5% at 8 (above Gemma's single-layer peak, matching Qwen's own full band)
  -> 16.1% at 16, where failures are coherent overshoots to a third country,
  not broken text. L8 peaks at 67.9% at alpha=4 despite harvesting half of
  L17's amplitude: unit amplitude is more effective early, consistent with the
  drift ordering. (On record: the script's old verdict heuristic printed
  STRUCTURAL for non-monotone curves; judge by the JSON curves.)
- Amplitude ladder (07-14, `run_e7_amplitude.py` ->
  `results/e7_amplitude_*.json`, incl. states/currency variants): per-layer
  harvested amplitude differs 4-8x between families on the Frobenius reading;
  an earlier per-position-mean regression that had "excluded amplitude"
  (partial R^2 = 0.0088) was an artifact of the wrong norm, and the gate
  test's causal ladder settles the question.
- Report-grader fix (07-14, `run_e7.py`): yes/no margins had been read at
  generated position 0, which is the decision token only for models that
  answer immediately (1.7B/4B/8B). 14B opens with whitespace, Gemma with a
  newline, Qwen3.6-27B with a think block. Fixed to a decision-index reader
  (skips whitespace and think blocks; position-0 readings kept as `*_pos0`
  for audit); all E7 self-report numbers re-extracted, old files kept as
  `*.marginfix.bak.json`. Net effect: 4B's edit-specific channel survives
  unchanged; 14B/Gemma apparent "denial" reclassified as non-answering
  (clean-arm non-answer rates: 14B 100%, Gemma-2-9B 92.9%, Gemma-2-2B 83.9%,
  1.7B 21.4%, 8B 7.1%, 4B 1.8%).
- Sharded-swap verification (07-14, `verify_sharding.py`, A100 pair,
  device_map=auto): a functional-equivalence check that the edit actually
  applied on a model split across two cards matches the analytic expectation
  c*(d_b - d_a) - all 45 Qwen3.6-27B band layers pass with per-layer cosine
  >= 0.9993 and residual <= 3.8% (bf16 noise level), directions matching a
  CPU recomputation to 1e-6. This gated every multi-GPU run. Lesson on
  record: the first version demanded bitwise CPU/GPU equality, which is the
  wrong standard (float32 summation order alone differs by ~5e-8); test
  functional equivalence of the applied edit instead.
- Corrected Qwen3.6-27B five-arm E7 (07-14, post grader-fix and sharding
  gate, `results/e7_perspectival_qwen36-27b.json`, supersedes the first-pass
  file): 56/56 baseline ok (no tokenization exclusions); capture 94.6% /
  flip 96.4% / decodable 100%. The report channel is alive (the none arm
  answers a fluent "No." in 89.3% of trials) and the decision-token margin is
  edit-specific: none -1.84 / randdir -2.08 (control moves the other way) /
  full -0.93, AUC vs randdir 0.794, while the explicit yes token appears 0/56
  on every arm. This kills both alternative readings of the 27B silence
  ("channel dead" and "does not understand the question"): the detection
  signal is measurably present in the channel's own logits and never
  surfaces as words.
- Verdict: three earlier mechanism readings died in these two days (the edit
  is healed away; capture is irreducibly cumulative; the family gate is
  structural) and the amplitude-threshold reading replaced them. All three
  retirements fed the design of the zone-map campaign below.

## 2026-07-16 E7 phrasing sweep (Qwen 1.7B/4B/8B on 4090, 14B on A100): the report channel comes and goes with the wording; no phrasing x scale-stable channel

- Model / config: new public `run_e7_phrasing.py` (decision-token grading via
  `make_report_grader`), five report phrasings incl. one REVERSED-polarity
  probe ("did everything seem normal?" - noticing means answering no), arms
  none/full/randdir, 56 capitals items, answer/restate stages shared per item
  via KV snapshot. Outputs `results/e7_phrasing_{model}.json`; CIs added to
  `bootstrap_ci.json` (new `phrasing_block`).
- Results (paired margin deltas vs none, * = 95% CI excludes 0):
  - 1.7B: full-arm flat on every phrasing (max |0.42|), randdir swings up to
    +/-2.5 by wording alone - under perturbation the probe is noise.
  - 4B: edit-specific and polarity-AWARE on 4/5 phrasings (up to +1.14*;
    reversed probe correctly moves toward "not normal", -0.94*), randdir
    opposite on each. The one scale where detection behaves like detection.
  - 8B: orig wording nonspecific (full +0.47* vs randdir +0.57*), but the
    reversed probe uncovers an edit-specific channel (-1.38* vs +0.07).
  - 14B: margin pushed toward "no" on ALL five wordings (3/5*), randdir the
    other way on 4/5 - polarity-BLIND, an edit-specific response bias rather
    than semantic report; explicit-yes specificity is wording-bound
    (23% vs 5% on orig, 0% on "tamper").
- Verdict: the single-phrasing objection to Section 6 is closed; the channel
  instability claim strengthens (rebuilt per scale AND per wording). Paper
  Section 6 gains a "Phrasing robustness" paragraph; Design/abstract/intro/
  interpretation/Limitations updated. Note: the old internal `_e8_phrasing_*`
  files (2026-07-13) predate the report-grader fix and are superseded.

## 2026-07-16 E6 competence reanalysis (no model runs): the Gemma efficacy gap is NOT explained by target-language knowledge

- What was run: `analyze_e6_competence.py` -> `results/e6_competence.json`,
  pure reanalysis of stored E6 runs. Every E6 item exists in both languages,
  so the opposite-language variant's clean baseline is an independent
  competence measurement for the flip target.
- Results (register@0.125): within every model, full flips concentrate on
  target-known items at ~2x the rate of the rest (Gemma 27.8% vs 16.7%;
  Qwen 62.0%/71.9% vs 30.8%/33.3%). But coverage of target-known items among
  flip-eligible ones is flat across families (Gemma 75.0% vs Qwen
  79.4%/81.0%), and on the target-known subset alone Gemma still full-flips
  27.8% (15/54) vs Qwen 62.0%/71.9%, with the bare language flip also weaker
  there (75.9% vs 96.0%/95.3%).
- Verdict: the paper's "capability-coupled" reading (inferred, flagged as
  unmeasured) is refuted in its knowledge form and replaced: competence gates
  retrieval item-by-item within a model, but the cross-family gap belongs to
  the register write itself. Paper abstract/intro/Section 5/discussion
  rewritten accordingly.

## 2026-07-16 Numbers audit v2 (no new runs): paper prose synced to the 07-14 report-grader fix; bootstrap E7 block now baseline-filtered

- What was run: a full-paper numbers pass (internal consistency, derived-quantity
  recomputation, artifact cross-check) over the paper draft (v2) against
  `results/*.json`. No model runs.
- Findings and fixes:
  - The 2026-07-14 E7 report-grader fix (`make_report_grader`: yes/no margin
    read at the decision token instead of position 0; old files kept as
    `*.marginfix.bak.json`) had refreshed the artifacts, `bootstrap_ci.json`,
    appendix C, and the figures, but the paper prose still quoted the
    pre-fix margins in six places (abstract-adjacent intro bullet, 1.7B/4B/8B/14B
    results paragraphs, Limitations). All synced to the current artifacts:
    1.7B delta -0.04 [-0.41, +0.33]; 4B +1.12 [+0.82, +1.41], randdir -0.85;
    8B +0.47 vs randdir +0.57 (control now nominally larger - the nonspecific
    reading strengthens); 14B -0.72 [-1.04, -0.41], randdir +0.09. No
    qualitative conclusion changes. The margin read position is now defined
    in the E7 design paragraph.
  - `sensitivity_reanalysis.json` was still pre-fix; re-ran
    `experiments/sensitivity/reanalyze.py`. 4B drop-top-3 is now +1.12 -> +0.98;
    the 8B randdir shift is still spread over 54/56 items.
  - `run_bootstrap.py` `e7_block` did not filter `baseline_ok` (every other
    block does); only Qwen3.5-9B has n_ok=51<56, so only its E7 CIs were on a
    mixed denominator. Filter added, bootstrap re-run, appendix regenerated:
    full restate 98.0% [94.1, 100], late-band capture 62.7% [49.0, 76.5] (n=51).
  - Six smaller prose-vs-artifact fixes: Qwen3.5-9B rawdir contrast now quotes
    the lens-arm flip (64.7%), not its restatement rate; "currency commits last"
    recounted as 7 of the 8 measured models (Gemma-2-27B has no currency
    deadline, n=0); drift-wall count recounted as 6 of the 7 models with a
    cutoff (Gemma-2-2B's flip never crosses the threshold); E6t echo range
    0-3 items (1.7B alpha=0.25 has 3); Qwen3.6-27B L40-L46 tail flips
    75.0-80.4%; Gemma-2-2B randdir restatement stated as at-floor (3.6% vs
    5.4% none), not zero.
- Verdict: v2 numbers now audited against artifacts end to end; draft note
  and \date bumped to v2.

## 2026-07-16 E7 randdir window null (Qwen3-8B, Gemma-2-9B, 4090): random-direction width-6 windows do nothing anywhere - zone-map writability is direction-specific

- Model / lens config: new E7_DIR=rand mode in `run_e7_kwin.py` - same
  window geometry (K=6, stride 2), same amplitude-harvesting transfer as
  run_e7's randdir arm, seeded random unit directions per band layer
  (E7_SEED=1), capitals, 56/56 ok on both models.
- Results: Qwen3-8B - 0.0% flip and 0.0% restate at every one of 10
  windows and the full band (its real map: plateau ~50%, full 82.1%).
  Gemma-2-9B - 0.0% flip at all 12 windows and full band; restate max
  7.1% = exactly its none-arm floor (full arm 5.4%, below floor); its
  real map runs 93-98%.
- Verdict: the width-6 coordination effect is specific to the
  lens-transported entity directions, not to any coordinated
  multi-layer perturbation at matched dose. Null established on the
  strongest dense-Qwen and Gemma maps.
- Agreement with the external review: n/a (internal).
- Next: none; control set for the zone-map family complete.

## 2026-07-16 E7 zone map + healing (Qwen3-14B, A100): the dense ladder's mid rung lands on the 0.6-0.7 drift wall and posts the strongest flip-free tail in the table (restate 44.6% at drift 0.88, floor 0)

- Model / lens config: Qwen/Qwen3-14B (40 layers), slide mode
  (E7_SLIDE_K=6, stride 2), band L11..L38, cuda:0 on the A100, 56/56 ok,
  capitals. Healing curve over the same band. Final stages of chain #2.
- Results (flip / restate; none 0/0, full 80.4/66.1): s11 26.8/12.5,
  s13 30.4, s15 30.4, peak s17 33.9/8.9, s19 25.0, cliff s21 17.9
  (window mean drift 0.51) -> s23 8.9 (0.61) -> s25 1.8 (0.70); then the
  tail REVIVES in restate only: s29 8.9/19.6, s31 8.9/41.1,
  s33 7.1/44.6 (drift 0.81-0.88).
- Reading:
  - Cutoff sits squarely in the 0.6-0.7 drift range, like 1.7B/4B/8B/
    Gemma-2B/Gemma-9B and q27 (0.51->0.66). Eight of nine models now cut
    off at window drift ~0.5-0.7; Gemma-2-27B stays the sole exception
    (dead by 0.21, but its whole band only reaches 0.42 - the candidate
    reading is that its deadline arrives before the drift wall, testable
    with a g27 deadline run).
  - Onset mildly suppressed (first window 0.33 of full; cf. 8B 0.46,
    q27 0.09, 4B interior peak). Plateau is the lowest of the open dense
    models (peak flip 33.9% vs full 80.4%).
  - The s29..s33 tail is the strongest flip-free capture measured so
    far: restate 44.6% against a 0.0% none floor while flip stays under
    9%. "Answer dead, workspace writable" at maximum contrast.
- Verdict: internal follow-up; dense-Qwen ladder mid rung consistent
  with the zone-geometry framework.
- Agreement with the external review: n/a (internal).
- Next: deadline runs for 14B and the two 27Bs (g27 is the sharp one).

## 2026-07-16 E7 healing curves, A100 trio + 14B: the pre-registered g27 prediction FAILS - early transmission is 4B-tiny yet the width-6 onset is wide open, so the "onset = healing filter" leg is demoted

- Model / lens config: `run_e7_healing.py` on gemma-2-27b, Qwen3.6-27B,
  Qwen3.5-9B, Qwen3-14B (A100); same metrics as the 07-15 healing entry.
- Results (mean dlogit_ab over the first third of the band, raw / per
  unit injected rel norm): g27 0.31 / 46, q27 0.73 / 34,
  3.5-9B 5.64 / 478, 14B 3.18 / 116. For reference from 07-15/16:
  1.7B 9.06 / 536, 2B 5.44 / 140, 9B 6.40 / 68, 8B 2.66 / 213,
  4B 0.80 / 45. surv_rel mean > 1 in all nine models, per-write-layer
  minima 0.8-2.5 (the 07-15 "always > 1" was exact only for the first
  three models): nothing approaches norm erasure anywhere.
- Reading:
  - The prediction logged in the g27 zone-map entry ("healing_g27
    should show mid-range early transmission, not 4B-tiny") is
    FALSIFIED: g27 has the smallest raw early transmission in the whole
    table (0.31) and per unit injection it ties 4B (46 vs 45) - yet its
    width-6 map opens at the first window (0.73 of full, no onset
    suppression). A model can filter single-layer writes as hard as 4B
    and still accept coordinated width-6 writes from the band start.
  - What survives: the two models with strong width-6 onset suppression
    (4B first-window/peak 0.05, q27 0.10) do have the lowest per-unit
    transmission (45, 34). Low k=1 transmission looks necessary but not
    sufficient for onset suppression. The onset leg of the
    zone-geometry framework is demoted from "onset = healing filter
    strength" to "correlated, mechanism unresolved"; deciding it needs
    dose-aware healing (k>1 writes tracked downstream), not more k=1
    curves.
- Verdict: pre-registered prediction falsified; plateau and cutoff legs
  of the framework unaffected.
- Agreement with the external review: n/a (internal).
- Next: k=6 healing variant if the onset mechanism is pursued.

## 2026-07-16 E7 raw-direction controls complete: basis necessity is absolute in Qwen (hybrid included) and graded by scale in Gemma - g27 flips 42.9% with raw unembedding rows

- Model / lens config: `run_e7_rawdir.py` full per-layer profile on
  Qwen3.5-9B (cuda:0), anchors-only full band on gemma-2-27b (auto).
- Results: 3.5-9B raw W_U full band 0.0 flip / 2.0 restate (lens arm:
  98.0 flip), per-layer profile flat zero at all 22 layers. g27 raw
  full band 42.9 / 62.5 vs lens 66.1 / 66.1 (none floor 0 / 10.7), i.e.
  ~65% relative flip efficacy. Previously logged: 2B 0 at every depth
  and dose, 9B 50.0 vs 98.2 (~51% relative), 1.7B 1.8 vs 78.6,
  8B 1.8 flip but 41.1 restate.
- Reading: clean family split. In Qwen, dense or hybrid, raw
  unembedding rows never move the answer (<=1.8% everywhere tested);
  the lens transport is causally necessary. In Gemma the raw-direction
  efficacy grows with scale: 0% (2B) -> ~51% (9B) -> ~65% (27B)
  relative to the lens arm. Gemma's writable-band residual stream
  becomes increasingly logit-lens-aligned with scale; Qwen's never is.
  The basis-necessity claim must be stated per family, not globally.
- Verdict: internal follow-up; lens necessity strict in Qwen, graded in
  Gemma.
- Agreement with the external review: n/a (internal).
- Next: none planned; control set complete for the current model pool.

## 2026-07-16 E7 deadline, second pair: Qwen3-8B is the cleanest currency-last case yet (+6 layers after capitals); Qwen3.5-9B is the first ordering exception (currency 0.4 layers BEFORE states, sub-resolution, n=12)

- Model / lens config: `run_e7_deadline.py`; Qwen3-8B on the 4090,
  Qwen3.5-9B on the A100, all three domains per model.
- Results (mean commit layer, capitals / states / currency):
  8B (36L): 27.9 (0.77) / 30.8 (0.86) / 33.9 (0.94), n=56/36/20.
  3.5-9B (32L): 26.2 (0.82) / 28.3 (0.88) / 27.9 (0.87), n=36/35/12
  (capitals skip 20, currency skip 17).
- Reading:
  - 8B is textbook: currency commits 6.0 layers after capitals, the
    largest gap measured, matching its zone-map currency late-tail
    revival point for point.
  - 3.5-9B breaks the currency-last ordering for the first time - but
    the states-currency gap is 0.4 layers, under the width-6 stride-2
    instrument resolution, and currency rests on n=12 with 17 skips.
    By the precedent set in the dense-Qwen caveat (deadline entry
    below), scored as unresolved, not as a contradiction. It is still
    the first exception and stays in the ledger.
  - Coverage now 6/9 models. Missing: 14B, g27, q27. The g27 run is
    the sharp falsifiable one: its zone map cuts off at frac depth
    ~0.55 while every measured deadline sits at 0.77-0.94, so the
    deadline reading of its early cutoff predicts an unusually early
    commit layer (~0.5-0.6 frac). Pre-registered here.
- Verdict: deadline model 5-of-6 orderings consistent, 1 unresolved;
  third evidence line intact.
- Agreement with the external review: n/a (internal).
- Next: deadline for 14B / g27 / q27 (g27 first).
- SAME-DAY UPDATE (chain #3, after fixing a device_map=auto bug in
  `run_e7_deadline.py` - final_norm and W_U can sit on different cards):
  - 14B: capitals 32.3 (0.81) / states 34.2 (0.86) / currency 36.4
    (0.91), n=56/36/16. Currency-last again; 7-of-8 with one unresolved.
  - g27: capitals 33.8 (0.73) / states 35.9 (0.78) / currency n=0 (the
    Gemma currency-emission problem at its worst). The pre-registered
    prediction is a PARTIAL HIT: direction confirmed - g27 has the
    earliest capitals commit of all eight models measured (0.73; the
    rest sit 0.77-0.93), matching its uniquely early zone cutoff - but
    the magnitude missed (predicted ~0.5-0.6, actual 0.73).
  - Cross-model rank test, cutoff frac (25%-of-peak crossing, window
    centre) vs deadline frac, n=6 models with both: Spearman 0.829,
    exactly at the two-sided 0.05 critical value for n=6. Deadline
    frac sits 0.05-0.19 above cutoff frac everywhere (mean 0.12) -
    consistent with a width-6 write needing to land wholly before the
    commit, but the g27 gap (0.19) is the largest, so "cutoff layer =
    commit layer" is too strong. The deadline model is ordinal, not
    metric: it orders cutoffs across models (rho 0.83) and within
    models (domain orderings), it does not pin their positions.
  - q27 first run returned zero usable items in all domains (the base
    model never emits the answer token within 12 greedy steps, the
    same verbosity that needed E7_NANS=24 on its zone map); rerunning
    with E7_DEADLINE_STEPS=24.
  - q27 retry (E7_DEADLINE_STEPS=24): capitals 59.0 (0.92, n=5,
    skip 51) / states 60.3 (0.94, n=3) / currency 60.6 (0.95, n=16).
    PROVISIONAL - even at 24 steps the base model rarely emits the
    answer token, so capitals/states rest on n=5/3. Ordering is
    currency-last again (8-of-9 with one unresolved). Adding q27 to
    the rank test: n=7, Spearman(cutoff, deadline) = 0.857, above the
    0.786 two-sided 0.05 critical value; per-model gap 0.05-0.19
    (mean 0.12). Deadline coverage complete at 9/9 (two provisional:
    q27 small-n, Gemma currency small-n).

## 2026-07-16 E7 zone map (Qwen3.6-27B, A100): the "closed" top hybrid is wide open at width 6 - a 28-layer plateau at 70-87% - and the open/closed gate taxonomy dissolves for good

- Model / lens config: Qwen/Qwen3.6-27B, slide mode (E7_SLIDE_K=6,
  stride 2, E7_NANS=24), band L18..L62 (45 layers), device_map=auto,
  56/56 ok, capitals. Chain #2 stage 2.
- Results (flip / restate; none 0/0, full 96.4/94.6): s18 8.9/5.4,
  s20 44.6/30.4, s22 69.6/53.6, s24 78.6/60.7, then a broad plateau
  through s46 (70-87% flip, peak s36 87.5), s48 35.7/5.4, s50 7.1/1.8,
  s52+ 0. Cliff at window mean drift ~0.51->0.66 (centre frac depth
  ~0.79).
- Reading:
  - The single-layer profile's faint L40..L48 cluster (peak 3.6%) sat on
    top of a 28-layer-wide writable plateau. At width 1 this model is
    "closed"; at width 6 it flips 87%. With all nine models mapped, every
    single one is writable somewhere at width 6: the open/closed gate of
    the profile era is entirely a dose artifact. The honest taxonomy is
    zone geometry: onset (healing filter strength), plateau height
    (write fidelity), cutoff (task deadline, drift-ordered within model).
  - Onset suppression is visible (s18 8.9% at drift 0.11 - lowest-drift
    window in the whole study, yet weakest in-zone window in this model),
    consistent with 64-layer early filtering, though much milder than
    4B's.
  - New: the REVERSE dissociation inside one map - s40..s46 keep flip at
    75-80% while restate falls from ~60% to ~20%. Late-mid writes change
    the answer but the restatement stage stops reporting the swapped
    entity. Together with the flip-free capture at other models' tails,
    both dissociation directions now exist; answer and restatement read
    the workspace at different depths, in different orders per
    architecture.
- Verdict: internal follow-up; gate taxonomy retired, zone geometry is
  the framework. Nine of nine models writable at width 6.
- Agreement with the external review: n/a (internal).
- Next: chain #2 continues (9B rawdir, healings, 14B); morning synthesis
  for the paper narrative.

## 2026-07-16 E7 deadline measurement (logit lens, no intervention): currency commits 3-4 layers later everywhere, and the Gemma-2-2B "miss" turns out to be a correct prediction of coinciding deadlines

- Model / lens config: new `run_e7_deadline.py` - greedy-decode each clean
  prompt until the answer's first token is about to be emitted, then read
  the logit lens (final norm + unembedding) at that position across all
  layers; commit layer = earliest layer from which the answer token stays
  argmax. No interventions anywhere. Local MPS, all three domains per
  model in one load.
- Results (mean commit layer; capitals / states / currency):
  Qwen3-1.7B (28L): 22.2 / 24.2 / 25.9 (n=56/36/20)
  Qwen3-4B (36L): 29.9 / 31.5 / 33.2 (n=56/35/15)
  Gemma-2-2B (26L): 24.2 / 24.2 / 24.8 (n=56/36/4)
  Gemma-2-9B (42L): 36.7 / 35.5 / 40.0 (n=56/36/3)
- Reading:
  - Currency commits last in all four models, by 3-4 layers - the only
    deadline gap large enough for a width-6 stride-2 zone map to
    resolve, which is exactly why the zone-map tails single out currency
    in every model while capitals-vs-states stays murky (their gaps are
    1-2 layers, under the instrument's resolution).
  - Gemma-2-2B capitals and states deadlines are EQUAL (24.2 = 24.2),
    which retroactively explains the "miss" logged one entry down: the
    two domains' cliffs sit on the same window (35.7% vs 36.1% at s19)
    because their deadlines coincide. The deadline model predicted that
    equality; it was scored as a miss before the deadline was measured.
  - Gemma-2-9B full ordering states < capitals < currency matches its
    zone-map cutoff ordering point for point.
  - Dense-Qwen caveat: 1.7B/4B measure capitals BEFORE states while
    their states zone maps decline faster - but their states full-band
    flip is also lower (write fidelity), and the deadline gap is under
    resolution; scored as unresolved, not as a contradiction.
  - Small-n caveat: Gemma currency deadlines rest on n=3-4 (the Gemma
    models rarely emit the currency answer token within 12 greedy
    steps); treat those two numbers as provisional.
- Verdict: internal follow-up; the consumption deadline is now an
  independent, intervention-free measurement that co-varies with the
  zone-map tails where resolvable. Third evidence line for the
  two-force-plus-deadline model.
- Agreement with the external review: n/a (internal).
- Next: deadline for 8B (4090) and the A100 trio; consider width-6
  stride-1 zone maps around the deadline layers if finer resolution is
  ever needed.

## 2026-07-16 E7 task-deadline model: both pre-registered 1.7B predictions hit (states steeper, currency tail revives at the last window); Gemma-2-2B is the partial miss

- Model / lens config: Qwen3-1.7B states (36/37 ok) + currency (27/29 ok)
  on the 4090; Gemma-2-2B states (36/37) + currency (29/29) on MPS. All
  none floors 0 (one 3.4% single-item currency flip on 2B).
- Results (flip):
  - 1.7B (capitals / states / currency): s8 76.8/66.7/51.9,
    s12 50.0/30.6/37.0, s16 25.0/8.3/18.5, s18 5.4/2.8/7.4,
    **s20 0.0/0.0/29.6**. The previous entry's predictions were logged
    before these runs: "monotone decline steepens for states" - yes
    (states falls to 8.3% by s16 where capitals holds 25.0) - and
    "stretches for currency" - yes, spectacularly: the last window,
    dead in both other domains, flips 29.6% for currency.
  - Gemma-2-2B (capitals / states / currency): ceiling until s17 in
    both capitals and states, and the s19 drop is IDENTICAL
    (35.7 vs 36.1); currency shows a lower plateau (86-90%), earlier
    sag (s15 72.4, s17 58.6) but a higher tail (s19 51.7).
- Reading: the consumption-deadline block-shift is now confirmed in
  1.7B, 4B, 8B, Gemma-2-9B; the currency-latest ordering holds in all
  five models with usable tails. The 2B states prediction missed: its
  18-layer band likely ends before the states/capitals deadlines
  separate (both cliffs sit at the same last window), while currency's
  softer profile suggests a lower-fidelity write for that pool rather
  than a deadline effect - 2B is the weakest reader of the deadline
  story, not a refutation, but it is recorded as the miss it is.
- Verdict: internal follow-up; task-deadline model 4-for-5 with one
  pre-registered double-hit.
- Agreement with the external review: n/a (internal).
- Next: q27 zone map due (A100); after chain #2, consider a
  deadline-measurement experiment (logit-lens style: at which layer does
  the correct answer's logit commit per task?) to close the loop
  quantitatively.

## 2026-07-16 E7 Qwen3-4B three-domain zone maps: the whole zone shifts as a block with the task - and the "zone position is architectural" reading of the first currency check is corrected

- Model / lens config: Qwen/Qwen3-4B, slide mode, states pool (34/37 ok,
  none floor 0/0), local MPS; compared against the capitals and currency
  maps already logged.
- Results (flip, capitals / currency / states): s10 1.8/6.9/17.6,
  s12 1.8/6.9/14.7, s14 3.6/13.8/17.6, s16 7.1/17.2/26.5,
  s18 25.0/20.7/32.4, s20 33.9/24.1/32.4, s22 28.6/20.7/20.6,
  s24 10.7/6.9/2.9, s26 7.1/13.8/0.0, s28 3.6/13.8/0.0;
  full 76.8/79.3/52.9.
- Reading:
  - The states zone is left-shifted as a whole (early windows viable at
    15-18% where capitals gives 1.8%; tail dead from s24) and the
    currency zone is right-extended - the same task ordering as
    Gemma-2-9B (states commits earliest, currency latest), now visible
    in a closed dense model, and now moving BOTH edges.
  - Correction of the 2026-07-15 currency-check entry: "the writable
    zone is a property of the model, not of the task" was too strong. It
    held because capitals and currency have nearby consumption depths at
    the peak; states breaks it. The refined two-force statement that
    fits all maps: healing filters the write over the span from write
    layer to the TASK'S consumption depth (not to the final layer), and
    drift/timing kills writes landing after that depth. An earlier
    consumption point therefore shifts the whole viable window left -
    which is exactly the states map. Architecture sets the filtering
    strength; the task sets the deadline.
- Verdict: internal follow-up; the block-shift is the strongest
  structural evidence yet for the consumption-deadline model.
- Agreement with the external review: n/a (internal).
- Next: 1.7B states/currency (4090, running) - prediction: its monotone
  decline steepens for states and stretches for currency.

## 2026-07-16 E7 Qwen3-8B follow-ups (4090): raw directions reach the restatement but never the answer, and the currency late-tail replicates

- Model / lens config: Qwen/Qwen3-8B on the 4090. (a) rawdir anchors-only,
  capitals. (b) slide-mode zone map on the currency pool (27/29 ok).
- Results:
  - Raw-W_U full band: flip 1.8% (lens: 82.1%) but restate_swapped 41.1%
    (none floor 0/0). The raw basis at 25-layer dose writes something the
    restatement stage reads back in 4 of 10 items while the answer never
    moves - the flip/restate dissociation again, this time produced by
    the direction basis instead of window position. The answer pathway is
    strictly basis- and timing-gated; the restatement pathway is
    permissive to both.
  - Currency zone vs capitals (flip): plateau lower (18-26% vs 37-50%;
    currency retrieval is weaker in 8B overall) and the late tail fatter
    (s26/s28: 7.4/18.5 vs 1.8/3.6, floors 0). Same direction as the
    Gemma-2-9B domain shift: currency commits later.
- Verdict: internal follow-up; task-dependent cutoff now in three models
  (4B, Gemma-2-9B, 8B), restate-permissiveness now in two manipulations.
- Agreement with the external review: n/a (internal).
- Next: 1.7B states + currency zone maps (4090) for the task ordering on
  the monotone-shape architecture.

## 2026-07-16 E7 Gemma-2-9B follow-ups: raw-W_U is weak-but-nonzero at dose in the 9B (basis story nuanced), and the late cutoff MOVES with the task domain (timing story strengthened)

- Model / lens config: google/gemma-2-9b, local MPS. (a) rawdir
  anchors-only (E7_RAWDIR_ANCHORS=1), capitals. (b) slide mode zone map
  on the currency pool (29 items).
- Results:
  - Raw-W_U full band (29 layers): flip 50.0%, restate 33.9% - versus
    98.2 / 78.6 with lens directions at identical dose, and versus
    1.8-3.6% raw in Qwen3-1.7B / Gemma-2-2B. The categorical "raw
    directions do not couple at all" (two entries down) is falsified at
    the 9B scale: raw is heavily attenuated (half the flip, less than
    half the restate) but not dead. Lens transport remains much the
    stronger basis everywhere tested; its necessity is strict only in
    the smaller models.
  - Currency zone map vs capitals (flip): plateau matches (93-97% vs
    94-98% over s12..s28), but the tail diverges hard: s30 75.9 vs 73.2,
    s32 58.6 vs 25.0, **s34 55.2 vs 8.9** (currency none floor 3.4%,
    single item). The late cutoff is domain-dependent in this model.
- Reading: the writable plateau is architectural (matches across
  domains, as in the 4B check), but the late boundary tracks the task:
  currency retrieval commits later than capitals retrieval, so late
  windows stay effective longer. This is what the timing reading
  predicts and what a pure-architecture reading forbids. The 4B
  cross-domain check (peak position stable) is consistent once read as
  "onset/peak architectural, cutoff task-dependent" - 4B's currency map
  also shows a fatter late tail (s26/s28 13.8/13.8 vs capitals 7.1/3.6),
  which I under-read at the time as noise.
- Verdict: internal follow-up; basis result nuanced at scale, consumption
  timing is task-dependent.
- Agreement with the external review: n/a (internal).
- Next: states-domain zone map on Gemma-2-9B to triangulate the cutoff
  shift across three tasks on one model.
- SAME-DAY UPDATE (states map landed, 36/37 ok, none floor 0/0): the
  triangulation is clean. Plateau identical across all three domains
  (94-97%); tail flip at s30/s32/s34: states 36.1/8.3/0.0, capitals
  73.2/25.0/8.9, currency 75.9/58.6/55.2. Three tasks, three cutoffs,
  one strict ordering (states < capitals < currency), one model. The
  answer-consumption depth is a property of the TASK within a fixed
  architecture; "drift kills late writes" cashes out as "late relative
  to when this particular retrieval commits".

## 2026-07-16 E7 zone map (Gemma-2-27B, A100): the single-layer closure is a right-shifted dose curve as predicted - but the "universal drift cutoff" of the previous entry dies on its sixth data point

- Model / lens config: google/gemma-2-27b, slide mode (E7_SLIDE_K=6,
  stride 2), band L13..L44 (46 layers), device_map=auto on the A100 pair,
  56/56 baseline ok, capitals. First stage of chain #2.
- Results (flip / restate; none 0/10.7, full 66.1/66.1): s13 48.2/57.1,
  s15 44.6/57.1, s17 33.9/41.1, s19 26.8/25.0, s21 16.1/16.1,
  s23 10.7/7.1, s25 0/0, s27 0/0, s29 7.1/3.6, s31 7.1/1.8, s33 1.8/5.4,
  s35 0/3.6, s37 0/10.7, s39 0/37.5. Window Spearman -0.87 (n=14).
- Reading:
  - Prediction confirmed: the model whose 46-layer stack closes every
    single layer (0 flips at k=1) writes fine at width 6 (48.2% at the
    band start). "Closed" = dose threshold shifted right, not unwritable.
    The two-force model needed this and got it.
  - Prediction half-wrong: the expected 4B-style interior peak is not
    there - the shape is a 1.7B-style monotone decline from the band
    start. So deep-stack closure does not imply 4B-style early filtering;
    the healing_g27 curve (later tonight) should show mid-range early
    transmission, not 4B-tiny. Falsifiable within hours.
  - The drift-cutoff "regularity" logged one entry ago breaks: g27's
    flip is dead by window mean drift ~0.25-0.34 (its whole band only
    reaches drift 0.42), far below the 0.6-0.7 of the five smaller
    models. Fractional-depth cutoffs are not universal either (25%-of-
    peak crossing: g27 0.55, 8B 0.68, 1.7B 0.73, 4B 0.79, 9B 0.87,
    2B none in band). Honest statement: the answer-consumption point is
    model-specific; drift orders writability WITHIN a model (all six
    maps) but carries no universal threshold ACROSS models. The
    cross-model story stays qualitative (two forces), not quantitative.
  - Tail flip-free capture yet again: s39 restate 37.5% vs flip 0
    (restate floor 10.7%, so ~3.5x floor).
- Verdict: internal follow-up; dose-shift confirmed, universal-threshold
  reading retracted same-night before it hardened.
- Agreement with the external review: n/a (internal).
- Next: q27 zone map (running); healing g27 to test the mid-range
  early-transmission prediction above.

## 2026-07-16 E7 zone map (Gemma-2-9B) + Qwen basis check: the flip cutoff sits at drift ~0.6-0.7 in every model so far, and late windows capture the restatement at full strength while the flip dies

- Model / lens config: (a) google/gemma-2-9b slide mode (E7_SLIDE_K=6,
  stride 2), band L12..L40, 56/56 ok, local MPS. (b) Qwen/Qwen3-1.7B
  `run_e7_rawdir.py` anchors-only (E7_RAWDIR_ANCHORS=1).
- Results:
  - Gemma-2-9B (flip / restate; none 0/7.1, full 98.2/78.6): plateau
    93-98% from s12 through s28 (mean drift 0.20-0.55), then s30 73.2,
    s32 25.0, s34 8.9. Restate across the same tail: 73.2 / 76.8 / 67.9 -
    it does not drop at all. Window Spearman -0.52 (ceiling ties).
  - Qwen3-1.7B raw-W_U full band: flip 1.8% (vs 78.6% with the lens
    directions at identical dose). The basis result replicates across
    families: raw unembedding rows do not couple into the computation.
- Reading:
  - Fifth zone map. Cross-model regularity now visible: the flip cutoff
    lands at window mean drift ~0.6-0.7 everywhere measured - Gemma-9B
    collapses 0.60->0.64, 8B cliffs 0.59->0.70, 4B fades past ~0.6,
    Gemma-2B's only sub-ceiling window is at 0.54, 1.7B reaches zero by
    0.75-0.83. Drift is a per-model depth clock whose answer-consumption
    point sits in a narrow common drift range across sizes and families.
  - The Gemma-9B tail is the cleanest flip/restate dissociation yet:
    late width-6 writes still capture the restatement at plateau level
    (67-77%, floor 7.1%) while the answer flip collapses. The write
    lands in the workspace and is read back by the restatement stage;
    only the already-computed answer is out of reach. This is direct
    behavioural evidence for the timing reading of the drift force, from
    the model with the strongest capture in the table.
- Verdict: internal follow-up; drift-cutoff regularity + timing evidence.
- Agreement with the external review: n/a (internal).
- Next: Gemma-2-9B healing curve locally; A100 chain #2 zone maps for
  the two 27Bs will test the cutoff at 46/64 layers.

## 2026-07-16 E7 single-layer profile (Qwen3.6-27B) lands: the top hybrid rung is (nearly) closed - the nine-model table is complete, and depth beats generation at the top

- Model / lens config: Qwen/Qwen3.6-27B (64 layers, hybrid DeltaNet),
  self-fitted lens, A100 pair (device_map=auto, E7_NANS=24), capitals,
  56/56 baseline ok. Last of the nine profiled models.
- Results: none 0/0, full band 96.4% flip / 94.6% restate. Single-layer
  flips: zero everywhere except a contiguous weak cluster L40..L48
  (0.63-0.75 depth) at 1.8-3.6% - 14 real flip events concentrated in 9
  of 45 band layers (the none arm shows the ans-grade floor is exactly 0,
  so these are genuine flips, just rare).
- Nine-model final table (flip metric): open gates = Gemma-2-2B/9B and
  hybrid Qwen3.5-9B (within-model Spearman flip vs drift -0.774 / -0.824
  / -0.811); closed = all dense Qwen3 and Gemma-2-27B (hard zeros or
  single items); Qwen3.6-27B sits at the boundary with the compressed
  mid-band cluster - which looks exactly like a 4B-style interior
  writable zone squeezed to single-layer scale by a 64-layer stack
  (the chain-2 zone map will test this directly).
- Reading:
  - The "graded generation gate" (gemma > hybrid > dense) survives only
    at matched scale: hybrid 9B is open where dense 8B is closed, but at
    the top rung BOTH 27Bs are closed regardless of family/generation.
    Depth (healing budget) dominates generation.
  - Flip-based pooled regression (243 layer obs, 9 models): drift alone
    R2 0.074, + generation dummies 0.155 (partial 0.088), + amplitude
    0.304. Compare the restate-based generation regression (R2 0.631,
    partial 0.589): most of the between-model "generation effect" on
    restate was floor-and-level structure, not per-layer signal. The
    robust cross-model law remains the within-model drift ordering in
    open models plus the two-force zone structure in closed ones.
- Verdict: profiles complete; the paper's per-layer story should lead
  with within-model drift ordering + zone maps, not the generation split.
- Agreement with the external review: n/a (internal).
- Next: chain #2 on the A100 (both 27B zone maps + healing, 9B rawdir +
  healing, then Qwen3-14B download + slide + healing).

## 2026-07-16 E7 sliding-window zone map (Qwen3-8B, 4090): early plateau with a sharp drift cutoff at ~0.65, and the strongest flip-free late capture yet

- Model / lens config: Qwen/Qwen3-8B, `run_e7_kwin.py` slide mode
  (`E7_SLIDE_K=6 E7_SLIDE_STRIDE=2`), band L10..L34, capitals, 56/56
  baseline ok, RTX 4090 (first jspace run on this box; CUDA path).
- Results (flip / restate-swapped; none = 0/0, full = 82.1/71.4):
  s10 37.5/28.6, s12 39.3/30.4, s14 48.2/28.6, s16 44.6/25.0,
  s18 50.0/23.2, s20 25.0/12.5, s22 0/0, s24 0/1.8, s26 1.8/17.9,
  s28 3.6/37.5. Window mean drift vs flip: Spearman -0.66 (n=10).
- Reading:
  - Fourth zone shape, intermediate: the early band is already writable
    at width 6 (37-50%, weak early filtering unlike 4B) and the zone ends
    in a cliff between s20 and s22 (window mean drift 0.59 -> 0.70). The
    drift cutoff for zero flip is consistent with the other maps
    (1.7B dies by 0.83, 4B fades past ~0.6): the timing force bites in
    the same drift range across dense sizes.
  - Within-family ladder so far: 1.7B monotone decline, 4B interior peak
    (strongest early filtering), 8B early plateau + cliff. Healing
    strength is NOT monotone in model size within dense Qwen3.
  - Flip-free late capture, strongest instance: s28 restate 37.5% with
    flip 3.6% and a none floor of exactly 0. Late writes feed the
    restatement stage hard while leaving the answer untouched - by now a
    reproducible phenomenon (4B s28 21.4/3.6, gemma-27b half2), not a
    curiosity.
- Verdict: internal follow-up; drift-cutoff consistency across dense
  sizes, healing non-monotone in size.
- Agreement with the external review: n/a (internal).
- Next: 8B healing curve on the 4090 (running); 14B zone map needs a
  40G card -> A100 queue behind the 27B jobs.

## 2026-07-16 E7 direction-source control (Gemma-2-2B): raw W_U directions do nothing at any depth or dose - the lens transport is causally necessary, and the late-layer failure is timing, not basis

- Model / lens config: google/gemma-2-2b, new `run_e7_rawdir.py`: the
  single-layer profile re-run with swap directions d = normalize(W_U[t])
  (no J_l^T transport), via DirectionSwapHooks; same staged generation and
  grading, 56/56 baseline ok, capitals, local MPS.
- Results (flip): every single layer 0-1.8% (vs lens direction 7.1-23.2%
  in the writable early band); **full band (18 layers) 3.6% vs lens 94.6%**
  (restate 5.4% vs 83.9%). The raw direction fails even where drift is
  lowest (L7-L14, drift 0.17-0.21, lens flips 7-23%, raw flips 0%).
- Reading:
  - The control was designed to separate "drift = coordinate rotation"
    from "drift = timing proxy"; it came back stronger than designed: the
    unembedding row is simply not the entity's coordinate anywhere in the
    band. The Jacobian transport J_l^T W_U is causally necessary for the
    write to couple into the model's computation at all. This is a direct
    causal vindication of the lens basis (the paper's coordinates are not
    just descriptive), worth stating in any writeup.
  - Combined with the healing measurement (previous entry), the late-layer
    story resolves as timing: lens-basis writes at high-drift layers DO
    arrive at the readout (1.7B dlogit +8..+21, rising with depth) yet do
    not flip, so what dies late is not the injection basis but the
    consumption - the answer-retrieval computation has already run. Drift
    is best read as a depth clock for how much readout-relevant
    computation remains, not as a basis error.
- Verdict: internal follow-up; lens basis necessary (positive result for
  upstream methodology), timing reading of the drift force supported.
- Agreement with the external review: strengthens the "linear-lens
  coordinates are meaningful" side against a pure-correlation reading.
- Next: same control on an open Qwen (qwen35-9b, A100) to check the basis
  result across families; healing curves for the A100 models.

## 2026-07-15 E7 healing curves (direct measurement): no model erases the write - "healing" is selective suppression of the readout-aligned component, ~10x stronger in Qwen3-4B than in its siblings

- Model / lens config: Qwen3-1.7B, Qwen3-4B, Gemma-2-2B; new
  `run_e7_healing.py`: single-layer swap at each band layer, then track the
  induced perturbation Delta h(l') at every downstream layer on the prompt
  forward (no generation). Metrics at the last prompt position: rel norm
  (||Delta h|| / ||h_clean||), and dlogit_ab = change in the
  (logit_b - logit_a) gap at the lm head (readout-level survival of the
  swapped entity axis). Capture hooks registered after the swap hook, so
  the written layer reads post-swap (the v5 pre-hook trap does not apply).
  56 items, capitals, local MPS.
- Results:
  - Norm-healing does not exist in any of the three models: surv_rel =
    final rel norm / injected rel norm is > 1 everywhere (1.7B: 2.3-42x,
    4B: 1.6-3.5x, Gemma-2B: 1.3-5.7x). Perturbations snowball; nothing is
    erased.
  - The real, now directly measured family difference is the readout-axis
    transmission of EARLY writes. Mean dlogit_ab for writes in the first
    third of the band: Qwen3-1.7B ~ 8.4-10.9, Gemma-2-2B ~ 4.5-6.5,
    **Qwen3-4B ~ 0.5-1.0** - an order of magnitude below its 1.7B sibling
    at comparable injected amplitude (e.g. L15: 1.7B inj 0.010 ->
    dlogit 9.6; 4B inj 0.014 -> dlogit 0.47). This is the healing force
    the 4B zone map inferred: 4B's early-band writes are not removed,
    their answer-axis content is filtered out while the energy scatters
    into other directions. 1.7B barely filters at all.
  - Gemma-2-2B tail collapse: dlogit rises to ~11 at L21 (0.81 depth) then
    crashes to 0.13 at L24 - the readout axis becomes unwritable in the
    last band layers, matching its zone-map tail drop and the drift force.
    Qwen dlogit instead keeps rising almost to the end.
  - New dissociation, the biggest surprise: in 1.7B every single-layer
    write survives to the readout (dlogit +8..+21) yet single-layer flips
    are 0% everywhere. Readout-surface survival is NOT sufficient for a
    flip. The flip has to go through the answer-retrieval computation
    consuming the edited entity representation; a swap can move the entity
    axis at the lm head without moving the retrieved capital at all. This
    strengthens the "drift as timing" reading over naive erasure, and is
    exactly what the raw-W_U direction control (next) is built to probe.
- Verdict: internal follow-up; the two-force model survives but "healing"
  must be restated as selective answer-axis filtering, not erasure.
- Agreement with the external review: n/a (internal).
- Next: raw-W_U direction control on Gemma-2-2B (basis vs timing);
  healing curves for the A100 models (does Gemma-2-27B's early filtering
  look like 4B's?).

## 2026-07-15 E7 zone map cross-domain check (Qwen3-4B, currency): the writable zone does not move with the task - peak stays at L20..L25

- Model / lens config: Qwen/Qwen3-4B, `run_e7_kwin.py` slide mode
  (`E7_SLIDE_K=6 E7_SLIDE_STRIDE=2`), currency domain, 29/29 baseline ok,
  local MPS. Same arms as the capitals map (band L10..L34, s10..s28).
- Results (flip, currency vs capitals): s10 6.9/1.8, s12 6.9/1.8,
  s14 13.8/3.6, s16 17.2/7.1, s18 20.7/25.0, **s20 24.1/33.9**,
  s22 20.7/28.6, s24 6.9/10.7, s26 13.8/7.1, s28 13.8/3.6;
  full 79.3/76.8; none 0/0 on both.
- Reading: same interior-peak shape with the peak at the same window
  (s20 = L20..L25) despite a different fact domain and a pool half the
  size (29 items, so one item = 3.4% - the tail wiggle at s26/s28 is
  within noise). The writable zone is a property of the model, not of
  the task content. This is the zone-map analogue of the E7 cross-domain
  replication already logged for gemma-9b/qwen35-9b.
- Verdict: internal follow-up; zone position replicates across domains.
- Agreement with the external review: n/a (internal).
- Next: A100 zone maps once the 27B profile frees the cards.

## 2026-07-15 E7 sliding-window zone map (Gemma-2-2B): a ceiling plateau - any width-6 window in the first two thirds of the band flips 100%, and only the band tail is drift-limited

- Model / lens config: google/gemma-2-2b, `run_e7_kwin.py` slide mode
  (`E7_SLIDE_K=6 E7_SLIDE_STRIDE=2`), band L7..L24, arms s7..s19 plus
  none/full, capitals, 56/56 baseline ok, local MPS.
- Results (flip / restate-swapped; none = 0/5.4, full = 94.6/83.9):
  s7 100/82.1, s9 100/87.5, s11 100/92.9, s13 100/89.3, s15 98.2/89.3,
  s17 91.1/89.3, s19 35.7/42.9. Window mean drift vs flip:
  Spearman = -0.906 (n=7).
- Reading:
  - Third zone-map shape, completing the predicted ordering: Gemma-2B
    (open single-layer gate) is at ceiling for every window except the
    band tail; Qwen3-1.7B declines monotonically from the start; Qwen3-4B
    has an interior peak. Under the two-force reading Gemma-2B has the
    weakest healing (a single layer already writes at 27-46%, six layers
    saturate), so only the drift force is visible, and only where drift
    is largest (s19, mean drift 0.54, drops to 35.7%).
  - Width-6 windows beat the full band (100% vs 94.6%): writing into the
    late high-drift layers on top of a clean mid-band write slightly
    hurts. Consistent with the drift law: late-layer writes inject
    off-coordinate content rather than reinforcing.
  - Unlike Qwen, restate tracks flip closely everywhere (Gemma's
    restatement stage reads back what the answer stage wrote; the
    flip-restate gap is a Qwen-family trait in these maps).
- Verdict: internal follow-up; two-force model now has all three
  predicted shapes (plateau / monotone decline / interior peak) in one
  fixed-width instrument.
- Agreement with the external review: n/a (internal).
- Next: A100 zone maps (Gemma-2-27B: does the closed 46-layer sibling
  show an interior peak like 4B? Qwen3-8B/14B: where do mid-size dense
  zones sit?) once the qwen3.6-27b profile frees the cards.

## 2026-07-15 E7 sliding-window zone map (Qwen3-1.7B): monotone decline from the band start, window-level Spearman -1.00 - the two-force reading gets its first clean confirmation

- Model / lens config: Qwen/Qwen3-1.7B, `run_e7_kwin.py` slide mode
  (`E7_SLIDE_K=6 E7_SLIDE_STRIDE=2`), band L8..L26, arms s8..s20 plus
  none/full, capitals, 56/56 baseline ok, local MPS.
- Results (flip / restate-swapped; none = 0/0, full = 78.6/71.4):
  s8 76.8/73.2, s10 73.2/64.3, s12 50.0/23.2, s14 48.2/17.9, s16 25.0/0,
  s18 5.4/0, s20 0/5.4.
- Reading:
  - No interior peak: the zone starts at the band start and decays
    monotonically. Window mean drift vs flip: Spearman = -1.000 (n=7).
    A width-6 window at the band start recovers essentially the whole
    full-band effect (76.8% vs 78.6% with 19 layers).
  - Contrast with Qwen3-4B (entry below): 4B's profile is an interior peak
    at L20..L25 (window Spearman +0.47, non-monotone); 1.7B's is a pure
    drift-limited decline. Under the two-force reading, 1.7B's downstream
    healing is weak enough that a width-6 write at the band start survives
    (though a width-1 write does not - its single-layer gate is closed),
    while 4B's healing kills everything before ~L18 regardless of width-6
    support. Same family, same "closed" single-layer verdict, different
    force balance - the single-layer profile alone cannot distinguish them.
  - Tail dissociation again, small: s20 restate 5.4% with flip 0.
- Verdict: internal follow-up; supports the two-force (healing vs drift)
  model of writability.
- Agreement with the external review: n/a (internal).
- Next: Gemma-2-2B slide map (open single-layer gate; prediction: monotone
  decline like 1.7B but starting higher); then the A100 models.

## 2026-07-15 E7 sliding-window zone map (Qwen3-4B): the writable zone is a single interior peak - too-early and too-late writes both die, so drift alone cannot be the gate

- Model / lens config: Qwen/Qwen3-4B, `run_e7_kwin.py` in slide mode
  (`E7_SLIDE_K=6 E7_SLIDE_STRIDE=2`): a fixed width-6 window slid across the
  band L10..L34, arms s10..s28 plus none/full anchors, capitals, 56/56
  baseline ok, local MPS.
- Results (flip / restate-swapped by window; none = 0/0, full = 76.8/58.9):
  s10 1.8/0, s12 1.8/0, s14 3.6/0, s16 7.1/0, s18 25.0/7.1, **s20 (L20..L25)
  33.9/7.1**, s22 28.6/8.9, s24 10.7/3.6, s26 7.1/12.5, s28 3.6/21.4.
- Reading:
  - Clean unimodal writable zone centred on L20..L25 (~0.56-0.69 depth).
    This confirms the anchor-confound diagnosis of the entry below: the
    early-anchored sweep was not measuring a width requirement, it was
    measuring the distance from the band start to this zone. "k50 ~ 13-15"
    for 4B was an artifact of anchoring outside the zone.
  - The zone peak sits at window mean drift ~ 0.40 (window-level Spearman
    flip vs mean drift = +0.47, n=10, non-monotone). So within 4B the
    single-direction drift law does NOT hold at window level: flip rises
    with drift up to ~0.4, then falls. Two-force reading: too-early writes
    are healed away by the many layers downstream (dense-stack healing);
    too-late writes land after the readout coordinates have rotated (drift).
    The writable zone is where neither force dominates. Open models
    (Gemma-2B/9B, Qwen3.5-9B) may simply have weak healing, which would
    leave drift as the only visible force and explain their monotone
    negative single-layer Spearman.
  - Reverse dissociation at the band tail: s26/s28 restate-swapped rises to
    12.5/21.4% while flip falls to 7.1/3.6% (and the none arm restate floor
    is exactly 0 here, so this is signal, not grading floor). Late writes
    increasingly nudge the restatement generation without flipping the
    answer - the same flip-free capture seen in Gemma-2-27B's half2 arm.
- Verdict: not a replication item - internal follow-up. The "minimum write
  width" quantity of the two entries below is superseded: width and position
  must be mapped jointly, and the zone map is the honest instrument.
- Agreement with the external review: n/a (internal).
- Next: slide maps for Qwen3-1.7B and Gemma-2-2B (do open models show a
  monotone-declining profile instead of an interior peak?); Gemma-2-27B and
  Qwen3-8B/14B on the A100 once the 27B profile chain frees the cards
  (prediction: 27B's zone, if any, sits mid-band and its single-layer
  closure is the healing force at work).

## 2026-07-15 E7 minimum write width (Qwen3-4B): the universal-width reading dies the same day - width is model-specific, and the anchor confound demands a sliding-window design

- Model / lens config: Qwen/Qwen3-4B, same `run_e7_kwin.py` sweep (early
  anchor = band start L10, k in {2,3,4,6,8,12}, late-6 control L22..L27),
  capitals, 56/56 baseline ok, local MPS.
- Results (flip / restate): e2 0/0, e3 1.8/0, e4 0/1.8, e6 1.8/0,
  e8 10.7/1.8, e12 37.5/14.3, full(25) 76.8/58.9; late-6 28.6/8.9.
- Reading:
  - The "near-universal ~2-3 layer minimum width" of the two entries below
    is falsified by the second dense point: anchored at its band start, 4B
    needs k ~ 13-15 to reach half its full-band flip rate - an order of
    magnitude wider than Qwen3-1.7B (~2.7) and Gemma-2-2B (~2.3).
  - Design confound identified: 4B's late-6 window (28.6% flip) beats its
    early-8 window (10.7%), i.e. its writable zone is mid-late - consistent
    with its single-layer blips at L22/L25 and its symmetric five-arm doses
    (half1 34.5% / half2 34.5%) - while 1.7B's zone is early. An
    early-anchored width sweep conflates "how wide a write must be" with
    "where the writable zone is". Cross-model k50 is only meaningful
    measured inside each model's own zone.
  - Drift's status after this: the one-directional constraint survives
    (high-drift windows still never capture: 4B late-6 restate 8.9% is its
    only high-drift signal and its flips there are mostly nudges), but low
    drift is now clearly not sufficient - 4B's early band has the lowest
    early drift on the Qwen ladder and still refuses width-6 writes. There
    is a third structure (zone position) that drift does not predict.
- Verdict: extended, self-correcting again. One robust one-directional law
  (high drift kills capture), plus model-specific writable zones and width
  requirements that need a 2D map, not a single anchored sweep.
- Agreement with the external review: not covered there (internal follow-up).
- Next: sliding fixed-width window (k=6, stride 2) across the full band to
  map writable zones directly; 4B first (the anomaly), then 1.7B/Gemma-2B
  for comparison; the A100 models after the 27B profile chain frees the
  cards.

## 2026-07-15 E7 minimum write width (Gemma-2-2B control): the open-gate prediction FAILS in the informative direction - both families need ~2-3 layers; the family difference is position, not width

- Model / lens config: google/gemma-2-2b, same `run_e7_kwin.py` sweep as the
  Qwen3-1.7B entry below (early-anchored windows band[:k], k in {2,3,4,6,8,
  12}, late-anchored width-6 control L16..L21, none/full anchors), capitals,
  56/56 baseline ok, local MPS.
- Prediction being tested (from the entry below): an open-gate model should
  saturate from k=1 - its dose curve should look categorically different
  from dense Qwen's.
- Results (flip / restate): e2 28.6/32.1, e3 66.1/60.7, e4 89.3/73.2,
  e6 100/82.1, e8 100/89.3, e12 100/91.1, full(18) 94.6/83.9;
  late-6 control 94.6/92.9.
- Reading:
  - Prediction falsified. Anchored at the band start, Gemma's dose curve is
    only marginally ahead of dense Qwen's (k50 ~ 2.3 vs ~ 2.7 layers; e2
    28.6% vs 17.9%, e3 66.1% vs 55.4%). The minimum coordinated write width
    is ~2-3 layers in BOTH families - it looks close to universal, not like
    the family gate variable.
  - The actual family difference lives in the late window: at matched width
    6, Gemma-2-2B flips 94.6% AND rewrites the restatement 92.9%, while
    Qwen3-1.7B manages 17.9% flips with 0.0% restatement rewrite. Position
    (= drift at the window), not width, is what separates the families -
    exactly the drift law again.
  - Consequently the single-layer "open/closed gate" of the same-day audit
    is best read as a threshold slice through two similar dose curves
    shifted by ~half a layer: Gemma's strongest mid-band layers clear the
    k=1 threshold (peak 23-25%), Qwen's don't (0%). "Open vs closed" was
    real but shallow; "k50 + drift" is the deeper coordinate pair.
  - Residual family fact that width does not explain: Gemma saturates at
    100% flip vs Qwen's ~86% asymptote, and Gemma's restate tracks its flip
    closely everywhere (no flip-without-capture regime).
- Verdict: extended, self-correcting. Two-model evidence that the write-
  width requirement is family-independent (~2-3 layers) while position
  sensitivity (drift) carries the family difference. The "writability gate"
  axis introduced this morning largely dissolves into the drift axis plus a
  near-universal minimum width.
- Agreement with the external review: not covered there (internal follow-up).
- Next: Qwen3-4B width sweep (dense point #2: does k50 ~ 2-3 hold);
  Gemma-2-27B width sweep on the A100 once free (is its single-layer closure
  a right-shifted curve, k50 ~ 4-6, or qualitatively different); Qwen3.6-27B
  profile still running.

## 2026-07-15 E7 minimum write width (Qwen3-1.7B): the "closed" dense gate opens at 2-3 coordinated layers; width cannot buy capture at high drift

- Model / lens config: Qwen/Qwen3-1.7B, pre-fitted neuronpedia lens, local
  MPS. New script `run_e7_kwin.py`: early-anchored contiguous swap windows
  band[:k] for k in {2,3,4,6,8,12} plus a late-anchored width-6 control
  (L17..L22) and none/full anchors; capitals, 56/56 baseline ok. Flip is the
  primary grade (this model's none floors are exactly 0 on both grades).
- Why: the same-day single-layer audit left "closed gate" ambiguous - dense
  Qwen3 has zero single-layer flips yet 63-74% flips at half/full band, so
  closure cannot mean unwritable. Hypothesis: a single layer's write does
  not survive the healing of downstream layers, making the gate a continuous
  quantity (minimum contiguous write width) rather than a binary property.
- Results (flip / restate, early-anchored):
  - k=1: 0% (profile, prior run) -> k=2: 17.9/10.7 -> k=3: 55.4/41.1 ->
    k=4: 69.6/62.5 -> k=6: 76.8/73.2 -> k=8: 82.1/75.0 -> k=12: 85.7/75.0;
    full band (19): 78.6/71.4. A steep smooth dose curve, no hard step:
    half-saturation at k ~ 2.5-3 layers, plateau from k ~ 6-8.
  - Late-anchored width-6 control: flip 17.9%, restate 0.0%. At matched
    width the high-drift window produces some answer flips but not one
    restatement rewrite - width cannot buy capture where drift is high, and
    the flips it does buy are output nudges, not workspace writes. This is
    the drift law's mechanism claim observed directly within one arm.
  - k=12 (85.7%) beats the full band (78.6%): appending the high-drift tail
    layers slightly hurts, consistent with the late window acting as an
    interfering output nudge. Delta is ~4-7 pp, single seed; noted, not
    leaned on.
- Verdict: extended. Dense "gate closed" = write-healing imbalance at width
  1, overcome by 2-3 coordinated layers; the honest gate variable is the
  half-saturation width k50 (1.7B: ~2.5-3; open models: 1 by definition).
  This reframes the open/closed split of the same-day audit as a continuous
  cross-model quantity and predicts Gemma-2-27B's closure may be deep-stack
  healing (46 layers) rather than a different mechanism.
- Agreement with the external review: not covered there (internal follow-up).
- Next: same sweep on Gemma-2-2B (expect k50 = 1, curve saturating from
  k=1) as the open-gate control, and on Qwen3-8B/14B (does k50 widen with
  scale?) once the A100 frees; if k50 tracks total depth, "healing depth"
  replaces "family gate" as the second axis of the story.

## 2026-07-15 Gemma-2-27B single-layer profile: lowest drift does NOT buy the highest writability; within Gemma the gate narrows with scale

- Model / lens config: google/gemma-2-27b, pre-fitted neuronpedia lens (as
  its drift/five-arm entries); single-layer profile over band L13..L44
  (32 layers), capitals, n_ok 56/56, device_map=auto across both A100s
  (~2.4 h). Eight-model merge rerun (`analyze_e7_driftlaw.py`).
- Why: the interlock prediction from the seven-model table. Gemma-2-27B has
  the lowest drift measured anywhere (late plateau ~0.35); if single-layer
  writability were also the highest, the drift law and the writability gate
  would lock together at the top rung.
- Results:
  - The prediction fails cleanly. Peak single-layer capture is 16.1% (L41),
    *below* Gemma-2-2B (32.1%) and 9B (26.8%) and level with the hybrid
    Qwen3.5-9B (15.7%). Within Gemma, single-layer writability declines
    monotonically with scale while drift also declines: the two axes move in
    opposite directions across scale inside the family.
  - Worse for the per-layer reading: the none-arm grading floor rises with
    Gemma scale (2B 5.4%, 9B 7.1%, 27B 10.7% [3.6, 19.6]), and no 27B layer
    clears that CI ceiling (0 of 32, vs 12 layers for 2B and 9B). At the
    single-layer level the 27B signal is not separable from restatement
    noise, even though its full-band capture (66.1%) towers over the floor.
  - The curve shape is new: U-shaped, not a hump. Early plateau 12.5%
    (L13-L19, drift 0.03-0.07), mid-band dip to 3.6% (L26-L31, drift
    0.22-0.30), late recovery to 10.7-16.1% (L39-L44, drift ~0.35) - the
    only model in the table whose late band shows any single-layer capture,
    and the only one whose late drift is that low. But the dip sits at
    *lower* drift than the recovery, so drift cannot order this curve:
    Spearman -0.366 [-0.49, -0.12], significantly negative yet the weakest
    Gemma coefficient by far.
  - Eight-model regression: drift alone R^2 = 0.16; + three-level generation
    dummies R^2 = 0.64 (partial R^2 of generation given drift 0.58);
    coefficients still order gemma +0.120 > hybrid +0.057 > dense 0. The
    ordering survives, but the gemma coefficient drops from +0.150 as the
    27B pulls its tier down.
- Verdict: partial. The three-grade gate ordering survives at the top rung
  only coarsely (Gemma still writable somewhere, dense still closed); the
  strong "two laws interlock at scale" reading is dead - lowest drift did
  not buy back writability, and the within-model drift law itself weakens at
  27B (U-shape). Standing caveats: elevated substring-grading floor at 27B,
  and the pre-fitted default lens recipe differs from the Qwen self-fits.
- CORRECTION (same evening, item-level audit): the 27B "U-shape" is
  retracted - it is the grading floor, not capture. Two facts settle it:
  (1) the 27B profile has ZERO answer flips at every one of its 32 band
  layers, while every genuinely writable model shows flips alongside
  restate capture (peak single-layer flip: Gemma-2-2B 23.2%, Gemma-2-9B
  25.0%, Qwen3.5-9B 23.5%; at Gemma-2-2B's peak layer, 10 of 18 restate
  hits co-occur with a flip). (2) Every 27B layer's restate hits contain
  all 6 none-arm floor items (the items whose *unedited* generation
  incidentally mentions the swap target), e.g. L41: 9 hits = 6 floor + 3
  jitter. The incidental-mention floor grows with scale (2B 5.4% -> 9B
  7.1% -> 27B 10.7%) because bigger models write more comparative
  restatements; at 27B it swamps the substring metric. Consequences:
  (a) the flip metric (none-arm flip floor is exactly 0 in every model) is
  the honest per-layer gate signal; under it the within-model drift law
  gets tighter and family-consistent - Spearman(flip, drift) = -0.774 /
  -0.824 / -0.811 for Gemma-2B / Gemma-9B / hybrid-9B - and Gemma-2-27B is
  simply CLOSED at the single-layer level, like dense Qwen. (b) The
  "graded three-tier gate" reading of the earlier entry weakens: peak
  single-layer flip is essentially level across the three open models
  (23-25%), so by flips the gate looks binary (open/closed), and part of
  the restate-capture grading (Gemma > hybrid) was the Gemma floor
  inflation. The open/closed split - open = {Gemma-2B, Gemma-9B,
  hybrid-9B}, closed = {all dense Qwen3, Gemma-2-27B} - follows neither
  family nor generation alone; the gate variable is still unidentified.
  (c) The regression numbers of both same-day entries below are on the
  floor-contaminated restate metric and must be re-based on flips (or
  floor-corrected capture) before any of this enters the paper. Re-based
  same evening (analyze_e7_driftlaw.py now emits flip stats +
  regression_generation_flip): on flips the generation regression largely
  collapses - R^2 0.10 (drift) -> 0.28 (+gen), partial 0.20, coefficients
  gemma +0.036 / hybrid +0.086 - i.e. the eye-catching R^2 = 0.78 of the
  restate version was mostly floor structure plus the Gemma restate
  inflation, not a clean generation effect. What survives on the clean
  metric: (i) the within-model drift law in every open model (rho -0.77 to
  -0.82); (ii) a binary open/closed gate whose membership matches neither
  family nor generation; (iii) the dense floor.
- Agreement with the external review: not covered there (internal follow-up).
- Next: Qwen3.6-27B profile (chained, running) - to be read on the flip
  metric first. Add flip-based statistics to analyze_e7_driftlaw.py so the
  nine-model pass is graded on the uncontaminated signal.

## 2026-07-15 Qwen3.5-9B single-layer profile: the writability gate is graded by generation, and the family residual now has a mechanism

- Model / lens config: Qwen/Qwen3.5-9B, same n=1000 self-fit lens as its drift
  entries; single-layer capture profile (`run_e7_profile.py`, one full-strength
  swap arm per band layer L9..L30, capitals, n_ok 51/56) on one A100. The
  seven-model merge and regression are `analyze_e7_driftlaw.py` ->
  `results/e7_driftlaw_table.json`.
- Why: the five-arm ladder left a family residual that aggregate drift could
  not explain (Gemma > hybrid Qwen > dense Qwen at matched drift). The
  single-layer profile asks whether that residual tracks per-layer
  writability: can one layer's swap capture the restatement at all?
- Results:
  - The gate is open on the hybrid. Single-layer capture runs 8-16% across
    the early-mid band (L9-L18, drift <= 0.45), peaking at L14 (15.7%,
    frac depth 0.44). Dense Qwen3 at all four scales never leaves the floor
    (peaks 0 / 1.8 / 1.8 / 5.4%). Peak single-layer writability now orders
    Gemma (26.8-32.1%) > hybrid Qwen (15.7%) > dense Qwen (~0) - exactly the
    family-residual ordering from the five-arm late-capture comparison. The
    residual is no longer unexplained: it tracks single-layer writability.
  - Within-model drift law, sharpest curve so far: Spearman(capture, drift)
    = -0.834 over 22 band layers, vs Gemma-2-2B -0.79 and Gemma-2-9B -0.68;
    15 layers sit above the none-arm bootstrap ceiling (the most of any
    model; none-arm floor is exactly 0 here). The shape is textbook: capture
    holds through drift <= 0.45, halves at L19 (drift 0.52), and is
    identically zero from L24 on (drift >= 0.71, frac depth >= 0.75).
  - Seven-model pooled regression (166 layer obs; the two 27Bs still to
    come): drift alone explains R^2 = 0.14. Adding three-level generation
    dummies (gemma / hybrid / dense-reference) jumps to R^2 = 0.78, with a
    partial R^2 of 0.74 for generation given drift; coefficients order
    gemma +0.150 > hybrid +0.057 > dense 0. Adding amplitude changes nothing
    (R^2 0.779): the gate is not amplitude-starved layers. The original
    six-model binary-family block is frozen and unchanged.
- Verdict: extended (not in the paper). New shape of the story: single-layer
  writability is a family/generation property with three grades (dense
  closed, hybrid half-open, Gemma open); where the gate is open, coordinate
  drift orders which layers can be written within the model. What looked
  like the drift law "shrinking to a coarse ordering" splits into two clean
  laws on separate axes.
  [Same-day caveat - see the Gemma-2-27B profile CORRECTION above: on the
  uncontaminated flip metric the three open models peak level (23-25%), so
  the *grading* of the open tier is partly a restate-floor artifact; the
  hybrid's open-vs-dense-closed contrast and the within-model drift law
  both stand, the latter tighter (rho -0.81 on flips).]
- Agreement with the external review: not covered there (internal follow-up).
- Next: Gemma-2-27B and Qwen3.6-27B single-layer profiles (chained on the
  A100, ~3-4 h + 6-8 h). They test whether the three-grade ordering holds at
  the top rung - Gemma-2-27B has the lowest drift measured, so if its
  single-layer writability is also the highest, the two laws interlock at
  scale.

## 2026-07-15 Gemma-2-27B drift: the biggest Gemma drops hard; "top-of-ladder models drift lower" now holds in both families

- Model / lens config: google/gemma-2-27b (46 layers), pre-fitted lens from
  the neuronpedia/jacobian-lens release (default recipe, same as the other
  pre-fitted rungs; no self-fit). Weights via the ModelScope mirror on the
  A100 box; drift + amplitude on capitals, states, currency.
- Why: the scale explanation for Qwen3.6-27B's low drift rested on Qwen alone,
  and the Gemma ladder so far pointed the other way (2B 0.330 -> 9B 0.416).
  Gemma-2-27B is the single cheapest point that could break or support the
  scale story cross-family.
- Results (band = layers >= 28% depth, drift_entity):
  - Band mean 0.234: the lowest of every model measured so far, well below
    Gemma-2-9B (0.416) and below Qwen3.6-27B (0.385). At 0.9 fractional depth
    drift is 0.355 vs the 9B's 0.707; at 0.5 depth 0.153 vs 0.226.
  - Domain-stable: capitals 0.234, states 0.253, currency 0.226.
- Five-arm follow-up (same day, `results/e7_perspectival_gemma2-27b.json`,
  n_ok 56): the drift law's out-of-sample prediction (lowest late-half drift
  0.351 => late capture should be the highest measured, ~75%+) came out
  mixed. Late-half capture is 55.4%: far above every dense-Qwen point at high
  drift (0-25%), so the qualitative direction holds; but below Gemma-2-2B
  (80.4%) and 9B (75.0%), which have *higher* drift, so within-family
  monotonicity fails. Full-band capture 66.1% / flip 66.1%, controls clean
  (none restate 10.7%, randdir 8.9%, randdir flip 0%). Self-report: says_yes
  is edit-elevated (full 51.8% vs randdir 26.8%) but detection stays
  independent of capture (Fisher p = 0.78). Together with Qwen3.5-9B (drift
  0.715 but late capture 62.7%) the strict per-layer threshold reading of the
  drift law is dead; what survives is a coarse separation (high-drift dense
  Qwen never captures late; lower-drift Gemma and hybrid Qwen do) plus a
  family/generation residual that drift does not explain.
- Verdict: the top-of-ladder drift drop replicates cross-family. Combined
  with the Qwen3.5-9B family control (architecture generation does not lower
  drift at matched scale), the drift law's scale reading is now supported in
  both families. Standing caveats: drift is not monotone mid-ladder (Gemma
  rises 2B -> 9B; Qwen3-4B dips to 0.385), and the 27B lenses differ in fit
  recipe (Gemma pre-fitted default vs Qwen self-fit n=1000), so the claim is
  about the top rung, not a smooth scaling curve.
- Agreement with the external review: not covered there (internal follow-up).
- Next: fold Qwen3.5-9B + Gemma-2-27B into the drift-law table, figures, and
  the paper's drift-law section in one pass.

## 2026-07-15 E7 five-arm + cross-domain drift on Qwen3.5-9B: capture replicates on the hybrid architecture; the drift verdict is domain-stable

- Model / lens config: Qwen/Qwen3.5-9B, same n=1000 lens as the drift
  screening entry below; five-arm capitals seed 1 (matching the treatment of
  14B / 27B / Gemma-9B, which are also single-seed), plus drift + amplitude
  on the states and currency domains. Whole batch ran in ~25 min on one A100.
- Results (files `results/e7_perspectival_qwen35-9b.json`,
  `e7_drift_{states,currency}_qwen35-9b.json`, and amplitude counterparts):
  - Five-arm pattern replicates on the hybrid DeltaNet architecture:
    none flip 0.0, full flip 0.647 / capture (restate_swapped) 0.98,
    half1 0.667/0.922, half2 0.490/0.627, randdir flip 0.0 / capture 0.02.
    Direction-specific and dose-ordered, like every other model.
  - Self-report stays near floor (says_yes 0.02 on full), matching the other
    new-generation Qwen (27B: 0.0) rather than the older 8B/14B (~0.22).
  - Drift-law nuance: late-half capture is 62.7% at a late-half mean drift of
    0.715, far above the dense-Qwen points at comparable drift (1.7B 0.758 ->
    0%, 8B 0.782 -> 17.9%, 14B 0.793 -> 16.1%). Together with Qwen3.6-27B
    (0.586 -> 55.4%) the hybrid generation sits systematically above the
    dense-Qwen trend: a strict "high drift => no late capture" threshold is
    falsified; drift ordering holds within a family/generation, with a real
    family residual on top (Gemma > hybrid Qwen > dense Qwen at matched
    drift).
  - Cross-domain drift band means: capitals 0.516, states 0.540,
    currency 0.518. The "ordinary drift at 9B scale" verdict is not a
    capitals artifact (Gemma-2-9B shows the same domain stability:
    0.416/0.464/0.426).
- Verdict: replicated. The capture phenomenon and the drift law transfer to
  the hybrid-attention architecture; nothing about the new family behaves
  qualitatively differently at matched scale.
- Agreement with the external review: not covered there (internal follow-up).
- Next: Gemma-2-27B drift/amplitude (pre-fitted neuronpedia lens, no self-fit
  needed) as the cross-family scale point: does the biggest Gemma drop, or is
  the "big models drift lower" pattern Qwen-specific?

## 2026-07-15 E7 drift screening on Qwen3.5-9B: the 27B's low drift is scale, not the new architecture family

- Model / lens config: Qwen/Qwen3.5-9B (32 layers, hybrid DeltaNet generation,
  same architecture family as Qwen3.6-27B), Jacobian lens n=1000
  Salesforce/wikitext, fit on 2x A100-40G sharded 500+500 with fla fast path
  and dim_batch=16 (see 2026-07-14 infra entry). Full chain (download, fit,
  merge, drift, amplitude) ran unattended in ~10 h.
- What was run: `run_e7_drift.py` and `run_e7_amplitude.py` on the merged
  lens, capitals domain; results in `results/e7_drift_qwen35-9b.json` and
  `results/e7_amplitude_qwen35-9b.json`.
- Question this screens: Qwen3.6-27B shows much lower coordinate drift than
  every smaller Qwen3. Is that geometry (scale) or family (the new
  hybrid-attention architecture)? Qwen3.5-9B is the family control: new
  architecture at old-model scale.
- Results (band = layers >= 28% depth, drift_entity band mean):
  - Qwen3.5-9B: 0.516. Same-scale dense Qwen3: 8B 0.553, 14B 0.549,
    1.7B 0.558. Qwen3.6-27B: 0.385.
  - At matched fractional depth 0.5 the 9B sits at 0.343 vs dense 8B 0.355 /
    14B 0.341, while the 27B is at 0.173. The 9B curve is indistinguishable
    from same-scale dense Qwen3 through mid depth; only past ~0.8 depth is it
    mildly lower (0.742 vs ~0.83).
- Verdict: family hypothesis rejected. The new architecture at 9B does not
  confer the 27B's low drift; low drift tracks scale, not the Qwen3.5/3.6
  lineage. Caveat: Qwen3-4B's band mean (0.385) matches the 27B via very low
  early-band drift, so drift is not strictly monotone in parameter count;
  the clean claim is only "architecture generation does not lower drift at
  matched scale".
- Agreement with the external review: not covered there; this is a
  project-internal follow-up to the E7 drift law.
- Next: no E7 five-arm on the 9B for the scale question (it adds nothing
  beyond the dense 8B); it remains available as a cross-architecture
  robustness arm if the paper wants one.

## 2026-07-14 Infra: Qwen3.5-9B lens fit was 4x too slow on the torch DeltaNet fallback; flash-linear-attention verified numerically safe and adopted

- Trigger: the Qwen3.5-9B n=1000 lens fit (2x A100-40G, sharded 500+500) ran at
  5.4 min/prompt (~48 h ETA). Root cause: transformers' Qwen3.5 hybrid DeltaNet
  layers fell back to the pure-torch path because `flash-linear-attention` was
  missing, and its fat intermediates had also forced dim_batch from 32 down
  to 8 (the earlier OOM). The fla import itself was blocked by the box's gcc
  4.8 lacking `stdatomic.h` (triton JIT); fixed with the bundled devtoolset-7
  gcc via `CC=`.
- Numerical gate before switching kernels (all on Qwen3.5-9B, bf16 production
  recipe, results in `/root/jspace-lens/_ab/` on the A100 box):
  - Op-level fp32: fla `chunk_gated_delta_rule` and `FusedRMSNormGated` match
    the torch fallbacks exactly (forward cos 1.000000; all input grads
    cos >= 0.999). Same math, different rounding.
  - Per-prompt bf16 Jacobians can still diverge hugely between paths (layer-0
    relF up to 1.2 on one prompt pair): per-op ~0.4% bf16 rounding noise is
    chaotically amplified through 20-30 layers of backprop, with
    prompt-dependent gain. Neither path is privileged; the fla norm is in fact
    marginally closer to an fp64 reference.
  - Verdict test (n-scaling): cross-path lens relF shrinks from ~1.2 (n=2,
    pathological prompts) to 0.001-0.13 (n=8), i.e. <=2.7% on the E7 analysis
    band (layers >= 28% depth) - zero-mean noise that averages out, not a
    systematic bias. At n=1000 the kernel choice is far below prompt-sampling
    noise. A forced-fallback control (monkeypatching the fla symbols to None)
    reproduces the pre-fla baseline bit-for-bit (relF = 0), validating the
    harness.
- Outcome: production fit relaunched with fla + dim_batch=16 (peak 29.4 GiB,
  1.6 -> expected <1 min/prompt; ETA hours instead of days). The estimator is
  unchanged: exact accumulation, dim_batch is pure compute batching.
- Caveat for anyone comparing raw per-prompt Jacobians across hardware/kernels:
  below ~85% depth they are precision-noise-dominated in bf16; only the
  n-averaged lens is a stable statistic.

## 2026-07-09 Cone geometry robustness: the self-trained 124M counterexample survives an n=1000 refit (not an under-fit artifact)

- Trigger: the self-trained 124M is the load-bearing counterexample in the cone
  geometry section (raw 23.4 -> transported 31.2, i.e. it does NOT collapse,
  proving the collapse is a property of the fitted lens, not of the transport
  operation). It is also the only cone point where both the model AND the lens
  are ours; the published lens was fit at n_prompts=150, vs the 27B reference at
  n=1000. Reviewer attack surface: "your 31.2 is an under-fit artifact."
- What was run: refit the self-trained lens from scratch to n=1000 on an RTX 4090
  (CUDA 12.4, torch 2.6.0+cu124), same recipe (dim_batch=32, max_seq_len=128),
  prompts from the cached wikitext-103 first-1000. Recomputed the participation
  ratio with a check script verified to reproduce n=150 (raw 23.39 / J 31.21).
- Results: n=1000 gives raw 23.39 (unchanged, lens-independent) / **J transported
  eff_dim 31.08**, mean_cos 0.111. Vs published n=150: J 31.21. Fit took 13.6 min
  on the 4090 (vs ~5 h projected on MPS).
- De-confound (backend vs fit-scale, since the above changed both at once): also
  fit CUDA at n=150 (2.1 min). Three points: MPS-150 **31.21**, CUDA-150 **31.21**
  (identical -> backend effect ~0), CUDA-1000 **31.08** (150->1000 -> fit-scale
  effect ~0). Robust to each variable independently. Artifacts:
  out/selftrained-124m_jacobian_lens_{n150_cuda,n1000_cuda}.pt.
- Verdict: counterexample confirmed robust. The "does-not-collapse" result is not
  a fit artifact. KEEP the self-trained point.
- Artifacts: out/selftrained-124m_jacobian_lens_n1000_cuda.pt (the refit),
  out/*.n150.bak.pt (published n=150 preserved). Published live lens/ckpt left at
  n=150 so results/cone_geometry.json stays reproducible.
- Done: Table 5 footnote now states the self-trained fit n and this robustness
  (both main.tex); README documents the control's provenance and links the public
  GPT-2 124M model it comes from; the fitted lenses (n150 mps, n1000 cuda) are
  published as a GitHub release asset.

---

## 2026-07-09 E7 mechanism: coordinate drift explains *where* the capture band lives, and reconciles Gemma's late-band capture with the Qwen dose asymmetry

- Trigger: the cross-family check (2026-07-08) left an apparent tension. On the
  Qwen ladder early-band swaps capture the restatement and late-band swaps do
  not (half1 vs half2 restate: 75.0% vs 0.0% at 1.7B), an asymmetry the paper
  uses against a "global content-blind logit shift" reading. But Gemma-2-2B's
  late half captures about as well as its full band (half2 restate 80.4%
  [69.6, 89.3] bootstrap, flip 83.9% [73.2, 92.9]; vs Qwen-1.7B half2 0.0% /
  16.1%). Read naively this looks like the asymmetry is a Qwen-only artifact.
- What was run: a generation-free coordinate-drift probe (run_e7_drift.py).
  The swap direction at band layer l is d_l(t) = normalize(J_l^T W_U[t]); the
  pure output coordinate of token t is its unembedding row W_U[t]. Define
  drift_l(t) = cos(J_l^T W_U[t], W_U[t]), averaged over the 56 entity tokens.
  High drift => the swap direction has rotated into logit space, so a swap
  there is an output nudge, not a workspace rewrite. Computed per band layer
  for gemma-2-2b, qwen17b, qwen4b, qwen8b. Also started a single-layer capture
  profile (run_e7_profile.py) to overlay capture vs depth on drift vs depth.
- Results (concrete numbers): drift rises with depth in every model, but the
  *rate* is family-/scale-specific. At matched fractional depth 0.8: Gemma-2-2B
  0.53, Qwen-4B 0.66, Qwen-1.7B 0.83. This rank-orders late-band capture
  inversely and monotonically: Gemma-2-2B (lowest drift) late-restate 80%,
  Qwen-4B (mid) 25%, Qwen-1.7B (highest) 0%. The metric also predicts the
  previously-unexplained Qwen-4B exception (4B is the one Qwen scale that keeps
  partial late capture; it also has the lowest Qwen drift). So "late band does
  not capture" is really "high-drift layers do not capture" — a family-agnostic
  law with an architecture-dependent onset; Gemma's late band sits at a drift
  level where Qwen still captures.
- Three-way separation:
  - Official paper: uses the early/late dose asymmetry (present at 1.7B/8B/14B,
    absent at 4B) as one argument that capture is not a global logit shift.
    Does not measure coordinate drift on the swap direction or test a 2nd family.
  - Nanda review: n/a (did not run E7 dose arms or a cross-family capture test).
  - This project: the asymmetry is not primitive; it is downstream of a
    measurable per-layer geometric quantity (drift) that predicts capture
    across two families and across the Qwen scale ladder including the 4B
    anomaly. This strengthens, not weakens, the anti-global-shift argument: a
    content-blind output shift cannot explain why capture tracks how far the
    swap direction has rotated toward the unembedding.
- Verdict: partially replicated + extended. The dose asymmetry replicates as a
  special case of a drift law; the law is new (not in the paper) and cross-family.
- Depth control (added same day): Gemma-2-9B E7 ran (42 layers, band 12..40,
  n_ok 56/56, ~2 h on local MPS). Late-half swap still rewrites the restatement
  in 75.0% [62.5, 85.7] and flips 94.6% [87.5, 100], vs the Qwen ladder's 0-25%
  late-half restatement (25% at the 4B exception). So the Gemma late-band capture holds at a model deeper
  than Qwen-14B (40L) -> it is a Gemma-architecture property, not a shallow-model
  artifact. 9B's late-half mean drift (0.60) is above 2B's but still below the
  Qwen band; at matched drift Gemma captures more than Qwen (residual family
  effect), so drift is the dominant axis, not the sole determinant.
- Artifacts: results/e7_drift_{gemma2-2b,gemma2-9b,qwen17b,qwen4b,qwen8b}.json,
  results/e7_perspectival_gemma2-9b.json; scripts run_e7_drift.py +
  run_e7_profile.py; figure fig_e7_mechanism (drift vs depth | late-half drift
  vs late-half capture, 5 models); Gemma late-band (SKY) star on the E7 capture
  figure; gemma2-2b + gemma2-9b in the bootstrap E7 block; paper main.tex gains
  a mechanism paragraph + Figure (fig:mechanism) and the 220 dose-asymmetry
  disclosure now lists Gemma as an exception alongside Qwen-4B.
- Cut / deferred: Gemma-2-27B dropped (54 GB won't fit a 24 GB 4090 in bf16, and
  the flaky local HF link would not pull it tonight; 9B already answers the
  depth question). Single-layer capture profile (run_e7_profile.py) held out of
  the paper: capture there collapses mid-band even in Gemma at low drift, so it
  measures a different axis (per-layer signal amplitude) and would muddy the
  aggregate drift story; kept as a repo diagnostic. Gemma cross-domain
  (states/currency) deferred to v2.
- Next (v2): give the 75.0% a multi-seed check if a report-tuned Gemma is used;
  add Gemma cross-domain; investigate the residual family offset at matched drift.

---

## 2026-07-08 Second model family (Gemma-2-2B): capture and typo-register replicate cross-family; language-register is causal but modulated by target-language ability

- Model / lens config: google/gemma-2-2b (26 layers, band 7..24, read_layers [11,16,20]), pre-fitted J-lens from the same neuronpedia/jacobian-lens repo (gemma-2-2b_jacobian_lens.pt). No self-fit needed. Local Mac / MPS, bf16.
- Engineering: added a `family()` helper + Gemma entry to run_e1 MODELS; output filenames now `{family}{key}` so Qwen files are untouched and Gemma writes `*_gemma2-2b.json`.
- E7 (perspectival capture): replicates and is *stronger* than any Qwen scale. Full-band swap flips 94.6%, restatement rewritten 83.9% (Qwen 1.7B/4B/8B/14B: 71.4/58.9/69.6/67.9%); randdir at floor (flip 0%, restate 3.6%); fresh-forward lens none 23.2% FP -> full 98.2%; n_ok 56/56. Self-report again non-specific, in a new shape: full-arm yes 8.9% is *below* randdir yes 25.0% and none 16.1%, 2x2 Fisher p=1.000. Dose asymmetry weak (half2 restate 80.4%), like Qwen-4B.
- E6t (typo register, English): clean replication. Correction survival erased dose-dependently (a=0.5: 79.6% vs random 100%, p=3.8e-6; a=1: 0%); random control unfixes nothing at a<=0.5. Dose-response sits slightly later than Qwen (Qwen-4B already at 24.5% by a=0.5).
- E6 (language register, zh/en): METHOD LESSON + nuanced result. First run used the script default a=1, but Qwen E6 is evaluated at a<=0.25 (alphas 0.0625/0.125/0.25); a=1 is 4-16x over the measured gap and shattered the output (zh->en degenerated to "the the the", en->zh produced fact-scrambled Chinese with French token leakage), giving a spurious 1.4%. Re-ran with comparable doses. True dose curve is an inverted-U peaking at a=0.125 (flip_full 25.0%, preserved 41.7%, random 0%, p=7.6e-6), same shape and peak location as Qwen (Qwen peak also a=0.125: flip_full 55.6%/64.6% at 17B/4B). So the language-register axis IS causally load-bearing on Gemma (significant, random control zeroed) but at ~40% of Qwen's efficacy — consistent with the axis's causal strength being coupled to the model's target-language content ability (Gemma's Chinese is weaker; the axis still moves it but into lower-quality/looser Chinese).
- Verdict: register/capture mechanism holds across model families (capture stronger, typo clean, language-register causal-but-modulated); self-report unreliability also holds cross-family, in yet another idiosyncratic shape. Direct evidence that the register/capture findings generalize beyond the Qwen family.
- Takeaway logged for method: when porting to a new family, align experiment hyperparameters (intervention doses), not just the model id — the script default nearly wrote a false "Gemma register fails" into the results.
- Next: fold Gemma results into the paper's cross-family analysis.

---

## 2026-07-08 E7 cross-domain: perspectival capture is not specific to country capitals

- Model / lens config: Qwen3-1.7B (band 8..26) and 4B (band 10..34), same E7 protocol as the capitals run.
- What was run: two new domains factored to vary one thing each. `states` keeps the relation ("capital of") and swaps the entity type (37 US states); `currency` keeps the entity set (countries) and swaps the relation ("currency of", 29 countries). Full/half1/half2/randdir arms, fresh-forward J-lens readout, self-report stage, all as in capitals.
- Results (full-band restatement rewritten to the swapped question):
  - states: 1.7B 80.6% (n_ok 36/37), 4B 58.8% (34/37)
  - currency: 1.7B 77.8% (27/29), 4B 79.3% (29/29)
  - capitals baseline for comparison: 1.7B 71.4%, 4B 58.9% -> new domains bracket it.
  - randdir control at behavioral zero in all four runs (0% flips, 0% captured restatements).
  - fresh-forward lens: swapped entity outranks original in full arm 31/36 (states 1.7B), 26/34 (states 4B), 23/27 (currency 1.7B), 26/29 (currency 4B), vs 5/36, 13/34, 0/27, 8/29 in none arm.
  - dose asymmetry (half1 > half2 restatement) present in 3/4 runs; e.g. states 1.7B 77.8% vs 2.8%, currency 1.7B 66.7% vs 14.8%. Absent only in currency 4B.
- Verdict: replicated across a new entity type and a new relation, at both scales.
- Addresses: review W6 (single-domain concern). Now in main.tex §6 (`Capture is not specific to country capitals`) + Limitations rewritten (cross-domain done at 1.7B/4B; 8B/14B ladder still capitals-only).
- Next: none for v1; extending 8B/14B to the new domains is a v2 nicety, not a gap.

---

## 2026-07-08 #17 sensitivity reanalysis: the register/plan split survives the threshold dial; E6 full flips are template-graded

- What was run: zero-model-cost reanalysis of stored artifacts
  (`experiments/sensitivity/reanalyze.py` -> `results/sensitivity_reanalysis.json`),
  plus a mouth-rank threshold sweep recomputed from E4b per-row ranks
  (tables generated into the paper appendix).
- Threshold sweep (T = 10/50/100/500): register sets decay slowly
  (multilingual 26.4->19.2% at 1.7B, 35.3->21.7% at 4B; typo 36.5%/15.6% at
  T=100 on a 0% floor, thinning at T=500), plan sets evaporate (multihop
  13.6/21.4% at T=10 -> 1.0/1.0% at T=500). The dichotomy is a decay-rate
  signature, not an artifact of the T=100 cut.
- E6 template cells: zh->en full flips are 13/13 on the capital template at 4B
  but 2/27 on one-off facts (1.7B: 7/13 vs 2/19); en->zh high everywhere.
- E7 influence checks: no detection margin is outlier-driven (4B drop-top-3
  +1.17 -> +1.04; 8B randdir shift positive on 54/56 items).
- Verdict: headline numbers robust to item resampling; two disclosed
  structure findings (template grading, single-domain E7 pool) folded into
  the paper.

## 2026-07-08 E6t typo-register erasure: the second register is causally load-bearing

- Model / lens config: Qwen3-1.7B and 4B, band as in E6, mass-mean typo axis
  from 16 clean/corrupted sentence pairs (disjoint from the official typo
  set's targets), residuals averaged from the first diverging token.
- What was run: `experiments/e6t-typo-register/run_e6t.py` -- erase the axis
  (gap translation toward clean) on the official 96 typo items under a
  one-shot correction elicitation; amplitude-matched random control; yes/no
  report probe as a secondary readout.
- Results: baseline corrects 94/96 (4B) and 88/96 (1.7B). Erasure kills
  correction dose-dependently (4B: 86.2% -> 24.5% -> 0% at alpha = 0.25/0.5/1;
  1.7B: 92.0% -> 54.5% -> 0%) while the random arm unfixes nothing at
  alpha <= 0.5 (exact McNemar p < 1e-4 both scales). Failure mode through
  alpha = 0.5: coherent wrong word, near-zero echo of the corrupted form;
  alpha = 1 breaks the output (same compounding overdose as E6).
- Verdict: replicated-and-extended; gives the register family a second
  causal instance.

## 2026-07-08 E7 scale ladder (8B, 14B): capture is scale-stable, self-report is not

- Model / lens config: Qwen3-8B and 14B with released lenses; same 56-item
  protocol and arms as the 1.7B/4B runs.
- Results: capture flat across the ladder (flip 82.1%/80.4%, restate 69.6%/
  67.9%; randdir behavioral zero everywhere; early/late dose asymmetry
  present at 8B/14B). Detection channels scramble: 8B margin shift +0.37 is
  indistinguishable from randdir +0.35 (explicit yes 23.2% vs 19.6%; "Wait"
  1/56); 14B margin reverses sign (-0.38 vs randdir +0.04) while explicit
  yes is edit-specific again (21.4% vs 5.4%). 14B detected-x-captured
  2x2 = 11/1/27/17 (Fisher p = 0.079).
- Baseline lens false positives: 8B 33.9%, 14B 8.9% (see cone entry).
- Verdict: capture replicated at both new scales; the 4B "emerging monitor"
  reading is retired in favor of scale-idiosyncratic report mappings.

## 2026-07-08 Transport-cone geometry ladder: universal collapse, model-specific severity, two usability regimes

- What was run: pairwise cosines and participation-ratio eff-dim of raw
  W_U vs J-transported directions (~190 entities) across pythia-70m, GPT-2,
  self-trained 124M, Qwen3 1.7B/4B/8B/14B/32B*, Qwen3.6-27B; per-layer
  profiles for 4B/14B (`results/cone_geometry.json`, `cone_profile.json`).
- Results: raw eff-dim grows 75->128 across the family; transported
  collapses to 2-12 everywhere (11-51x). 27B (converged official-demo lens)
  is the most collapsed: eff-dim 2.2, cos 0.67. Self-trained 124M is the
  counterexample (transport more isotropic than raw). Layer profiles rule
  out the depth confound (4B plateau 1.9-3.4 vs 14B 10-15 at matched depth).
- Usability cross-check (with the E7 ladder): the two tightest cones
  (4B cos 0.58, 8B 0.36) both show 33.9% baseline lens false positives; the
  two loosest (1.7B 0.28, 14B 0.24) show 3.6%/8.9%. Reported as a two-regime
  split, not a law (n = 4).
- Verdict: original low-dimensionality observation confirmed and given a
  baseline, a scale axis, and a behavioral consequence.

## 2026-07-07 E7 perspectival capture (1.7B, 4B, 56 items): the edit rewrites the question, not just the answer

- What was run: `experiments/e7-perspectival-capture/run_e7.py` -- three-stage
  shared-KV protocol (answer / restatement / self-report), J-lens read over
  hook-free continuation forwards, arms none/full/half1/half2/randdir.
- Results (1.7B): flips 78.6%, restatement rewritten 71.4%, lens-behavior
  item agreement 54/56 (96%), zero captured restatements the lens disagrees
  with; self-report margin flat (+0.07 [-0.26, +0.41]); dose asymmetry
  (half1 captures 75.0%, half2 0/56).
- Results (4B): capture replicates (76.8%/58.9%); baseline lens false
  positives jump to 33.9%; margin shift +1.17 [+0.88, +1.46] with explicit
  yes 17.9% and "Wait" self-corrections 30/56, all edit-specific (randdir
  moves nothing); detected-x-captured 7/3/26/20 (Fisher p = 0.50).
- Verdict: new phenomenon (no prior demonstration of question-memory rewrite
  known to us); detection, where present, is not escape.

## 2026-07-07 E6 covert language register (60 pairs): causally load-bearing, and the covert subset dissociates register from content

- What was run: `experiments/e6-covert-register/run_e6.py` -- mass-mean
  parallel-sentence axis, gap translation at alpha = 0.0625/0.125/0.25,
  amplitude-matched random control, per-item covert precondition (target
  form outside clean top-100). Expanded 24 -> 60 pairs same day (covert
  n = 24 per model); old results kept as *.24items.bak.json.
- Results at alpha = 0.125: full flips 64.6% (4B, [54.4, 74.7]) / 55.6%
  (1.7B), language-only flips 91.1%/93.7%, random control 0% (McNemar
  p < 1e-4). Direction asymmetry: en->zh markedly cleaner (92.3% vs 37.5%
  at 4B). Covert subset: register still flips language (79-96%) but full
  flips drop to ~30%, loss entirely on zh->en items.
- Operator failures recorded: per-position reflection destroys the state
  (ambient coordinate ~2453 vs ~66 on random); full-gap translation
  compounds ~band-fold and overdoses.
- Verdict: register causally confirmed; the register carries the language
  setting, not the content.

## 2026-07-07 E4b mouth-exclusion audit: covert content is registers, not plans

- What was run: `experiments/e4-lens-eval/run_e4_covert.py` -- every lens hit
  on the six official sets scored against the model's own next-token
  distribution (shadow = mouth top-10; covert = rank >= 10; covert-strict =
  rank >= 100), permutation floors under identical scoring.
- Results (covert-strict, target vs floor): multilingual 31.3%/23.6% vs
  3.3%/1.4% (4B/1.7B); typo 15.6%/36.5% vs 0%/0%; poetry 0%; order-ops at or
  below floor; association 1-2%; multihop 2.9%/4.9% residue vs 0% floor.
- Verdict: the original paper's one-off ablation filter, promoted to a
  formal score, splits its own evaluation sets into surviving registers and
  dead plans.

## 2026-07-07 E2t trained-probe rerun: the answer-token dominance is a property of lens directions, not of the representations

- What was run: official-recipe trained linear probes (multinomial over the
  entity pool, held-out top-1 = 100% at every band layer) replacing lens /
  mass-mean directions in the E2 probe-swap arms.
- Results: trained probes restore the intermediate swap (50.0% at 4B, 65.5%
  at 1.7B; p = 0.0013/0.0063 vs mass-mean) while the answer-token control
  falls to 39.6%/41.4%, erasing the E2 asymmetry at 4B (McNemar p = 0.30)
  and reversing it at 1.7B (paired-difference bootstrap CI [-44.8%, -3.4%]).
- Verdict: C2 verdict revised to "partially replicated, at 50-65% rather
  than the original rates"; the deflationary reading of E2 is withdrawn.

## 2026-07-07 E5 specificity ablation: the swap is directed steering, not surgery on a designated thought

- Motivation: interactive exploration showed that swapping a *non-answer*
  workspace candidate into a target flips the final answer just like
  swapping the answer itself — suggesting the swap's effect does not depend
  on hitting the concept it claims to edit. The swap harvests amplitude
  along the source direction and re-injects it along the target; nothing in
  the operator checks that the harvested amplitude belongs to the source
  concept. Three arms on the official 90 two-hop items, same target
  direction (swap_answer), varying only the source. Code:
  `experiments/e5-specificity/run_e5.py`.
- Results (baseline-correct items; arm B reused from E2):

  | arm / source | Qwen3-4B hit | Qwen3-1.7B hit |
  |---|---|---|
  | B: the answer itself (official) | 85.4% | 82.8% |
  | C: the intermediate (in context, related, NOT the answer) | **77.1%** (vs B p=0.29) | 58.6% (vs B p=0.039) |
  | D: an absent word ("piano") | 0.0% | 0.0% |

- The mechanism dissected:
  - **Arm C flips are predicted by direction overlap, not by harvested
    amplitude**: at 4B, flipped items have mean cos(d_source, d_answer) =
    0.755 vs 0.324 for unflipped, while harvested amplitude barely differs
    (3.65 vs 3.31). Same split at 1.7B (0.591 vs 0.331). The effect travels
    through the correlation between the source's and the answer's
    transported directions — the collateral channel, not the advertised one.
  - **Arm D shows raw amplitude is not the currency**: its mean harvested
    |h.d| is comparable to C's (4B: 3.96 vs 3.57), yet nothing ever flips
    and outputs stay intact. What matters is position-coherent signal from
    a concept actually present in context; an absent word's direction picks
    up only diffuse background.
  - **The transported directions live in a narrow cone**: even
    cos(d_piano, d_answer) averages 0.549 at 4B (0.257 at 1.7B). With
    off-the-shelf overlaps this large, cross-concept collateral is
    structural, and it *grows* with scale in this pair.
- Verdict: at 4B, sourcing the swap from a related non-answer concept is
  statistically indistinguishable from sourcing it from the answer itself.
  The official demo's reading ("we rewrote the model's answer
  representation") is not supported: the same outcome is available without
  touching the answer's designated representation. The honest description
  of the operator is **directed steering whose gain is whatever coherent
  in-context signal the source direction happens to harvest** — including,
  via direction overlap, the answer's own. Combined with E2 (answer-token
  control beats intermediate swap) and E2p (probe-family directions do not
  rescue it), the "surgical thought edit" interpretation has no surviving
  support at this scale.

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
