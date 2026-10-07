# Decision Log

Each entry: date/session, issue, options considered, decision, rationale.

---

## 2026-09-17 — 2019 has zero qualifying heatwave episodes

**Issue:** Under the current definition (train-only ±7-day-of-year
climatology, anomaly > 1.5σ, ≥3-day consecutive run to count as an
"episode"), 2019 has **zero** qualifying episodes, despite being cited in
both `Retrieval_Augmented_Forecasting_Roadmap_Updated.md` and
`docs/P1_HANDOFF.md` as one of the two headline severe-heatwave years the
sanity checks were built around (alongside 2022, which has 3 episodes / 30
hot days).

**Investigation:**
- 2019's peak single-day anomaly is 2.28σ; 2022's peak is 2.32σ — essentially
  the same intensity.
- 2019's hot days are isolated spikes (e.g. May 30 and June 10, 11 days
  apart) rather than a sustained run; nothing in 2019 reaches a 3-day
  consecutive streak above 1.5σ. 2022 has a genuinely sustained run.
- Verified directly from `datasets/all_daily.parquet` — this is not a
  computation bug; the climatology and anomaly values match the documented
  method exactly.

**Options considered:**
1. Leave the definition as specified.
2. Lower the anomaly threshold (e.g. 1.2σ) so 2019 qualifies.
3. Loosen the run-length rule (e.g. allow a 2-day gap instead of 1) so 2019's
   spikes merge into one episode.

**Decision:** Option 1 — leave as specified. Flagged to the user, who did not
request a threshold change.

**Rationale:** The roadmap's own stratification (normal / unusual / extreme)
already captures 2019's spike days in the "unusual" stratum (1.0–1.5σ or
<3-day run) rather than discarding them — so evaluation isn't blind to 2019,
it just doesn't count it as an "extreme episode." Changing the threshold to
force a known year into the extreme bucket would be fitting the label
definition to a narrative rather than measuring what the stated definition
produces. Test-split episode count overall (17) is not "unexpectedly small"
per the roadmap's own Check 2 threshold for concern.

**If revisited:** the sensitivity is available — at 1.2σ, 2019 has 25
days above threshold instead of 8; a 1.2σ/3-day run may cluster into a
qualifying episode. Worth an ablation, not a silent change to the primary
definition.

---

## 2026-09-17 — Retrieval candidate boundary: roadmap wording is inconsistent, code follows the stricter reading

**Issue:** `Retrieval_Augmented_Forecasting_Roadmap_Updated.md` Section 5's
leakage rule states a candidate's 14-day input + 5-day forecast must fully
finish before the query's forecast period begins (implies candidate start
≤ query_date − 19 days), but a parenthetical in the same section says
"at least 5 days before the query date" — inconsistent with the sentence it's
clarifying.

**Decision:** No code change. `prepare_datasets.py`'s
`test_retrieval_candidate_boundary` test enforces the stricter 19-day rule,
matching the roadmap's main sentence. Logged here so the discrepancy isn't
mistaken for a bug in the code later, and so nobody "fixes" the code to match
the looser parenthetical.

---

## 2026-09-17 — Weighted-loss LSTM added as a separate exploratory variant, baseline kept frozen [SUPERSEDED — see next entry]

**Issue:** user asked whether the baseline LSTM's heatwave-detection
performance (4.8% recall) can be improved.

**Decision:** implement a weighted-MSE variant as a *new, separate* model
(`models/lstm_weighted_hotloss/`), not a modification of
`models/baseline_lstm/`. The frozen baseline is required as-is for Step 5's
controlled comparison (same training procedure, retrieval on vs. off) —
changing its training procedure now would invalidate that future comparison.

**Result:** the variant meaningfully improved detection (recall 4.8%→27.5%,
F1 0.089→0.307) at a small cost to overall fit (MAE +0.11°C). Full
write-up in `ml_notes.md`.

**Follow-on implication:** this raises the bar for Step 5/6 — retrieval now
needs to beat 27.5% recall / 0.307 F1 to demonstrate value beyond a much
cheaper loss-reweighting fix, not just beat the original 4.8% baseline or
persistence's 22.7%. Logged in `ml_notes.md`'s open questions.

**Not done:** `hot_weight=10` was not tuned/swept — a legitimate next step
if this direction is pursued further, but out of scope for this pass.

---

## 2026-09-18 — Which metric matters for this project: extreme-stratum RMSE, not detection recall

**Issue:** after fixing recall via weighted loss, then attempting to push it
further via a dual-head/focal-loss architecture to hit a 75-80% recall
target, it became worth stepping back and asking whether recall was ever
the right metric to optimize in the first place.

**Decision:** the project's actual north-star metric is **extreme-stratum
RMSE (or MAE), baseline vs. retrieval-augmented, with a bootstrap CI** —
exactly what `Execution_Pipeline.md` Step 6 already specifies ("the core
scientific result"). Detection recall/precision/F2 are retained as
supporting diagnostics, not the optimization target.

**Rationale:**
1. Recall is threshold-tunable post-hoc on an already-trained model — the
   dual-head experiment proved this explicitly (same model, 70/75/80/85/90%
   recall all directly selectable by moving a cutoff). A metric that can be
   hit by adjusting a threshold, independent of whether the underlying
   forecast improved, is a weak metric to stake a project's conclusion on.
2. Recall throws away magnitude and timing. "Correctly flagged a hot day"
   is much less informative than "predicted the peak within 1°C on the
   right day" — which is what the roadmap's peak/timing/onset/duration
   error metrics (not yet implemented — need P3/episode-level analysis)
   are for.
3. It directly tests the project's actual hypothesis: retrieval-augmentation
   is supposed to make the *continuous forecast* better during extremes by
   supplying real historical analogues, not just move a decision boundary.

**Consequence — evaluation code rewritten to match:** `training/evaluate_lstm.py`
and `training/visualize_results.py` narrowed to exactly 4 metrics: global
MAE/RMSE (credibility floor), stratified RMSE (headline), detection
precision/recall/F2 (diagnostic; F2 not F1, since recall matters more than
precision for this use case specifically), and the bias diagnostic
(explains the detection numbers mechanistically). R², MAPE, per-horizon
breakdown, plain accuracy, F1, and the raw confusion matrix were all
dropped as either redundant with RMSE or not decision-relevant. See
`ml_notes.md` for the full stratified results table.

**New function added:** `training/data.py`'s `build_split_target_stratum`,
implementing the roadmap's own normal/unusual/extreme definition (Section
4) at the forecast-day-instance level. This did NOT require touching
`training/train_lstm.py` or retraining anything — purely a new read-side
function over existing predictions/labels.

---

## 2026-09-18 — Dual-head (focal-loss) model built, evaluated, NOT adopted

**Issue:** explicit request to push heatwave-day recall to 75-80%+ without
substantially hurting global MAE/RMSE.

**What was built (in a working sandbox, not applied to the delivered
project):** `DualHeadLSTMForecaster` — shared LSTM encoder, an unweighted
regression head plus a separate focal-loss classification head, with the
decision threshold tuned post-hoc on the validation PR curve.

**Result:** the stated goal was achieved exactly (75-80%+ recall confirmed
at several operating points, global MAE improved to 1.566 vs canonical's
1.660). But once evaluated on stratified RMSE — built as a direct result of
the "which metric matters" decision above — the dual-head model turned out
**worse than canonical on both unusual and extreme RMSE**, worse than even
persistence on extreme. Bootstrap 95% CI on the extreme-stratum RMSE
difference (canonical − dual-head) = −0.998°C, CI [−1.190, −0.808],
excludes zero: canonical is decisively better on the metric that actually
matters.

**Decision:** dual-head model NOT adopted. Canonical (weighted-MSE,
single-head) remains the baseline. Full mechanism explanation and numbers
in `ml_notes.md`. The architecture idea isn't necessarily dead — a version
where both heads use weighted/focal losses (untested) might avoid this
trade-off — but nobody has built or tested that variant.

**Process note:** per the user's explicit formatting request for that
exchange, this was delivered as inline code snippets in conversation, not
as files or a packaged zip — the dual-head code (`training/losses.py`,
`training/train_lstm_dualhead.py`, `training/compare_models.py`, the
`DualHeadLSTMForecaster` class) exists only in the sandbox that produced
these numbers and in the conversation transcript, not in any file handed to
the user. If a future session needs to reconstruct it, it is not part of
this repo's current file tree — see `architecture.md`.

---

## 2026-09-18 — P2 (baseline forecaster) closed for now

**Decision:** per user request, P2/Step 3 is considered closed for the
present session, with the option to add further baseline model comparisons
(e.g. gradient-boosted trees, a plain feedforward net) in a later session.
Canonical model, evaluation code, and context docs are all in sync as of
this entry. See `progress.md` and `build_plan.md` for the closing-state
summary, and `ml_notes.md` for the full numbers this decision rests on.

**Known limitations carried forward, explicitly not blocking closure:**
- All reported numbers are val-split; test remains untouched (correct, per
  convention — test is for Step 6, once RAG exists).
- No confidence interval yet on LSTM-vs-persistence's extreme-stratum RMSE
  gap (bootstrap machinery exists, just hasn't been pointed at this specific
  comparison).
- n=154 extreme-stratum instances is thin; expect wide CIs once computed.
- Only one architecture (LSTM) evaluated; a reviewer would reasonably ask
  why, if this becomes a paper.
- `hot_weight=10` untuned/unswept.

**Superseded by:** user explicitly requested "just keep the better
performing model" — see the following entry.

---

## 2026-09-17 — Consolidated to a single baseline: weighted-loss LSTM promoted, plain-MSE version retired

**Issue:** having two parallel LSTM models (plain-MSE baseline, kept frozen
for Step 5, and a separate weighted-loss exploratory variant) added
structural complexity. User explicitly asked to keep only the better-
performing model rather than maintain both.

**Decision:** the weighted-loss model (hot_weight=10) is now **the**
canonical baseline. Concretely:
- `models/lstm_weighted_hotloss/` renamed to `models/baseline_lstm/`,
  replacing the old plain-MSE checkpoints (deleted, not archived — the
  plain-MSE results remain fully documented in `ml_notes.md`, so nothing is
  lost, just the checkpoint files themselves).
- `training/train_lstm_weighted.py`'s logic merged into
  `training/train_lstm.py`, which is now the single canonical training
  script (weighted MSE, hot_weight=10). The old plain-MSE version of
  `train_lstm.py` no longer exists as a separate path.
- `training/evaluate_variant.py` removed (no longer needed once there's only
  one model to evaluate); `training/evaluate_lstm.py` remains the canonical
  evaluation script and was re-run against the new checkpoints to regenerate
  `evaluation/baseline_lstm/val_extended_metrics.json` (numbers match the
  prior "weighted MSE" run exactly, confirming a clean consolidation with no
  accidental retraining-induced drift).

**Material consequence for Step 5 (flagging clearly, not burying this):**
the "baseline" now being carried forward uses weighted MSE
(hot_weight=10), not the roadmap's originally-specified plain MSE. When the
retrieval-augmented model is eventually built (Step 5), it must use this
same weighted-loss objective to preserve Execution_Pipeline.md's
requirement that baseline and retrieval-augmented models share an identical
training procedure, differing only in retrieval on/off. Using plain MSE for
the retrieval-augmented model while the baseline uses weighted MSE would
invalidate that comparison. This is a deviation from the roadmap's literal
spec ("Loss: MSE on z-normalized target") — flagged here explicitly rather
than silently carried forward, in case the team wants to revisit it before
Step 5 starts.

**Rationale:** the user's priority (heatwave detection performance) is
better served by a single, better-performing model than by preserving two
models for a hypothetical comparison that hadn't been requested. The
plain-MSE result's scientific value (motivating *why* weighted loss was
needed) is preserved in `ml_notes.md` regardless of whether the checkpoint
files themselves still exist.

---

## 2026-09-17 — P3 (retrieval) deliberately deferred

**Decision:** Per explicit user instruction, proceeding with P2 (baseline +
eventually retrieval-augmented model architecture) first; P3 (retrieval
system itself: statistical feature vectors, FAISS index, query-time
eligibility filter, dedup) is deferred to a later session. This means Step 5
(joint integration) cannot start until a future session picks up P3.

---

## 2026-09-19 — P3 retrieval system: two findings worth flagging

**1. Naive episode-based dedup was insufficient — fixed with temporal-adjacency dedup.**
Manual inspection (roadmap's own Step 4 validation requirement) surfaced a
real gap: only 590/17,025 windows (3.5%) have a `target_episode_id` at all,
so the roadmap's literal "max-2-per-episode" rule left 96.5% of candidates
completely unprotected from duplication. A real query's top-5 came back as
5 consecutive days from the same non-qualifying 2010 warm spell (shifted
1-4 days each) — near-identical vectors since consecutive windows share 13
of 14 input days. Fixed by adding a `min_days_apart=10` general temporal
spacing rule alongside (not instead of) the episode cap — see
`retrieval/query.py`'s `dedup_max_per_episode`. Verified via
`tests/test_retrieval_eligibility.py::test_dedup_respects_temporal_spacing`.

**2. "Heatwave" episodes are NOT concentrated in summer — 37% occur in
Jan-Mar, more than the classic Apr-Jun pre-monsoon season (23%).** Because
the definition is a *relative* anomaly (1.5σ over that calendar day's own
±7-day climatology), and winter's day-to-day variability appears tighter
than summer's, the same fixed sigma threshold is easier to cross in winter.
This is a pre-existing property of Step 2's definition (not something P3
introduced), only now surfaced by inspecting retrieval results directly.
**Practical implication:** "heatwave" in this pipeline's technical sense
means "anomalously warm for the time of year, year-round," not
specifically "dangerous summer heat" — worth reconciling with the paper's
framing/title before writing methods/results sections. Not fixed or
changed — flagged for the user's decision, doesn't block P3.

**Decision:** proceed with P3 as specified; both findings logged rather
than silently worked around. See `ml_notes.md` for the retrieval system's
full spec and validation results.

---

## 2026-09-22 — hot_weight swept, canonical updated to 15

**Sweep results (val, 5 seeds each):** hot_weight ∈ {5,10,15,20,25} —
extreme-stratum RMSE improves monotonically (1.932→1.145) as hot_weight
increases; global MAE degrades monotonically (1.583→1.842) in the same
direction. No free peak — a real trade-off curve, not a bug.

**Decision:** canonical changed from hot_weight=10 to **hot_weight=15**.
Extreme RMSE improves meaningfully (1.598→1.459, ~9%) for a moderate global
MAE cost (1.660→1.724, +3.9%) — better trade-off point than 20/25, which
push global MAE too far for the credibility-floor role it needs to play.
Checkpoints promoted directly from the sweep (no retraining needed,
`models/hw_sweep/hw_15/` → `models/baseline_lstm/`). `training/train_lstm.py`'s
`HOT_WEIGHT` constant updated to 15.0 to match.

**Full sweep table saved:** `evaluation/hw_sweep/sweep_results.json`.

**Consequence:** Step 5's retrieval-augmented model must use hot_weight=15
(not 10) to preserve the "same training procedure" comparison — updating
the note in `ml_notes.md`/`progress.md` accordingly.

---

## 2026-09-22 — Step 5 will default to Colab, not local

**Issue:** user's laptop is ~5 years old; even baseline LSTM training
(100-160s/5 seeds) may be more strain than in the dev sandbox this was
built in.

**Decision:** Step 5 (attention-fusion retrieval-augmented model) will be
built Colab-first — code structured to run there without modification,
with clear notebook/run instructions in `context/RUN_COMMANDS.md`. Local
CPU remains an option for quick single-epoch smoke-tests only, not full
training runs.

---

## 2026-09-22 — canonical changed again: hot_weight 15 -> 20

**Issue:** user pointed out the 15->20 marginal trade-off wasn't properly
checked before picking 15 -- correct catch.

**Marginal analysis (per +5 step):**
| Step | dMAE | dExtremeRMSE | Efficiency (gain/cost) |
|---|---|---|---|
| 10->15 | +0.064 | -0.139 | 2.17 |
| 15->20 | +0.051 | -0.174 | **3.39** |

15->20 is a strictly better marginal trade than 10->15 (smaller MAE cost,
bigger extreme-RMSE gain) -- if 10->15 was justified, 15->20 was more so.
Original choice of 15 was a "moderate middle ground" heuristic, not derived
from this marginal comparison -- acknowledged as a weaker justification.

**Caveat:** extreme-stratum RMSE has real seed noise here (std ~0.13-0.17
at these points, n=154 extreme samples / 5 seeds) -- the 15->20 gap (0.174)
is ~1-1.3 std, a real but not rock-solid signal. Bootstrap CI machinery
exists (used for canonical-vs-dual-head) but was not run for this specific
comparison -- offered to the user, not requested, may be worth doing before
this number goes in a paper.

**Decision:** canonical changed to hot_weight=20. Checkpoints promoted from
`models/hw_sweep/hw_20/` (no retraining in the sandbox; user opted to
retrain locally rather than receive checkpoint files this time).
`training/train_lstm.py`'s `HOT_WEIGHT` updated to 20.0.

**New canonical numbers (val):** MAE=1.775, global RMSE=2.401,
extreme RMSE=1.285, recall=0.339, precision=0.268, F2=0.322,
bias on hot days=-1.193C, bias on normal days=+0.884C.

**Consequence:** Step 5 must use hot_weight=20 (not 15) for the
retrieval-augmented model's training procedure to match.

---

## 2026-09-22 — bootstrap CI: hw=15 vs hw=20 extreme RMSE gap is NOT statistically significant

**Result:** mean diff (hw15-hw20 extreme RMSE) = +0.172 (favors hw20),
95% CI = [-0.075, +0.595], **includes zero**. n=67 unique extreme-stratum
windows, 2000 bootstrap resamples (window-level resampling + seed
resampling across the 5 trained seeds per value). Saved to
`evaluation/hw_sweep/bootstrap_ci_15_vs_20.json`.

**Interpretation:** the point estimate still favors hw20, but with only 67
extreme windows the gap is not distinguishable from noise at 95%
confidence. Neither hw15 nor hw20 is "provably correct" over the other at
this sample size.

**Decision:** keep hw20 as canonical (no evidence to revert to 15 either).
**For any future writeup:** report this honestly as "extreme-stratum RMSE
improved with hot_weight up to ~15-20; further distinctions beyond that
were not statistically significant at this sample size" rather than
claiming hw20 as a confirmed optimum. Do not oversell precision this
hyperparameter search doesn't actually have.

---

## 2026-10-04 — Plan v5 adopted; v1 frozen; Week-1 decisions

**Plan:** `docs/PLAN_REVIEW_v5.md` replaces Execution Plan v4 for a 2-person, 8-week scope.
Its recorded decisions: regional upstream DSTGNN + physics-structured WBGT head;
label rule (Tmax ≥ 40 °C and anomaly ≥ 3 °C, IMD 4.5 °C as severe); headline metric =
extreme-stratum RMSE with an absolute climatology skill floor and a forecast-conditioned
stratum; Phase 6 health-only unless citable energy sources are found.

**Statistics:** paired comparisons use `evaluation/stats.paired_cluster_test`, a
cluster-jackknife t-test with year × season clusters. It replaces the window-level
bootstrap, which was too narrow: the old method falsely called "RA beats persistence"
significant. Diebold-Mariano is a secondary check only.

**RA-v1 zero-analogue padding:** `models/retrieval_lstm.py` spreads attention over
zero-padded slots when a query has no eligible analogue (163/13,131 train windows, no val).
Decision (user-approved): keep RA-v1 exactly as built (frozen reference) and fix this in
the v2 retrieval model (empty retrieval → zero context), not by editing v1 code.

**WBGT target (BoM approximation):** hourly-derived BoM WBGT exceeds air temperature in
cool humid weather and reaches ~40 °C in the monsoon (22.8% of days ≥ 35 °C). It is a
known high-side approximation that ignores actual radiation and wind. Consequences:
- No absolute WBGT thresholds (ISO 7243 etc.) are applied to it. Labels are percentile-based.
- Official alert tiers stay Tmax-based (Delhi HAP / IMD).
- Tier-B (Liljegren) validation moves up from "cut" to "recommended".
- The paper names the target "BoM-approximated WBGT".

**Premise check (train years only):** seasonal Tmax shows no significant trend, while
WBGT (+0.28 to +0.53 °C/decade) and Heat Index trend upwards. So the non-stationarity
hypothesis is framed on humid heat, with Tmax as the negative control.

**Seeds — DECIDED 2026-10-04 (team agreed):** with 5 seeds, training randomness alone
sets an extreme-stratum detection floor of ~0.26 °C (`evaluation_v2/mde.md`). Use 10 seeds
for every decision run (LSTM: ~2 min per 5 seeds). The 5/10-seed ensemble-mean forecast is
reported as an extra, not as the primary model.

**Damped persistence as the reference floor — DECIDED 2026-10-04 (team agreed):**
damped anomaly persistence (5 train-fitted coefficients) beats both v1 neural models
overall on val. Rule: every v2 model must be not significantly worse than damped persistence on
all-days RMSE (paired cluster test), in addition to the extreme-stratum headline and the
forecast-conditioned stratum; report RMSE by lead day. A2r (anomaly target) becomes a
required control, not an option.

Practical consequences of the two decisions: every decision-feeding run uses seeds 0-9;
every results table includes damped persistence and an RMSE-by-lead breakdown; a model
that wins on extreme days but fails the floor is reported as a trade-off, not a win.

**WBGT target = physical (Liljegren) WBGT; WBGT models use the WBGT label: DECIDED 2026-10-05 (user)**
- New WBGT runs forecast `wbgt_lj_max` (physical WBGT, `pipeline/wbgt_liljegren.py`), not the BoM index.
- Their hot-day loss weight and their "extreme" stratum use the **WBGT label**: physical WBGT, Mar 15–Sep 30,
  95th in-season percentile (training years), chosen by the ≥ 25-episode rule (`configs/wbgt_label.json`).
- Tmax models keep the Tmax label. WBGT models are also reported on the Tmax label, so cross-target
  comparisons stay visible.
- Conditions:
  1. Phase-6 official alert tiers stay Tmax-based (IMD-style). Humid heat is a separate, clearly labelled
     heat-stress note, never an official heatwave declaration.
  2. The final model must output both Tmax and WBGT (the physics head, 2a, does this by design).
- Limitation: no officially documented humid-heat events exist to check the WBGT label against (Cowork asked).
- One-change chain: A1′ (Tmax, Tmax label) → A2L_t (physical WBGT target, Tmax label) → A2L (WBGT label)
  → A2Lr (anomaly target). BoM runs A2/A2r stay as historical results.

**Improvement candidates: PRE-REGISTERED 2026-10-05, before any of these runs**
Goal: bring the controls above the damped-persistence floor. The candidate list is fixed in advance and each
candidate changes one thing:
1. Tmax anomaly target: A1prime_r.
2. Damped-persistence residual (the model learns a correction to damped persistence): A1prime_dp, A2L_dp.
3. hot_weight ∈ {1, 5, 10} (20 already run) for A1′, A2L and A2Lr.

The 10-seed ensemble mean is reported as an extra only (decision 2026-10-04 stands); the primary score is the
mean of per-seed RMSEs.

Control choice (G2), applied after all runs, separately for the Tmax and the WBGT families:
- Among variants **not significantly worse than damped persistence on all days** (paired cluster test,
  2007-2018 pooled), choose the one with the lowest all-days RMSE.
- If two are within 0.02 °C, take the simpler one: fewer changes from the parent.
- If none passes, report the family as failing the floor and keep the variant closest to it.
- Extreme-day and forecast-conditioned results are reported for every variant but do not drive the choice.

**G2 controls: A1prime_hw5 (Tmax) and A2Lr_hw5 (WBGT physical): DECIDED 2026-10-06 (user), a disclosed deviation**
- All 15 pre-registered improvement runs finished (10 seeds × 4 folds each). Report: `evaluation_v2/week3_controls.md`.
- The rule above, as written, picks hot_weight = 1 in both families (`A1prime_hw1`, `A2L_hw1`). Lower hot_weight
  always lowers all-days RMSE, and the rule looks only at all days. The result defeats the project's purpose:
  - Tmax hw1: no extreme-day gain over damped persistence (−0.03 °C, CI [−0.32, +0.26]).
  - WBGT hw1: forecasts **0** WBGT hot days in 12 validation years (observed: 360 extreme instances).
- Decision: fix hot_weight = 5, then apply the pre-registered rule unchanged among the hot_weight = 5 runs.
  - Tmax → `A1prime_hw5`: all days −0.009 vs damped persistence (CI [−0.042, +0.025], a tie);
    extreme days −0.93 °C (CI [−1.19, −0.67]).
  - WBGT → `A2Lr_hw5` (2.373 vs A2L_hw5 2.400, beyond the 0.02 °C tie margin): all days +0.004
    (CI [−0.060, +0.069], a tie); extreme days −1.20 °C (CI [−1.34, −1.06]).
- Why hw5 rather than hw10: hw10 fails the floor for WBGT and is borderline for Tmax (p = 0.052), with a
  larger warm bias on forecast hot days (+1.4 °C Tmax).
- Honesty conditions:
  1. This is a post-hoc change to a pre-registered rule, made after seeing development-fold results. Report it
     as such in the paper, with the rule's own choice (hw1) shown alongside.
  2. It was decided on 2007–2018 development folds only; the test period (2019+) remains locked.
  3. `evaluation/compare_v2.py` reports both choices (`ADOPTED_HOT_WEIGHT = 5`).
- Lesson for later pre-registrations (retrieval ladder, G3): a selection rule must score both all days AND
  heat days, or it will select models that never warn.

**Retrieval ladder, first rungs (R0 / R0-rand / R1): PRE-REGISTERED 2026-10-06, before any of these runs**
- Controls: `A1prime_hw5` (Tmax family) and `A2Lr_hw5` (WBGT physical family). Each rung is the control plus
  retrieval, nothing else changed (same target form, hot_weight 5, 10 seeds, folds f1-f4, inner_2y early stop).
- Retrieval is rebuilt per fold from that fold's training years only (features, normalisation, candidate pool,
  episodes). v1 eligibility rules kept: analogue finishes >= 19 days before the query; validation queries see
  training windows only; no analogue from the query's own episode; dedup max 2 per episode, >= 10 days apart; K = 5.
- Rungs:
  - **R0**: top-K by cosine similarity on v1's 17 features (the RA-v1 design).
  - **R0-rand**: K random eligible analogues (same eligibility and dedup), fixed per fold and seed. Control.
  - **R1**: only candidates whose window-end day of year is within ±30 days of the query's, then top-K by
    similarity. Our calendar-alignment rung, in the spirit of SARAF's time alignment (exact SARAF formula not
    checked: the paper is not in the repo).
- Model: `models/retrieval_lstm_v2.py` = RA-v1 architecture; a query with no eligible analogue gets a zero
  context vector (v1 attends over zero-padded slots).
- **Selection rule (G3), applied per family after all rungs have run.** It scores heat days as well as all days
  (lesson of 2026-10-06). A rung "helps" only if ALL hold, using paired cluster tests on 2007-2018 pooled folds:
  1. not significantly worse than its control on all days;
  2. significantly better than its control on extreme days (95% CI entirely below 0);
  3. lower extreme-day RMSE than R0-rand (for R0-rand itself: n/a, it is the control).
  Among rungs that help, pick the lowest extreme-day RMSE; within 0.02 °C, the simpler rung (R0 < R1).
  If none helps, retrieval is reported as not helping at this stage (a valid result; see plan risk 5).
- Always reported: R0 vs R0-rand on all strata (the key "is the retrieved information used?" test), the
  forecast-conditioned count and bias (so a rung cannot win by forecasting warmer), and the AnEn baseline.

**G3 rule amendment: 2026-10-06, BEFORE any retrieval-queue result was looked at** (prompted by an independent code review)
1. **Condition 3 must be significant:** "lower extreme-day RMSE than R0-rand" becomes "significantly lower: 95% CI of
   (rung - its random control) on extreme days entirely below 0". A 0.001 °C point difference must not count.
2. **R1 gets a calendar-matched random control, R1-rand:** random eligible analogues within ±30 days of the query's
   day of year. R0-rand draws from all seasons, so R1 could beat it just by carrying season-matched climatology.
   Condition 3 for R1 is judged against R1-rand; for R0, against R0-rand. R1-rand is added to the queue as two more
   runs (one per family), using the same seeds and folds.
3. **The warm-bias check stays descriptive** (forecast-conditioned count and bias, reported for every rung).
   It is not a pass/fail condition, because no threshold was pre-registered and picking one now would be arbitrary.
   Condition 1 (not significantly worse on all days) remains the only enforced guard against a general warm bias.
   The report says this plainly.
4. All of R0 vs R0-rand (and R1 vs R1-rand) are reported on all four strata (all / normal / unusual / extreme).

**Correction to the wording of the G3 amendment above (2026-10-06, after a second review)**
- "BEFORE any retrieval-queue result was looked at" is accurate about *viewing*, but results already *existed*.
  When the amendment was written (about 22:01 IST), the queue had finished `A1prime_hw5_R0` fold f1 (registry row
  16:25:10 UTC = 21:55 IST). Fold f2 followed at 16:36:06 UTC (22:06 IST). No score from the queue had been
  displayed to the author: the log checks printed only run start lines.
- Disclosure: during the second review (after the amendment existed), the reviewer's hash check printed the two
  R0 registry rows, so the reviewer saw two extreme-day RMSE values for R0 alone (f1 1.835, f2 2.164). No control,
  random-control or comparison numbers were seen. These values were not used for any decision.
- Training code unchanged since the queue started: the reviewer recomputed `_code_sha256()` = 49129b3d…e9c3, which
  equals the registry's code_sha256 for both rows.

**Phase 6 scope: energy actions INCLUDED: DECIDED 2026-10-06 (user)**
- The 21 energy actions in `sources/phase6_candidate_actions.json` (CEA advisory, BEE AC 24 °C notice, DISCOM
  demand-side notices) are in scope for the advisory. This replaces "health-only unless official energy sources
  are found": official energy sources were found.
- Same rules as for health actions: word-for-word quote, page, verified; a tier only from a source or a recorded
  team decision.
- Still open: observation-triggered actions (in or out of the forecast advisory), and the tier mapping
  (Part E brief for Cowork, `docs/PHASE6_PART_E_BRIEF.md`).

**Phase 6: observation-triggered actions: DECIDED 2026-10-06 (user)**
- Actions triggered by something that has already happened (a heat-stroke patient, body temperature >= 40 °C,
  a cluster of heat deaths, a workplace measurement) are NOT forecast actions. They go in a separate
  **"If this happens"** section of the advisory and demo page, clearly labelled "not a prediction", and are never
  mixed with the forecast-based actions.
- One page for everyone (no separate health-worker view). The public first-aid steps are shown directly. The
  clinical protocols (IV fluids, cooling targets, emergency-department steps) sit in a collapsed
  "For health professionals" sub-section behind a disclaimer ("clinical guidance quoted from NCDC/NPCCHH, for
  trained health workers only; members of the public: call 108/102").
- The advisory output keeps two separate lists (forecast actions; if-this-happens notes), and the verifier checks
  both against the verified quotes.
- Front end (open): plan v5 says a static demo page (Week 8). Proposed: a single interactive HTML page (no build
  tools, nothing installed). Upgrade to React only if Week 7 has slack, and only with Node.js approved.

**Front end: interactive single-file HTML page: DECIDED 2026-10-06 (user)**
- Week 8 demo = one interactive HTML page: 5-day forecast chart (Tmax / WBGT toggle), colour-coded alert days,
  click a day for its advisory, the "If this happens" section, and a replay of past heatwaves once the test
  lock opens. No build tools and nothing installed; a charting library is loaded from a CDN.
- It reads the advisory's JSON output, so a later upgrade to React reuses the same data. Upgrade only if needed,
  and only with Node.js approved.

**Final-model architecture: graph backbone + physics head + retrieval: CONFIRMED 2026-10-06 (user)**
- Plan v5 §4 "Option 1 + 2a together", now written down explicitly. Final model = regional DSTGNN backbone +
  physics-structured output head (predicts temperature and humidity, computes WBGT inside the model) + the best
  retrieval rung (G3).
- Only the backbone is conditional. If G-D0 (upstream signal) or G-D3 (non-inferiority vs an LSTM given the
  same upstream data) fails, the backbone becomes an LSTM. The physics head and retrieval stay; the head is also
  required by the 2026-10-05 condition (output both Tmax and WBGT). BB* is decided by the end of Week 5.
- Wording fix: "fall back to the physics-guided LSTM" (plan and guide) means LSTM backbone + physics head, not
  dropping the physics part.

**Gate G-D0 (does upstream heat help Delhi's forecast?): PRE-REGISTERED 2026-10-06, before any upstream number
was computed**
- Question: do the 27 upstream points carry information about Delhi's next 5 days that Delhi's own recent weather
  does not? This is a cheap linear check, done before any graph network is trained.
- Data: `datasets_v2/upstream_daily.parquet` (from the verified raw download). Only folds f1-f4, validation 2007-2018;
  nothing from 2019 on is read. Per fold, from that fold's training years only: each point's Tmax day-of-year
  climatology, standardised anomalies, and all fitted coefficients.
- Reference: Delhi damped persistence (per-lead factor on Delhi's last standardised Tmax anomaly).
- Augmented: per lead L = 1..5, ridge regression of Delhi's standardised Tmax anomaly on Delhi's last anomaly PLUS
  the 27 points' standardised Tmax anomalies on the last 3 input days (81 extra inputs). The ridge strength is
  chosen from {0.1, 1, 10, 100, 1000} on each fold's last 2 training years (inner split, as inner_2y), then
  refitted on all training years. Forecasts are converted back to °C with the fold's Delhi climatology.
- **PASS** if, pooled over the four validation blocks, the augmented forecast has significantly lower RMSE than
  damped persistence on ALL days over leads 1-3 (paired cluster-jackknife test, year x season clusters,
  95% CI entirely below 0). Otherwise FAIL: the backbone becomes an LSTM (architecture decision 2026-10-06).
- Reported but not part of pass/fail: each lead separately; extreme days (Tmax label); WBGT (physical) with the
  same method; and WHICH points help: per-point skill gain (Delhi + that one point's 3 lags) and the ridge
  weights per point and lag, shown on a map. If advection matters, the points that help should lie upwind
  (north-west/west) and the useful lag should grow with distance. A "control direction" summary compares points
  to the north-west/west with points to the south-east/east.
- Upstream soil moisture: ERA5 gives tiny negative values (-0.001 to -0.003) on 29 days at 3 dry points; they are
  clipped to 0 in the dataset (not used by this gate).

**Gate G-D0 result: PASS (2026-10-07), the graph backbone stays**
- Delhi Tmax, all days, leads 1-3: RMSE 2.009 (damped persistence) -> 1.898 (+ 27 upstream points, 3 lags, ridge);
  Δ -0.111 °C [-0.144, -0.077], p < 0.001. The gain is significant at every lead 1-5, and on extreme days
  (leads 1-3: Δ -0.233 [-0.333, -0.133]). Physical WBGT (descriptive): Δ -0.167 [-0.210, -0.125] on all days;
  on extreme days not significant (Δ -0.113 [-0.294, +0.067]).
- Every point helps on its own. The strongest are to the west and south-west (Thar / Kutch, 430-1050 km;
  bearings 227-266°); the weakest are next to Delhi, whose information is already in Delhi's own history.
- Not seen: best lag is 1 day at 26 of 27 points, so this linear check shows no travel time that grows with
  distance. The signal looks like the large-scale heat pattern a day earlier. The east-vs-west contrast is weak by
  design (only 1 point east/south-east). G-D3 must show the graph beats an LSTM given the SAME upstream data
  (flattened) before any claim about graph structure or advection.
- Report: `evaluation_v2/gd0_upstream.md`; map: `evaluation_v2/figures/gd0_upstream_map.png`.

**Gate G3 result: NO rung selected (2026-10-07), retrieval does not help yet**
- Runs: R0, R0-rand, R1, R1-rand on each control (`A1prime_hw5`, `A2Lr_hw5`), 10 seeds x 4 folds, scored out of
  fold on 2007-2018. R1-rand (mode `time_rand`: random draws from R1's own +-30-day pool, seeded per fold and seed)
  was added after the first 6 runs, as the amendment required; the 192 analogue-list fingerprints of those runs
  were re-checked unchanged first. Code: commit 312ec5f (code_sha256 49129b3d..., R0/R0-rand/R1) and 5a3cb99
  (a99ec74b..., R1-rand). The runs were trained before that code was committed (registry git_commit 6fad43b,
  dirty); the code hash identifies the exact code.
- Tmax: both rungs slightly WORSE than the control on extreme days (R0 +0.072 [-0.050, +0.194], R1 +0.022) and no
  better than their random controls (R0 vs R0-rand +0.070 [-0.102, +0.242]; R1 vs R1-rand +0.015 [-0.160, +0.189]).
- WBGT (physical): every variant, the random ones included, is slightly better on extreme days but none
  significantly (R0 -0.064 [-0.139, +0.010]; R1 -0.065 [-0.133, +0.002]; R0-rand -0.044; R1-rand -0.038). Rung vs
  its random control: R0 -0.020 [-0.107, +0.066]; R1 -0.028 [-0.087, +0.032]. So any gain is not attributable to
  the retrieved information. All variants are ~0.015 °C worse on all days; WBGT R1 significantly so (p = 0.045),
  failing condition 1 as well. On WBGT the retrieval variants forecast more hot days with a larger warm bias than
  the control (R0 75/seed, +1.96; R1 77, +2.03; control 54, +1.61): part of the small extreme-day gain looks like
  forecasting warmer.
- Power: the extreme-day test can detect differences of about 0.24 °C (Tmax) and 0.08-0.12 °C (WBGT) at 80%
  power; smaller benefits are not ruled out. 2 families x 2 rungs x 3 conditions, no multiple-comparison
  correction (cannot turn this fail into a pass).
- The non-neural analogue ensemble (AnEn) is far worse than the control (+1.13 Tmax, +1.52 WBGT on extreme days):
  the raw analogue outcomes carry little skill on their own.
- Independent review: signed off with changes (numbers reproduced to 4 decimals from the prediction files; R1-rand
  verified as a valid control; no leakage found). Applied: p values and a per-condition pass/fail column in the
  report (two all-days failures were hidden by rounding), detectable-effect sizes, the caveats above, and a missing
  CI bound now fails condition 1 too (it already failed conditions 2 and 3; no effect on this result).
- Consequence: no rung is attached to anything yet. Week 4 continues the ladder (R2 diversity, R3 drift gate, R4
  climate-shift correction), each judged by the same rule against its own random control, plus mechanism metrics
  (analogue age, redundancy, season mismatch) to show why retrieval does or does not help. Which retrieval, if
  any, goes into the final model (G-R*) is decided after R2-R4.
- Report: `evaluation_v2/week3_controls.md` (retrieval sections).

**Retrieval information check: SPECIFIED 2026-10-07, before it was run** (user asked whether tuning could help)
- Question: do the retrieved analogues' outcomes carry information about the query's next 5 days BEYOND what the
  query's own 14 input days already say? If yes, the network is not using it (tune how it is fed in); if no, tune
  how analogues are matched, or accept the null. No training; no validation block (2007-2018 val years) is used.
- Data: per fold and family (Tmax/v2 labels; physical WBGT/WBGT labels), the fold's TRAINING windows only.
  Queries with a full set of 5 analogues. Analogues from `FoldRetriever` exactly as in the runs: R0 (sim), R1
  (time), and their random versions (rand, time_rand; seeds 0-9, averaged).
- Signal: the mean standardised anomaly of the 5 analogues' own next 5 days (the AnEn signal), per lead.
- Baseline: ridge (alpha 1, fixed) of the query's standardised target anomaly per lead on the query's own inputs:
  v1's 17 window features plus the target's last-day and 14-day-mean standardised anomaly. Augmented: the same plus
  the analogue signal. Fitted on all training queries before the fold's last 2 training years; scored on those last
  2 years (8 held-out years over the 4 folds), in °C (anomaly x the fold's climatological SD + mean).
- Reading rule: "information present" if, pooled over folds and leads 1-5, the augmented model's held-out RMSE on
  all days is lower than the baseline's with a year-cluster bootstrap 95% CI entirely below 0, AND the gain for R0
  (or R1) is larger than for its random version (CI of the difference entirely below 0). Extreme days and each lead
  are descriptive (few events). Only 8 year-clusters: fragile, and stated as such.

**Retrieval information check result (2026-10-07): the information is NOT there; tuning the feeding will not help**
- Pre-specified rule: information present for neither rung in either family. Pooled held-out RMSE change from adding
  the analogue signal to the query-only linear model: Tmax R0 -0.001 [-0.002, +0.000], R1 -0.000; WBGT R0 +0.000
  [-0.002, +0.003], R1 +0.000. Rung vs its random version: the same, about 0.
- Why: the analogue signal DOES track the truth (correlation at lead 1: Tmax +0.69, WBGT +0.28; random analogues
  about 0), but only because analogues are matched on the query's own recent state. What followed them is what the
  query's own inputs already predict, so it adds nothing new. Retrieval as matched here repeats information the
  model already has.
- Descriptive only: on WBGT extreme days R0 shows a tiny gain over its random version (-0.011 [-0.021, -0.001] °C);
  too small to matter.
- Sanity check: the query-only linear model scores 2.185 °C (Tmax, all days), close to the LSTM control (2.175 on
  the validation blocks), so the comparison baseline is realistic.
- Consequence: tuning K, the input form or the attention design is not worth it. A retrieval variant can only help
  if it matches on information the query's own inputs lack (e.g. the regional upstream pattern that G-D0 found
  useful), or if the R2-R4 ladder changes what is retrieved. Report: `evaluation_v2/retrieval_information_check.md`.

**Regional-pattern retrieval rung Rg: PRE-REGISTERED 2026-10-07, before any Rg number was computed** (user approved)
- Why: the information check found that analogues matched on Delhi's own recent weather add nothing the query's own
  inputs lack. Matching on the REGIONAL pattern (which G-D0 showed carries extra forecast information) might.
- Matching (mode `region`; no tuning): per window, the standardised anomalies on the last 3 input days of daily Tmax
  at the 27 upstream points plus Delhi (84 values). The WBGT family adds the standardised anomalies of daily mean
  dew point at the 27 points plus Delhi's daily mean relative humidity (Delhi has no dew point in the daily table),
  same 3 days (168 values in total). Climatologies from the fold's training years only (`pipeline.climatology`, as
  G-D0). Each value is z-scored over the fold's candidate windows, then cosine similarity, as R0.
- Everything else as R0: same candidate pool and eligibility, same dedup, K = 5, same model and trainer.
- Random control: R0-rand (its pool is identical to Rg's), so no new random runs.
- Step 1, screen (no training, training years only): the information check, with the same pre-specified rule
  (held-out RMSE gain vs the query-only baseline with CI entirely below 0, AND larger than R0-rand's with CI entirely
  below 0). If information is present for a family, step 2 runs that family's Rg config; otherwise Rg is recorded as
  not worth training for that family and is not run. Descriptive: the same check with the upstream readings added to
  the baseline (predicts whether Rg could help the graph backbone, which sees upstream data anyway).
- Step 2, G3 rule unchanged: Rg helps only if (1) not significantly worse than the control on all days, (2) extreme
  days significantly better than the control, (3) extreme days significantly better than R0-rand. If Rg and earlier
  rungs both help, lowest extreme RMSE wins, ties within 0.02 °C go to the simpler rung (R0, then R1, then Rg).
- Added after G3 had been seen (G3 = none): reported as a post-G3 exploration, with that disclosed.

**Rg screen result (2026-10-07): information PRESENT in both families -> both Rg runs started**
- Held-out (training years) RMSE change from adding the Rg analogue signal to the query-only baseline: Tmax -0.028
  [-0.037, -0.018] °C, WBGT -0.039 [-0.051, -0.028]; vs R0-rand the same. Extreme days (descriptive): Tmax -0.120,
  WBGT -0.087. Correlation of the Rg signal with the truth at lead 1: Tmax +0.74, WBGT +0.47 (R0: +0.69, +0.28),
  and the gain over R0 grows with lead.
- Descriptive, important for G-R*: when the baseline ALSO gets the upstream readings directly, Rg adds about nothing
  on all days (Tmax -0.001 [-0.002, +0.000]; WBGT +0.000) and little on Tmax extreme days (-0.012). Rg's extra
  information is the regional pattern itself, which a model given the upstream data already has. Expectation: Rg
  may help the LSTM control (Delhi-only inputs) but little on the graph backbone. To be tested, not assumed.
- Runs: `A1prime_hw5_Rg`, `A2Lr_hw5_Rg` (configs; queue `configs/week4_rg_queue.txt`), code_sha256 f1dd5eae...
  (uncommitted when started; to be committed after, as before).

**Rg result (2026-10-07): real all-days gain for WBGT, but G3 still selects nothing** (post-G3 exploration, disclosed)
- WBGT (physical), Rg vs control `A2Lr_hw5`: all days -0.044 [-0.071, -0.016] °C (p = 0.003, about 2%), in all 4
  folds; extreme days -0.160 [-0.307, -0.013] (p = 0.035), in 3 of 4 folds (f3 worse). Rg vs R0-rand: all days
  -0.059 (p < 0.001), normal days -0.058 (p < 0.001), extreme days -0.116 [-0.265, +0.032] (p = 0.114) -> condition 3
  fails, so G3 = none. Condition 3 is NOT relaxed after seeing p = 0.114 (that would make the rule outcome-dependent).
  The extreme-day test detects about 0.21 °C at 80% power. The all-days results survive a Bonferroni correction for
  2 families x 3 rungs; the extreme-day p = 0.035 does not.
- Where the WBGT extreme-day gain comes from: entirely a smaller cold bias on those days (mean error -3.02 -> -2.81
  °C; spread 1.28 -> 1.34, slightly worse). All-days bias is unchanged (+0.44), so it is not a uniform warm shift.
  Rg forecasts many more hot days (147 vs 54 per seed): about 18 extra hits and 75 extra false alarms per seed.
  Both models still catch only a few % of the observed hot lead-days. Rg is still not significantly better than
  damped persistence (-0.039 [-0.112, +0.033]).
- Tmax, Rg vs `A1prime_hw5`: all days -0.003 [-0.029, +0.023]; extreme +0.045 [-0.090, +0.180]: no detectable gain,
  though the all-days interval still includes the screen's -0.028. Possible reasons (hypotheses, untested): the Tmax
  control gets analogue outcomes in raw units while the screen used standardised anomalies (the WBGT model's form);
  and Rg analogues are season-mismatched (35% within +-30 days, mean 66 days apart, vs 95% and 12 days for R0).
- Interpretation: consistent with Rg passing regional information INDIRECTLY to a Delhi-only model (the network
  never sees upstream data, only which past cases are retrieved). Whether Rg adds anything to a backbone that sees
  upstream data directly is the G-R* question.
- Independent review: signed off with changes (no leakage: region features read exactly q-1..q-3, date alignment
  checked, candidates use their own input days; numbers reproduced to 4 decimals). Applied: Rg post-G3 disclosure
  and corrected caveats in the report, an extreme-day bias column, a unit test that region features read only the
  last 3 input days. Provenance: Rg runs' code_sha256 f1dd5eae...; they read `datasets_v2/upstream_daily.parquet`
  (sha256 ab7b368579f5b85b...), which their registry data_sha256 does not include, and the trainer's CODE_FILES omit
  `pipeline/download_era5_upstream.py` (sets the node order). Both fixed for future runs in a separate commit.
