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

**Physics head design: exact Liljegren (A) + per-cell head at each cell's own peak hour (D3): DECIDED 2026-10-07 (user)**
- Evidence (`evaluation_v2/physics_head_standin.md`, 2016-2018, perfect ingredients): a learned stand-in misses the
  exact formula by 0.40 °C at hot-day peak hours, larger than any retrieval gain measured so far, so the formula is
  exact (ported to torch, differentiable). Representation floors on hot days: 9-cell-average ingredients 1.08 °C;
  per cell at a shared hour 0.69; average + linear correction 0.71; per cell at each cell's own peak hour 0 (equals
  the target by construction). D3 chosen: the head predicts each of the 9 cells' ingredients at that cell's WBGT
  peak hour (incl. the peak-hour sun angle), applies the exact formula per cell, and averages.
- Fallback if D3 does not train: average head + learned correction (D4). The physics head is a rung judged against
  direct WBGT prediction; it goes into the final model only if it helps.

**Retrieval R2-R4 move onto Rg's regional matching, each screened first: DECIDED 2026-10-07 (user)**
- The information check showed R0's matching (Delhi's own features) adds nothing the query lacks, so R2 (MMR
  diversity), R3 (drift gate) and R4 (climate-shift correction) are built on Rg's regional matching instead. Each is
  specified here before it is computed, screened with the training-years information check, and trained only if
  the screen passes. Random control for each: R0-rand (same pool), unless the rung changes the pool. Post-G3
  exploration, disclosed.

**Graph backbone runs and gates G-D2 / G-D3 / G-D4: PRE-REGISTERED 2026-10-07, before any graph training** (user approved, incl. the tightened G-D3)
- Runs per family (Tmax: like `A1prime_hw5`; physical WBGT: like `A2Lr_hw5`): same target form, hot_weight 5,
  inner_2y early stopping, 10 seeds x 4 folds, same loss and optimiser recipe as the LSTM controls.
  C2 = DSTGNN graph "none"; C3 = "static" (geographic edges); C4 = "dynamic" (wind-gated advective edges);
  C4a = C4 + adaptive adjacency (reported next to C4); U1 = the control LSTM given the same upstream inputs
  flattened (Delhi features + 27 x upstream features per day). C1 = the existing controls.
- Upstream node inputs (fixed): 11 daily variables (dew point, relative humidity, shortwave radiation, top-layer soil
  moisture, surface pressure, Tmax, Tmean, Tmin, wind speed, wind u, wind v) plus the Tmax standardised anomaly
  (day-of-year climatology, fold training years). Each standardised per node and variable over the fold's training
  years. Delhi node: the control's own feature set, unchanged.
- Edges: day t's message uses day t-1's edges (model design). Advective weights from each source node's daily mean
  wind (speed from u/v, direction from u/v); Delhi's own wind direction is not in its table, so Delhi's outgoing
  edges use the mean u/v of its 8 neighbours.
- G-D4 compute: time one C4 seed-fold first. <= 15 min: run as planned; slower: the graph configs use 5 seeds
  (0-4), disclosed.
- G-D2 trainability, per graph config: training losses finite for every seed-fold; the early-stopping loss at the
  best epoch lower than after epoch 1 in >= 90% of seed-folds; seed-to-seed SD of the all-days RMSE <= 2x the
  control LSTM's.
- G-D3 (end of Week 5): the graph candidate is the lowest all-days RMSE among C3/C4/C4a that passed G-D2 (ties
  within 0.02 °C -> the simpler, C3 < C4 < C4a). It is the backbone only if, on all days, (a) vs the control LSTM
  and (b) vs U1, the upper end of the 95% CI of Δ RMSE (graph minus other) is below +0.05 °C (non-inferiority
  margin). If (a) fails: backbone = control LSTM. If only (b) fails: backbone = U1 (the gain is the data, not the
  graph). (b) is a tightening of the plan's G-D3, decided before results because Rg showed upstream data helps WBGT.
- Claims about graph STRUCTURE (advection) need C4 or C3 significantly better than U1 and than C2; reported
  descriptively otherwise.

**Retrieval rungs R2-R4 on Rg's regional matching: SPECIFIED 2026-10-07, before any R2-R4 number** (user approved the move onto Rg + screening)
- R2 (Rg + MMR diversity): from Rg's eligible pool, the 200 most similar candidates (regional similarity) are picked
  greedily by maximal marginal relevance, score = 0.7 x sim(query, c) - 0.3 x max over already-chosen s of
  sim(c, s) (regional vectors; lambda 0.7 fixed a priori, not tuned), with the usual dedup rules applied during the
  greedy pass; K = 5. Random control: R0-rand (same pool).
- R3 (Rg + one drift gate): per query, ADF and KPSS tests (statsmodels, already pinned; ADF autolag AIC, KPSS
  "c" with automatic lags) on the target's standardised anomaly over the 365 days before the query's first input
  day. Drift if ADF p > 0.05 OR KPSS p < 0.05 (as plan v5). If drift: Rg restricted to candidates from the last 15
  years before the query; otherwise plain Rg. Power check first: the firing rate on training queries; if it is
  below 5% or above 95% the gate is inert (R3 = Rg or = "recent-only Rg") and that is recorded instead of screening
  R3. Random control: random draws from the same (gated) pool, seeded per fold and seed.
- R4 (Rg + climate-shift adjustment): each analogue's outcome is shifted by beta x (query year - analogue year),
  beta = OLS slope of the target's heat-season (Mar-Sep) mean anomaly on year over the fold's training years. For
  Tmax beta is expected near 0 (negative control). Random control: R0-rand with the same adjustment (so the
  adjustment alone is not credited to retrieval).
- Screen (training years, the information check): a rung is trained only if (i) it passes the rule already used
  (gain vs the query-only baseline and vs its random control, both CIs entirely below 0) AND (ii) it adds to Rg
  (Δ RMSE vs Rg, CI entirely below 0), since each is built on Rg. If trained: same configs as Rg plus the change,
  G3 rule unchanged (random control as above), post-G3 exploration, disclosed.

**G-D4 result: PASS (2026-10-07)** - one C4 seed on fold f4 (the largest) took 3.2 min (20 epochs) on this CPU,
under the 15-min limit, so the graph configs run with all 10 seeds. Graph queue (`configs/week4_graph_queue.txt`,
10 runs) started with code_sha256 b5e778d6... (uncommitted; hashed files are not edited while it runs). It runs
fold by fold (a background-job cutoff loses at most one fold; finished folds are skipped on relaunch).

**R2-R4 screen result (2026-10-07): R2 passes (both families); R3 and R4 do not add to Rg** (`evaluation_v2/retrieval_ladder_screen.md`)
- R2 (Rg + MMR): vs query-only baseline Tmax -0.037 [-0.045, -0.028], WBGT -0.049 [-0.060, -0.039]; vs R0-rand the
  same; vs Rg Tmax -0.009 [-0.016, -0.001], WBGT -0.010 [-0.015, -0.006] -> train. MMR lowers the analogues' mean
  pairwise similarity (Tmax 0.79 -> 0.70, WBGT 0.73 -> 0.63), so the gain is consistent with less redundancy.
- R3 (drift gate): power check passed (firing rate Tmax 26.5%, WBGT 11.6%, inside 5-95%), but vs Rg +0.002 (Tmax)
  and -0.001 (WBGT), CIs include 0 -> not trained. Restricting to recent analogues when the series looks
  non-stationary adds nothing.
- R4 (climate shift): trend beta Tmax +0.04 to +0.10, WBGT +0.10 to +0.14 SD/decade (per fold); vs Rg -0.000 (Tmax),
  +0.002 (WBGT) -> not trained. Correcting old analogues for the warming trend adds nothing.
- R2 training runs (`A1prime_hw5_R2`, `A2Lr_hw5_R2`; mode to be added to retrieval/fold_retrieval.py) wait until
  the graph queue is done, because fold_retrieval.py is hashed and must not change while it runs. Random control:
  R0-rand.

**Physics head loss: WBGT loss + ingredient supervision, weight 0.1: DECIDED 2026-10-07 (user), before any physics-head training**
- Loss = the run's usual hot-weighted MSE on its target (in the target's normalised units) + 0.1 x the mean squared
  error of the 7 x 9 predicted peak-hour ingredients against `datasets_v2/peak_ingredients_daily.parquet`, each
  ingredient standardised by its fold-training-years mean and SD (per cell). Weight 0.1 fixed a priori, not tuned.
- Why: keeps the ingredients physically meaningful (interpretable drivers for the advisory, e.g. humidity-driven
  heat stress) and gives the head more to learn from; with WBGT alone any ingredient mix giving the right WBGT
  would do.
- Built (no training yet): `pipeline/wbgt_liljegren_torch.py` (exact formula, implicit-function gradients; matches
  the numpy reference, gradients checked against finite differences), `pipeline/build_peak_ingredients.py` (9-cell
  mean of peak WBGT = target to < 1e-9 °C), `models/physics_head.py`. First rung: WBGT family on the control LSTM vs
  `A2Lr_hw5`, wired into the trainer after the graph queue (hashed code).

**Physics head outputs Tmax too, trained on equal footing, and stays unless clearly worse: DECIDED 2026-10-07 (user), before any physics-head training**
- Gap found: the D3 head predicts each cell's temperature at its WBGT peak hour, not the daily maximum, so it did
  not output Tmax, which the 2026-10-05 condition requires (official alert tiers are Tmax-based).
- Tmax output: per cell, Tmax_c = (temperature at the WBGT peak hour) + softplus(learned gap) >= that temperature
  (the daily maximum can never be below any hour's temperature); Tmax = mean of the 9 cells' Tmax_c, which is
  exactly how the Tmax target is defined (each cell's daily maximum, then the cell mean).
- Loss (option ii): hot-weighted MSE on WBGT (WBGT label, the A2Lr_hw5 recipe) + hot-weighted MSE on Tmax (Tmax
  label, the A1prime_hw5 recipe), each in its own normalised units, + 0.1 x the ingredient term (now 8 per cell: the
  7 peak-hour ingredients and the cell's daily Tmax).
- Keep rule (reconciles 2026-10-06 "the head stays" with evidence): the head stays in the final model unless, on
  all days, it is worse than the separate direct models by more than 0.05 °C, i.e. the upper end of the 95% CI of
  Δ RMSE (head minus direct) is >= +0.05 °C for WBGT vs A2Lr_hw5 OR for Tmax vs A1prime_hw5 (same non-inferiority
  margin as G-D3). If it fails: two separate direct models (one Tmax, one WBGT), which still output both but
  without guaranteed consistency, disclosed.

**Physics-head rung run: PRE-REGISTERED 2026-10-07, before any physics-head training**
- Run `PH_lstm` (config to be added with the trainer hook after the graph queue): the control LSTM encoder + the
  physics head (`models/physics_head.py`), trained by `training/physics_train.py`; 10 seeds x 4 folds, hot_weight 5,
  inner_2y early stopping on the total loss, the LSTM recipe (batch 64, Adam 1e-3, 100 epochs, patience 10).
- Inputs: the union of the two control families' inputs (A2Lr_hw5's 14 columns + A1prime_hw5's 3 Tmax-climatology
  channels = 17), because one model forecasts both targets. Disclosed: each control saw only its own columns.
- Outputs scored separately: WBGT (WBGT label strata) vs `A2Lr_hw5`, Tmax (Tmax label strata) vs `A1prime_hw5`,
  same paired cluster-jackknife tests. Keep rule as decided: the head stays unless, on all days, the upper end of
  the 95% CI of Δ RMSE (head minus direct) is >= +0.05 °C for WBGT or for Tmax. Extreme-day results and the
  ingredient errors are reported, not pass/fail. On the final backbone the same rule is re-checked (G-R* stage).

**Retrieval mechanism metrics (2026-10-07, descriptive)** (`evaluation/retrieval_mechanisms.py`, `evaluation_v2/retrieval_mechanisms.md`)
- Attention is essentially uniform in every rung (unevenness 0.004-0.04 on a 0-1 scale; Spearman of attention with
  similarity +0.03 to +0.08 for retrieved analogues): the network averages its 5 analogues and does not prefer the
  most similar. (With random analogues it does lean towards the more similar ones: +0.12 to +0.46.)
- R0/R1 analogues are near-duplicates (mean pairwise similarity 0.88 in Delhi's feature space). Rg's are far less
  alike in Delhi space (0.34-0.38) but often off-season (mean 57-66 days apart; 35-41% within +-30 days, vs 95% for
  R0). Analogue age is similar for all rungs (14-17 years on average), so the warming correction (R4) had little to
  correct.
- Outcome signal (analogue outcomes vs the query's truth, out of fold): Rg strongest on all days at every lead
  (WBGT lead 1 +0.47 vs R0 +0.30; Tmax +0.74 vs +0.70). On WBGT observed-extreme days Rg's correlation is about 0
  (-0.08 to -0.02; correlations on extreme days are depressed by the restricted range, but R0/R1 stay slightly
  positive at lead 1). Consistent with the review: Rg's extreme-day gain came from a smaller cold bias, not from
  tracking extremes.
- Implication for the ladder: R2's MMR targets redundancy, which the metrics show is real for R0-style matching;
  fusion that discriminates between analogues (non-uniform attention) is not happening and would need a different
  design (not planned; noted for the paper).

**G-D3 across families, and a graph tuning round before any fallback: DECIDED 2026-10-07 (user), before any graph result was looked at**
- Cross-family rule (the final model is one backbone for both targets; implemented in `evaluation/graph_gates.py`):
  graph if (a) and (b) pass in BOTH families (if the families pick different graph configs, the WBGT family's is
  used, WBGT being the primary target); control LSTM if (a) fails in either family or no G-D2-passing candidate
  exists in a family; U1 if (a) passes in both and (b) fails in at least one.
- If the result is "control LSTM" or "U1", the user is told first and ONE tuning round is run before falling back
  (the user prefers the graph if it can be made to work):
  1. Tuned: the G-D3 candidate config (C4 if none passed G-D2), over a fixed grid of 8: hidden size {32, 64},
     learning rate {1e-3, 3e-4}, dropout {0, 0.2}.
  2. Selection only on each fold's last 2 TRAINING years (the inner early-stopping block), 3 seeds (0-2), lowest
     mean all-days RMSE (°C) over the 4 folds. No 2007-2018 validation number is used to choose.
  3. U1 gets the same tuning round (8 combinations: hidden {64, 128}, learning rate {1e-3, 3e-4}, dropout
     {0, 0.2}), so check (b) stays fair. The control LSTM is not retuned (all other comparisons rest on it); check
     (a) therefore slightly favours the tuned graph, disclosed.
  4. The tuned graph and tuned U1 then run once, 10 seeds x 4 folds, and the same G-D3 rule decides. No second
     tuning round: if the graph still fails, the fallback above applies. Cost ~15-18 h of background compute.

**G-D3 result: U1 (situation 3), 2026-10-08** (`evaluation_v2/graph_gates.md`, code b5e778d6, commit 8749dc6)
- Candidate C3 in both families (all four graph configs pass G-D2). (a) vs the control LSTM passes in both families;
  (b) vs U1 passes for Tmax (Δ -0.052 [-0.092, -0.013]) and fails for WBGT (Δ +0.076 [+0.042, +0.110]) -> U1.
- Edges hurt in both families: C3 / C4 / C4a vs C2 all days +0.08 to +0.10 °C (CIs entirely above 0). The edgeless
  C2 is the best model (Tmax 2.051 vs control 2.175, U1 2.199; WBGT 2.254 vs control 2.373, U1 2.270), but it is the
  no-structure control and not eligible under G-D3. No structure claim holds.
- Likely mechanism (a hypothesis, not tested): with edges, the forecast reads Delhi's node only, so upstream
  information must pass through Delhi's neighbours one hop a day and is averaged at every hop (over-squashing;
  Alon & Yahav 2021, ICLR); C2 reads every upstream state directly.
- The user was told before anything else was run, as agreed.

**Graph tuning round, with one added arm "C3-pool": DECIDED 2026-10-08 (user), AFTER the G-D3 result above**
- The agreed round runs unchanged (tuned C3 and tuned U1, grids and selection as above). The tuning is done per
  family (each family selects its own hidden / lr / dropout on its inner blocks); the architecture is what the
  cross-family rule decides.
- Added arm, disclosed as decided after seeing the result: C3-pool = C3's static edges + C2's readout (Delhi's state
  concatenated with the mean of the upstream states), tuned on the same 8-combination graph grid and selected the
  same way. It tests the over-squashing hypothesis: if edges only hurt because of the readout, C3-pool >= C2.
- Dropout in the DSTGNN (new; default 0 = the runs so far): applied to the node embeddings before the GRU and to the
  readout before the head. U1's dropout is the LSTM's existing inter-layer dropout (default 0.2).
- Decision after the 10-seed runs, applying G-D3 (a) and (b) in both families:
  1. tuned C3 passes -> backbone graph C3 (the agreed rule);
  2. else tuned C3-pool passes -> backbone graph C3-pool, reported as chosen after a post-result change; the
     locked 2019+ test run is its confirmation;
  3. else the agreed fallback, judged on tuned C3 (the agreed candidate): control LSTM if its (a) fails in either
     family, otherwise tuned U1. The tuned graph is never compared with an untuned U1.
  G-D2 is re-checked for each tuned graph arm; an arm failing it does not pass.
- Also reported: C3-pool vs C2 (all / extreme) -- do edges add anything once the readout sees the upstream nodes?
  C2 is not retuned, so this comparison favours C3-pool; disclosed.
- Order: tuning grids for both families (C3, C3-pool, U1) -> 10-seed runs of the 6 selected configs -> PH_lstm and
  the two R2 runs. ~28-30 h of background compute; safe to pause between folds.

**Independent ML review (2026-10-08) and the decisions taken on it: DECIDED 2026-10-08 (user), while the tuning
round was running and BEFORE any tuning result (selection or 10-seed) was looked at**
- A read-only reviewer found, among others (findings verified by the owner where marked):
  - (verified) the G-D0 ridge (Delhi's last anomaly + 27 x 3 upstream Tmax anomalies, per lead, alpha on the inner
    block, refit on all training years) has all-days RMSE 2.104 (Tmax) / 2.217 (WBGT): better than U1 and C3 in
    both families and than every neural model on WBGT all days; it is in no results table. On extreme days the
    hot-weighted LSTMs are far better (Tmax control 2.153 vs ridge 2.908).
  - the extreme-day stratum is dominated by conditional bias (WBGT control: extreme bias -3.02 °C, ~85% of the
    extreme MSE), so "better on extremes" is confounded with "forecasts warmer" (Lerch et al. 2017).
  - the backbone rule can return a dominated model (U1 is worse than C2 in both families).
  - (verified) tuning is not blind to the reported years: the f2/f3/f4 inner blocks (2008-09, 2011-12, 2014-15)
    lie inside the pooled 2007-2018 evaluation; selection reuses the early-stopping block; the trainer logs
    validation RMSE for tuning runs (never used to select, but visible).
  - the one-hop-per-day graph cannot carry the previous-day signal G-D0 found 500-1000 km away (likely more
    important than over-squashing); C2 has no node identity (a regional mean).
  - (verified) the physics trainer steps the optimiser on a non-finite loss; hard clamps zero the gradient.
  - neural models never see the last 2 training years (no refit; the ridge refits); +0.4 °C all-days warm bias;
    the test period has ~23 year x season clusters (~8 hot-season ones), so a confirmatory protocol is needed.
- **Backbone rule, amended (disclosed; replaces steps 1-3 of the tuning-round decision above):**
  - Candidates: tuned C3, tuned C3-pool, C2 (as run, 10 seeds; not retuned, disclosed), tuned U1, the control LSTM.
  - Eligible: non-inferior on all days (95% CI upper end < +0.05 °C) to the control LSTM AND to tuned U1, in BOTH
    families (the control and U1 are compared only with the other).
  - Among the eligible: the lowest WBGT all-days RMSE (primary target); ties within 0.02 °C go to the simpler
    (control < U1 < C2 < C3 < C3-pool). Tmax and extreme-day results reported for every candidate.
  - C2 is named a "regional node model (no edges)", not a graph; graph-structure claims stay governed by the
    structure-claim rule (CI entirely below 0 against both U1 and C2).
  - The ridge baselines (below) are reported next to every candidate but are not eligible (no encoder for the
    retrieval and physics-head rungs).
  - **Nothing is final until the user has seen the result and approved the choice.**
- **Physics head, before PH_lstm runs (robustness only; the design is unchanged):** skip the optimiser step on a
  non-finite loss (count skipped batches in the training log; any skip is reported); replace the hard clamps on t
  and pressure with smooth clamps (identity inside the range, non-zero gradient outside). Applied after the
  tuned runs finish (the files are hashed), committed, then PH_lstm and R2 run under the new code hash.
- **Reporting additions (evaluation only; no gate changes):**
  1. Ridge baselines in every results table: the G-D0 ridge ("ridge_up") and a hot-weighted version (weighted least
     squares, weights 1 + (hw - 1) x hot with the family's hot_weight 5), by lead and stratum, with paired tests
     against C2 and the final backbone.
  2. Bias split and calibration: extreme-day bias and error SD for every model; a recalibration control (per-lead
     linear a + b x forecast, fitted on each fold's inner block, from checkpoint predictions); the 10-seed ensemble
     mean as the forecast; a hot-day Brier score (Gaussian predictive distribution: mean = ensemble mean, SD from
     inner-block residuals).
  3. Selection-free-years check: G-D3 / backbone tests re-run on 2007, 2010, 2013, 2016-2018 only (years never used
     for early stopping or selection), next to the pooled result.
  4. A confirmatory test protocol (docs/CONFIRMATORY_PROTOCOL.md), frozen before the 2019+ test lock is opened.
- Not adopted now (possible later, disclosed exploration only): attention-readout graph arms, node embeddings, a
  Delhi-only DSTGNN ablation, refit-on-all-training-years sensitivity.

**Multi-hop graph arm "C3-hop": DECIDED 2026-10-08 (user), after the G-D3 result and the ML review, before any
tuning result; disclosed as post-result exploration**
- Defect found while designing it (verified by unit test, `tests/test_dstgnn_multihop.py`): in `models/dstgnn.py`
  with the Delhi readout, a message on day t carries neighbours' states of day t-1 one hop, so NO upstream point's
  last input day ever reaches the forecast, and a point h hops away contributes data at least h + 1 days old.
  G-D0 found the useful upstream signal is the last input day (best lag 1 at 26 of 27 points), 2-4 hops away.
  This, more than over-squashing, plausibly explains why the edged graphs lose to C2 (whose pooled readout reads
  every node's last-day state).
- C3-hop (`models/dstgnn_multihop.py`): C3's static edges; K = 4 hops per day inside the recurrence (DCRNN-style
  K-step diffusion, one weight per hop) and a final K-hop transport of the last day's states into Delhi before
  the head; Delhi-only readout, so upstream data still reaches the forecast only along the edges. K = 4 = the
  largest hop distance from any upstream point to Delhi (not tuned).
- Hyperparameters: tuned C3's selection, per family (no separate tuning). 10 seeds x 4 folds x 2 families.
- Joins the amended backbone rule's candidates (simplicity after C3-pool; must pass G-D2, (a) and (b)), and is
  reported against C3 and C2. Nothing is final without the user's approval.
- Runs after the tuned runs: the trainer hook (`hops` in the graph config) is a prepared patch applied together
  with the physics-head fixes, because the trainer and model files are hashed by the running tuning queue.

**Focused code review of the new code, and C3-hop changed to ring averaging: DECIDED 2026-10-10 (user), before any
tuning, tuned or C3-hop result was looked at**
- A read-only reviewer checked the multi-hop model, the prepared patches (applied to copies: every anchor matched),
  the tuning hooks and the new evaluation code. No blocker. Findings and what was done:
  - M1 (verified by the owner): DCRNN-style diffusion shrank a 4-hop point's weight into Delhi ~40x relative to a
    1-hop neighbour (A_hat^4 into Delhi: 3.1e-2 / 1.4e-2 / 3.6e-3 / 7.9e-4 for hops 1-4), and W_k is shared by every
    walk of length k, so a C3-hop loss to C2 could not refute "the last-day upstream signal matters". **C3-hop now
    uses exact-distance ring averaging** (shortest-path message passing, Abboud, Dimitrov & Ceylan 2022): node i
    receives sum_k W_k mean_{d(j,i)=k} s_j for k = 1..4, in the recurrence and in the final transport into Delhi.
    Rings come from the static topology (neighbour mask); C3's edge weights are not used. At initialisation the
    last-day gradient per hop distance is now balanced (0.0017 / 0.0013 / 0.0019 / 0.0027 for hops 1-4; test
    requires max/min < 5). Disclosed: rings make C3-hop a distance-structured relative of C2's single mean.
  - M2: the physics NaN guard now also skips a step on any non-finite gradient, and stops the run if more than 1%
    of an epoch's batches are skipped; behaviour tests replace the source-text test. The reviewer found the WBGT
    solver returns NaN only at the edge of the head's range (t = 55 °C with p = 850 hPa).
  - M3: C3-hop is optional in `evaluation/graph_gates.py --tuned`; the decision is reported both with and without it
    (without = the five candidates declared before the review).
  - M4: the structure-claim rule is computed for C3 / C3-pool / C3-hop.
  - Minor: `hops: true` and hops with dynamic edges rejected; a separate guarded launcher for the C3-hop / PH / R2
    stage (refuses to start before the tuned runs finish or with uncommitted code); ridge rows and ridge tests for
    every candidate in the tuned report; tuned runs in the ridge and calibration reports; the hot-day Brier now on
    in-season days only, with in-season sigma (Tmax BSS +0.28 to +0.34, WBGT ~+0.03, unchanged conclusions); the
    soft clamp is described as approximately identity (deviation ~e^-distance to the bound).
  - Noted, not changed: tuning selects on the early-stopping block (already disclosed); sqrt-Brier through the RMSE
    jackknife tests equal Brier but may be anti-conservative with hot days in few clusters (stated in the report).

**Results of the tuning round, C3-hop, the physics head and R2: 2026-10-12** (code 25995975 for tuning + tuned runs,
commit 0a39fa6 / code 3b0639c4 for C3-hop, PH_lstm, R2; all registry rows clean)
- Backbone, amended rule (`evaluation_v2/graph_gates_tuned.md`): eligible in both families U1 (tuned), C2, C3-pool,
  C3-hop; not eligible: control (worse than tuned U1), C3 (tuned; its WBGT selection equals the original C3).
  WBGT all days: U1 2.247, C3-hop 2.248, C3-pool 2.252, C2 2.254 (all pairwise CIs cross 0) -> **provisional pick U1
  by the 0.02 °C simplicity tie-break**, with and without C3-hop. Against U1, C2 / C3-pool / C3-hop are better on
  Tmax all days (-0.09 / -0.08 / -0.08, significant) and on WBGT extreme days (-0.27 / -0.27 / -0.17, significant);
  U1 is better only on Tmax extreme days (n.s.). Awaiting the user's approval.
- Graph structure: C3-hop (ring averaging) fixes C3's defect: vs C3 -0.075 (Tmax) / -0.098 (WBGT) all days,
  significant, confirming that the lost last-day upstream signal was the problem. But C3-hop = C2 (+0.015 / -0.006,
  n.s.) and C3-pool = C2: no structure claim holds (CI below 0 vs both U1 and C2 for none of C3 / C3-pool / C3-hop).
  The regional upstream data, read directly, is what helps; graph edges add no measurable skill over pooling.
  Selection-free years give the same picture.
- Ridge (reported): C3-hop vs ridge all days -0.039 (Tmax, n.s.) / +0.031 (WBGT, n.s.); extreme days -1.08 / -1.35.
- Hot-day Brier skill (in season, recalibrated): Tmax C3-hop +0.36, C3-pool +0.34, C2 +0.34, U1 tuned +0.31, control
  +0.28; WBGT +0.03 to +0.04 for every model.
- **Physics head: DROPPED by the keep rule** (`evaluation_v2/late_rungs.md`): WBGT vs A2Lr_hw5 +0.041 [+0.001,
  +0.082]; Tmax vs A1prime_hw5 +0.132 [+0.077, +0.187] (both upper ends >= +0.05). Training was healthy (0 skipped
  batches, all finite). As agreed: two separate direct models, disclosed.
- **R2: fails G3 in both families**, like Rg: WBGT all days -0.048 (p < 0.001) but extreme-day CIs vs the control
  and vs R0-rand cross 0; R2 vs Rg ~0 (-0.004 / -0.007, n.s.). No retrieval rung passes G3.

**Final backbone: C3-hop. DECIDED 2026-10-12 (user), after seeing the results above; disclosed deviation**
- The amended rule's tie-break gives U1 (four eligible models tied on WBGT all days within 0.02 °C). The user chose
  C3-hop instead: eligible in both families, statistically tied with the best (C2) on every measure, better than U1
  on Tmax all days (-0.076) and WBGT extreme days (-0.173), best Tmax hot-day Brier skill, and a genuine graph
  network (the project's stated backbone preference). Disclosed in the paper as: chosen after results, overriding a
  simplicity tie-break; C3-hop itself added after the G-D3 result; the 2019+ test run is its confirmation.
- Claim limits (binding for the paper): no graph-structure claim (C3-hop = C2 within noise); the upstream regional
  data, read so that every point's last input day reaches the forecast, is what helps. C2 and U1 are reported next
  to C3-hop everywhere.
- Not chosen: C4 / C4a (wind-gated edges). Why: G-D3 picked C3 as the candidate (C3 / C4 / C4a within the
  pre-registered 0.02 °C tie, so the simplest), the tuning round tuned only that candidate, and C3-pool / C3-hop were
  built on it for a clean comparison; C4 suffered the same last-day defect as C3 and was no better (Tmax 2.142 vs
  2.147, WBGT 2.338 vs 2.346); and G-D0 found no delay growing with distance (a large-scale pattern, not air
  advected point to point), so wind gating had little to work with. A wind-weighted ring variant was not tested.

**G-R*: retrieval on the C3-hop backbone. PRE-REGISTERED 2026-10-13 (user approved the design), before any G-R* run**
- Rung: Rg (best rung: WBGT all days -2%, beats random analogues; R2 adds ~0.01 and also fails G3, not used).
- Model (`models/dstgnn_retrieval.py`): the tuned C3-hop of each family, unchanged, plus Rg's attention at the Delhi
  readout. Query = a linear map of C3-hop's readout [s_T(Delhi) ; ring transport]; keys = a 1-layer LSTM (hidden =
  the family's C3-hop hidden size) over each analogue's 14 Delhi input days; values = a projection of [key ;
  the analogue's own 5-day outcome in model-target units]; head on [readout ; context]. Same masking as
  RetrievalAugmentedLSTMv2 (no eligible analogue -> zero context). Analogues are NOT passed through the graph
  (6x compute; the information check showed their value is in the outcomes). Same recipe: Adam, the family's lr,
  batch 64, 100 epochs, patience 10, weighted MSE, inner_2y early stopping, 10 seeds x 4 folds.
- Runs: `{A1prime_hw5,A2Lr_hw5}_C3hop_Rg` (mode region, K 5) and `{...}_C3hop_rand` (mode rand, K 5: R0-rand, the
  random control Rg was already judged against).
- Pass rule per family = G3's three conditions (`evaluation.compare_v2.g3_conditions`): (1) C3hop_Rg vs C3hop, all
  days: CI not entirely above 0; (2) vs C3hop, extreme days: CI entirely below 0; (3) vs C3hop_rand, extreme days: CI
  entirely below 0. "Retrieval helps the graph model" in a family only if all three hold; otherwise no claim. Not
  loosened afterwards. Expected power: extreme-day MDE ~0.2 °C, so a null is likely and will be reported as such.
- Consequence for the test protocol: the final model of a family is C3-hop + Rg only if G-R* passes in that family,
  otherwise C3-hop alone.
- Exploratory diagnostics (no claims; evaluator's question "which spatio-temporal part does RAG help"):
  (a) per-lead deltas C3hop_Rg - C3hop (temporal); (b) "upstream heat arriving" windows = mean of the 27 upstream
  points' Tmax anomalies on the last input day (Rg's own anomalies, fold-train climatology) in the top quarter of the
  fold's training windows; delta inside vs outside (spatial); (c) attention concentration: mean max weight and
  normalised entropy vs uniform, next to LSTM Rg; (d) 2x2 overlap: (C3hop_Rg - C3hop) - (LSTM_Rg - LSTM) with a
  cluster-jackknife CI (does the graph already carry Rg's information?). Fusion point stated: Delhi readout only.

**Phase 6 alert colours and action tiers. DECIDED 2026-10-13 (user, the recommended fallback after Part E)**
Part E (`sources/PHASE6_PART_E_RESULT.md`): no Delhi document maps actions to colours beyond the 12 known; Ahmedabad's
HAP (2019 update) gives 30, IMD's FAQ 2; 91 stay unspecified; Delhi HAP p.10 prints Orange and Red both as >= 45 °C.
- Colour levels: IMD's colour-coded heat warning definitions (IMD FAQ p.8, including Red = severe heat wave for more
  than 2 days or more than 6 heat/severe heat wave days). Delhi HAP's p.10 table is not used for levels (it cannot
  separate Orange from Red). How our forecast maps onto IMD's criteria is designed with the rule engine.
- The 91 unspecified actions: shown at any alert level (Yellow and above), because Delhi's HAP lists them for
  "during a heat alert"; tier_basis "delhi_any_alert". No escalation is invented.
- The 30 Ahmedabad colours: used, each shown as "precedent from Ahmedabad's Heat Action Plan, not Delhi policy"
  (tier_basis "ahmedabad_precedent"). The 12 Delhi/NDMA and 2 IMD colours as sourced.
- Optional later: an expert may assign tiers, recorded as expert judgement (name, date).
- Paper limitation: no Delhi document maps actions to alert colours.

**G-R* result: FAILS in both families. 2026-10-16** (16 folds, code e10914a8 = commit 15f791a, all registry rows clean;
`evaluation_v2/gr_star.md`, rule above, applied unchanged)
- Tmax: C3hop_Rg vs C3hop all days +0.018 [-0.006, +0.042]; extreme -0.021 [-0.199, +0.157]; vs random analogues,
  extreme +0.028 [-0.139, +0.195]. Conditions (True, False, False).
- WBGT: all days +0.002 [-0.013, +0.017]; extreme -0.018 [-0.106, +0.070]; vs random, extreme -0.036 [-0.117,
  +0.045]. Conditions (True, False, False). Extreme-day MDE 0.25 (Tmax) / 0.12 °C (WBGT).
- **Final test models: C3-hop alone** (A1prime_hw5_C3hop, A2Lr_hw5_C3hop).
- Diagnostics (exploratory, no claims): (d) overlap: Rg's WBGT gain on the Delhi-only LSTM (all days -0.044,
  extreme -0.160) disappears on C3-hop (+0.002 / -0.018); difference +0.046 [+0.023, +0.069] all days and +0.142
  [+0.012, +0.272] extreme (jackknife without seed variance; indicative). Consistent with Rg having passed regional
  information to a model that could not see it; C3-hop reads that information directly. Tmax: no Rg gain on either
  backbone; on C3-hop Rg is slightly worse at leads 1-2 (+0.026, +0.023, p ~0.04, uncorrected, 10 lead tests).
  (b) no difference between upstream-heat windows and other windows. (c) attention stays near uniform (normalised
  entropy 0.96-0.97; mean max weight 0.28-0.29 vs 0.20 uniform), slightly more concentrated than on the LSTM.

**Rz: retrieval on the large-scale (500 hPa) weather pattern. PRE-REGISTERED 2026-10-17 (user approved; post-G-R*
exploration, disclosed as such), before any z500 data was downloaded**
- Why: both backbones see only surface weather; Delhi heatwaves are driven by upper-level anticyclones (Ratnam et
  al. 2016, Sci. Rep.); the GraphCast yardstick's 40% advantage is largely its view of the whole atmosphere. Rz is the
  only retrieval option that brings information C3-hop lacks.
- Data: ERA5 geopotential at 500 hPa, 12 UTC daily, WeatherBench 2 (Rasp et al. 2024) 2.8125° grid
  (`1959-2022-6h-128x64_equiangular_conservative.zarr`), box 10-45 N x 40-100 E (12 x 21 cells), 1980-2018 only
  (2019+ not downloaded unless Rz passes). Geopotential height in m. Streamed; only the box extract is kept.
- Matching ("synoptic" mode): per cell, standardised anomaly vs the fold's training-years day-of-year climatology
  (as Rg); the last 3 input days (q-1..q-3), each cell weighted by sqrt(cos lat), concatenated; PCA fitted on the
  fold's training windows, first 20 components (unscaled scores); cosine similarity of the centred scores (pattern
  correlation of the truncated fields). Same pool, eligibility, episode and spacing rules and K = 5 as R0/Rg; random
  control R0-rand (the existing C3hop_rand runs are that control for the identical model).
- Screen (no training, training years only, evaluation.retrieval_information_check's method): baseline =
  query-only features + Rg's regional readings + C3-hop's own normalised upstream inputs (27 points x all features)
  on the last input day, i.e. a linear stand-in for what C3-hop already sees. Rz passes in a family only if, on all
  days, (i) baseline + Rz signal vs baseline and (ii) baseline + Rz signal vs baseline + R0-rand signal both have 95%
  CIs entirely below 0. Extreme days reported, not used.
- If it passes in a family: one run `{family}_C3hop_Rz` (C3-hop + Rz at the readout, models.dstgnn_retrieval, 10
  seeds x 4 folds), judged by G3's three conditions against C3hop and C3hop_rand. Only then can retrieval enter the
  test protocol. If it fails: stop; report; go to option 2 (analog error correction).
- **Rz screen result 2026-10-17** (`evaluation_v2/synoptic_screen.md`; extract 14,245 days x 252 cells, 20.5 MB,
  `data/external/era5_z500_southasia_1980_2018.parquet` + meta): **WBGT PASSES** (Rz vs baseline -0.004 [-0.006,
  -0.002]; vs R0-rand -0.004 [-0.006, -0.002]); **Tmax fails** (-0.002 [-0.005, +0.001] for both). Over the same
  C3-hop-like baseline, Rg adds nothing (+/-0.000), consistent with G-R*. Extreme days (descriptive): -0.012 (WBGT),
  -0.016 (Tmax), CIs cross 0. Rz signal correlation with the truth, lead 1-5: WBGT +0.32..+0.21, Tmax +0.40..+0.19.
  Per the rule, only `A2Lr_hw5_C3hop_Rz` is trained. Expected: the effect is small (screen -0.004 on all days), and G3
  needs a significant extreme-day gain (MDE ~0.12 °C), so a G3 pass is unlikely; run as pre-registered.

**Option 2, analog error correction ("AnEn-corr"). PRE-REGISTERED 2026-10-18 (user approved), before it was run;
post-G-R* exploration, disclosed** (Delle Monache et al. 2011, Mon. Wea. Rev. 139: 3554, analog post-processing)
- Why: on observed heatwave days the final models forecast far too cold (WBGT extreme bias ~-2.7 °C, Tmax ~-1.2 °C);
  retrieval may correct this state-dependently, without retraining.
- Corrected model: the final C3-hop of each family (A1prime_hw5_C3hop, A2Lr_hw5_C3hop), every seed separately.
- Pool of honest past errors for fold k (per seed): fold k's own inner-block forecasts (its 2 early-stopping years;
  not fitted on) plus the validation forecasts of earlier folds on dates the inner block does not cover; only rows
  whose target date is before fold k's validation start. (f1: 2 years ... f4: ~9 years.)
- Matching, per seed and lead: candidates = pool rows of the same lead whose target day of year is within +-45 days
  (circular); predictors = the forecast at this lead, the mean of the window's 5 lead forecasts, and the last observed
  value (q-1), all in °C, each standardised by its SD over the candidate pool; Euclidean distance; K = 20 nearest;
  correction = mean error (forecast - actual) of the K; corrected = forecast - correction.
- Controls: raw C3-hop; random (K = 20 random candidates from the same +-45-day set, seeded: isolates similarity from
  season); linear (per seed and lead, OLS actual = a + b x forecast on the whole pool).
- Pass rule per family: (1) vs raw, all days: CI not entirely above 0; (2) vs raw, extreme days: CI entirely below
  0; (3) vs random, extreme days: CI entirely below 0; (4) vs linear, extreme days: CI entirely below 0. All four, or
  no claim. Forecaster's-dilemma checks reported next to it: all-days deltas, extreme bias, and the forecast-
  conditioned hot days (count per seed, RMSE, bias) for raw vs corrected.
- If it passes in a family, AnEn-corrected C3-hop is offered as the retrieval hypothesis for the test protocol
  (user decides); otherwise reported as a null.
- **AnEn-corr result 2026-10-18: FAILS in both families** (`evaluation_v2/analog_correction.md`, rule applied
  unchanged). Tmax: vs raw all +0.005 [-0.040, +0.050], extreme **+0.729** [+0.584, +0.874]; WBGT: all -0.032
  [-0.080, +0.015], extreme **+0.901** [+0.753, +1.050]; it also loses to the random and linear corrections on
  extreme days. Why: conditional on the FORECAST, C3-hop is too warm when it forecasts heat (forecast-hot-day bias
  Tmax +0.89, WBGT +1.81 °C; mostly false alarms), so any correction conditioned on forecast information cools those
  forecasts; conditional on the OBSERVATION, extreme days were already too cold, so they get worse (extreme bias
  Tmax -1.25 -> -2.09, WBGT -2.69 -> -3.68). Regression to the mean / the forecaster's dilemma (Lerch et al. 2017):
  the observed-extreme cold bias is mostly limited predictability, not a correctable systematic error; only new
  information can reduce it. Side results (descriptive): the corrections cut forecast-hot-day error (Tmax RMSE 2.26
  -> 1.64) and false alarms (forecast hot days per seed 681 -> 306 Tmax, 201 -> 12 WBGT); the plain linear
  correction improves WBGT all days (-0.074 [-0.113, -0.035]), as the earlier recalibration check found.
- **Rz result 2026-10-18: FAILS G3 (WBGT)** (`evaluation_v2/rz_gate.md`; 4 folds, code 47ca47a6 = commit 9ba341a, clean
  rows). Conditions (True, False, False): vs C3hop all days **-0.023 [-0.041, -0.005] (p 0.012)**; extreme +0.004
  [-0.120, +0.128]; vs random, extreme -0.013 [-0.124, +0.097]. **Final test model stays A2Lr_hw5_C3hop.**
  Exploratory (no claim under the rule): the first retrieval that improves C3-hop on all days (~1%), vs random -0.020
  [-0.040, -0.000] (p 0.050), vs Rg -0.026 [-0.045, -0.007]; gains largest at leads 2-4 (-0.025..-0.030); attention
  more selective than Rg (max 0.325 vs 0.276, entropy 0.927 vs 0.967); no heatwave-day gain (extreme bias unchanged,
  -2.69). Data limit for any test use: WeatherBench 2's 2.8° ERA5 ends 2021-12-31 (1.5°: 2023-01-10), so Rz could
  be scored on at most 2019-2021 (2.8°) of the 2019-2026 test period without another data source.

**Test protocol H3 changed. DECIDED 2026-10-18 (user, option d), before the freeze**
- H3 is now "C3-hop beats ridge_hw5 (the ridge weighted like the neural loss) on WBGT extreme days" (development:
  -0.847 [-1.056, -0.639]). The old H3 (vs the plain ridge, WBGT all days) tied in development (+0.031 [-0.021,
  +0.082]) and is still reported, as exploratory. Why: the old H3 tested the one comparison where the backbone has no
  edge; H3 now asks whether the graph network beats a fair linear model on the days that matter. Extreme-day RMSE
  rewards warm bias, so the bias / SD split and the hot-day Brier score are reported next to it.

**Heatwave-information screen ("S2"): can new information, delivered directly or by retrieval, improve WBGT
heatwave days? PRE-REGISTERED 2026-10-18 (user approved the 12 points), before the new data was downloaded;
post-G-R* exploration, disclosed**
- Why: on observed WBGT heatwave days C3-hop is ~2.7 °C too cold, mostly limited predictability (AnEn-corr result).
  Only information the model lacks can help. WBGT heatwaves are humid heat; the 27 upstream points all lie to the
  north-west (dry advection). Moisture reaches Delhi from the Bay of Bengal / Gangetic plain (east) and the Arabian
  Sea (south-west), which the model cannot see. Rz showed the 500 hPa pattern carries some information.
- New data: 12 points on the same 2° ERA5 lattice, same 10 daily variables and source (Open-Meteo ERA5) as the 27,
  1980-01-01 .. 2026-09-06, all below 600 m: east 28N80E, 26N80E, 26N82E, 26N84E, 26N86E, 24N84E, 24N88E;
  south-west 22N70E, 22N72E, 22N74E, 22N76E, 20N72E. Raw files gitignored; 2019+ stays locked at read time.
- Screen method: as evaluation/synoptic_screen.py (training years; each fold's last 2 training years held out;
  bootstrap over the 8 held-out years; baseline = the C3-hop-like linear model: query-only features + Rg's regional
  readings + C3-hop's upstream inputs on the last input day), but the ridge is fitted with the neural loss weights
  1 + 4 x hot (as ridge_hw5), since the question is heatwave days. Arms, each added to the baseline:
  (A) humid-air direct: the 12 points' standardised anomalies (fold-train doy climatology) of dew point, relative
      humidity, Tmax and surface pressure on the last 3 input days;
  (B) Rh retrieval: K = 5 analogues matched (cosine) on (A)'s features, standardised on training windows, Rg's pool
      and rules; signal = mean of the analogues' next-5-day standardised anomaly; random control R0-rand;
  (C) heatwave-onset matching: the mean of the top-5 cosine similarities between the query's (Rg regional + (A))
      vector and the vectors of training windows whose 5 target days contain a WBGT-label episode onset (eligible
      by the chronological rule, not the query's own episode); control = the same with an equal number of random
      non-onset windows;
  (D) upper-air direct: the Rz state's 20 principal-component scores (500 hPa, last 3 input days).
- Pass rule per arm, WBGT family (Tmax reported, descriptive): (1) WBGT extreme days: baseline + arm vs baseline,
  95% CI entirely below 0; (2) all days: CI not entirely above 0; and for (B) and (C), (3) extreme days vs their
  random control: CI entirely below 0.
- If an arm passes, the follow-up run (pre-specified now): (A) C3-hop with the 12 points added as graph nodes
  (rings recomputed; 39 nodes); (B) A2Lr_hw5_C3hop_Rh (models.dstgnn_retrieval); (C) the onset score as an extra
  Delhi input; (D) the 20 PCs as extra Delhi inputs (needs 2019-2026 z500 from another source for the test, e.g.
  the Copernicus CDS, which needs the user's own account). Judged vs C3-hop: all days not worse; extreme days CI
  entirely below 0. Nothing that fails is trained.
- **S2 arm D (upper-air direct) result 2026-10-18: FAILS** (`evaluation_v2/heatwave_info_screen.md`, run with
  `--arms D` while the humid points download; deterministic, rerun unchanged with all arms later). WBGT extreme days
  +0.053 [-0.045, +0.140]; all days -0.040 [-0.058, -0.021]. Tmax (descriptive): extreme -0.081 [-0.161, +0.029], all
  -0.031 [-0.047, -0.015]. Not trained (rule: extreme-day gain). Descriptive note: fed directly, the 500 hPa pattern
  improves ordinary days in both families, more than via retrieval; it does not reach heatwave days.

**W: retrieval for heatwave WARNINGS (hot-day probability). PRE-REGISTERED 2026-10-18 (user approved), before it was
run; post-G-R* exploration, disclosed**
- Question: does "how many of the most similar past situations became hot days" make C3-hop's hot-day probability
  more reliable? The temperature forecasts are not changed. Scored by the Brier score (a proper score: it cannot be
  improved by always warning), in season only, as in evaluation/calibration_check.py.
- Model probability: exactly calibration_check's "recal" probability for C3-hop (10-seed ensemble mean, inner-block
  linear recalibration, Gaussian with the inner-block residual SD per lead, event = the hot-label rule).
- Analogue probability: K = 20 analogues per window from the fold's training windows (Rg's pool, eligibility and
  dedup); p_analog(lead) = share of those analogues whose own outcome day at that lead was a hot day (the family's
  label). Two arms: Rg ("region", usable for the whole test period) and Rz ("synoptic"; data end 2021).
- Blend: logistic regression on the fold's inner block (in-season rows, pooled over leads):
  logit P = a + b logit(p_model) + c p_analog. Controls: model-only logistic (logit P = a + b logit(p_model), same
  rows; the fair comparison, since the blend includes a recalibration) and the random-analogue blend (p_analog from
  20 random eligible windows, R0-rand, 10 seeds).
- Pass rule per arm and family: (1) blend vs model-only, Brier: CI entirely below 0; (2) blend vs random-analogue
  blend: CI entirely below 0. (Paired cluster-jackknife on sqrt-Brier, as calibration_check; disclosed there as
  possibly anti-conservative.) Reported: Brier skill vs climatology, reliability, and at P >= 0.5 the hits and
  false alarms per season. WBGT is the family the user asked about; Tmax judged the same way.
- If it passes, the blended warning is offered for the advisory (Phase 6) and as an exploratory test check.
- **W result 2026-10-18: FAILS for both arms in both families** (`evaluation_v2/warning_retrieval.md`). Tmax: Brier
  model-only 0.0544 (BSS +0.366); Rg 0.0552, Rz 0.0550 (sqrt-Brier deltas +0.0016 / +0.0012, n.s.). WBGT: 0.0571 (BSS
  +0.028) for model-only, Rg and Rz alike; the WBGT warning almost never reaches P >= 0.5 (~0.1 hits / season).
  Analogue hot-day shares add nothing beyond C3-hop's own probability. Not offered for the advisory.

**Range forecasts (option 2) and what the graph learned (option 3). SPECIFIED 2026-10-18 (user approved), before
either was computed; descriptive (no model choice depends on them); option 1 (upper-air input) dropped by the user**
- Range forecasts (`evaluation/range_forecasts.py`): for C3-hop and, for comparison, the Delhi-only control, each
  family: the 10-seed ensemble mean recalibrated on the inner block (as calibration_check), with (a) a Gaussian whose
  SD is the inner-block residual SD per lead, and (b) the empirical inner-block residuals per lead added to the
  centre. Scores on 2007-2018 out of fold: CRPS (all days, extreme days), vs a climatology forecast (fold-train
  day-of-year mean and SD, Gaussian); coverage of the 80% interval (10th-90th percentile) and of the 50% interval.
  Deliverable for the advisory: the 10th, 50th and 90th percentile per forecast day.
- What the graph learned (`evaluation/graph_explain.py`): occlusion on the trained C3-hop checkpoints (10 seeds x 4
  folds, validation windows): replace one upstream point's inputs by its training mean (0 after z-scoring) on all 14
  input days, and separately on the last input day only; the rise in RMSE (all days, extreme days) is that point's
  importance. Also per hop distance (all points at distance k occluded together) and Delhi's own inputs. Reported as
  maps and tables; no claims beyond "the model's forecasts depend on these inputs".
- **Range-forecast result 2026-10-18** (`evaluation_v2/range_forecasts.md`; descriptive; deliverable quantiles in
  `predictions_v2/range/`): C3-hop CRPS all days Tmax 1.082 (CRPSS +0.24), WBGT 1.191 (+0.14); better than the control
  on all and extreme days in both families (sqrt-CRPS, all p <= 0.004). Calibrated on all days (80% interval
  coverage 0.79-0.82, 50%: 0.49-0.55). Tmax extreme days: 80% coverage 0.78. WBGT extreme days: 80% coverage only
  0.23-0.25 (the cold bias exceeds the spread), stated as a limitation for the advisory.

**Exam code built (steps 2-4). 2026-10-18 (user approved the plan), before the freeze; nothing from 2019 on was read**
- `training/test_lock.py`: 2019+ is readable only if the tag `test-lock` exists, HEAD contains it, and no file under
  training/ models/ pipeline/ retrieval/ evaluation/ configs/ differs from the tag or is uncommitted.
- Guarded test folds (training/folds.py `TEST_FOLDS`, not in FOLDS): "test" (train <= 2018, inner_2y stopping on
  2017-2018, validation 2019-01-01 .. 2026-09-06) and "test2021" (validation to 2021-12-31, for the Rz side-check,
  whose 500 hPa data ends 2021). Readers of 2019+ (folds, graph_data, fold_retrieval, the z500 downloader's
  --test-years) all call test_lock.require() first. Checked on synthetic test-period rows: every statistic is
  identical whatever the test rows contain.
- `training/test_run.py` (one evaluation run; never retrains a model whose test forecasts exist): main = C3-hop and
  the control, both families (with early-stopping-block forecasts), + both ridges; extra = C2, U1 (tuned), C3-pool
  (tuned), LSTM + Rg, C3-hop + Rg; rz = 2019-2021 z500 download, then A2Lr_hw5_C3hop_Rz on "test2021".
- `evaluation/test_evaluation.py`: H1-H3 with Holm; exploratory as listed in the protocol; case-study windows fixed
  now: 2019-05-25..06-15, 2022-04-01..05-15, 2024-05-15..06-20 (widely reported Delhi heat events).
- **What-the-graph-learned result 2026-10-18** (`evaluation_v2/graph_explain.md`, map `figures/graph_importance.png`;
  descriptive; every rebuilt checkpoint reproduced its saved forecasts). Single points are redundant on all days
  (switching one off changes RMSE by -0.02..+0.002 °C; neighbours carry the same information), but groups matter:
  Tmax, a whole hop ring switched off costs +0.11..+0.24 °C on all days and +0.88..+1.36 °C on extreme days, the
  farthest rings (hops 3-4, ~600-1000 km) most; Delhi's own inputs +1.20 / +4.04. On extreme days the far-western
  desert points (68-72 E: Thar / Pakistan) matter most individually (up to +0.35 Tmax, +0.17 WBGT), consistent with
  hot dry air arriving from the north-west and with G-D0. WBGT leans on the nearest ring (hop 1, +0.045) on all days;
  on extreme days on hops 2-4 and Delhi (+0.33..+0.52). Occlusion is an out-of-distribution perturbation; read the
  numbers as dependence, not causation.
- **S2 result (all arms) 2026-10-19: every arm FAILS for WBGT; nothing is trained** (`evaluation_v2/heatwave_info_screen.md`;
  humid data 564/564 files verified, `datasets_v2/humid_daily.parquet` 17,051 days x 144 columns). WBGT extreme / all
  days: A humid direct +0.024 [-0.047, +0.114] / +0.005; B Rh -> +0.004 [-0.003, +0.011] / +0.000 (vs R0-rand
  +0.003); C onset matching -0.042 [-0.060, -0.023] / **+0.005 [+0.002, +0.007]** (beats its random control
  -0.028, but significantly worse on all days: the forecaster's-dilemma trade, fails condition 2); D upper-air
  +0.053 [-0.058, +0.136] / -0.040 [-0.059, -0.021] (the all-arms run requires full analogue sets for B, so D's
  numbers differ slightly from the D-only run; same decision). Tmax (descriptive, not a decision): A extreme -0.123
  [-0.286, -0.027], B -0.021 [-0.036, -0.011] (and vs R0-rand -0.021), C -0.022, D -0.079 [-0.159, +0.031]; not
  followed up (post hoc; effects far below the extreme-day detectable effect of a training run, ~0.25-0.3 °C).
- **Conclusion of the post-G-R* exploration (2026-10-17..19):** no source of new information, delivered directly or
  by retrieval, improves WBGT heatwave days beyond C3-hop; the upper-air pattern improves ordinary days. The final
  models stay C3-hop alone.

**Protocol FROZEN 2026-10-19 (user: "yes add H4 and go ahead with the freeze")**
- H4 added: LSTM + Rg (A2Lr_hw5_Rg) vs the Delhi-only control, WBGT all days (development -0.044 [-0.071, -0.016],
  p 0.003; test detectable effect ~0.05 °C). Holm family = H1-H4. A2Lr_hw5_Rg moved into the test run's main stage.
- Final models: C3-hop (both families), controls, both ridges, A2Lr_hw5_Rg (H4). No retrieval rung or physics head in
  the final models. `docs/CONFIRMATORY_PROTOCOL.md` frozen as the commit tagged `test-lock`.

**TEST RESULTS 2026-10-19 (one evaluation run under the frozen protocol, tag test-lock / commit 53992b4; trained
11:46-23:46, scored once; no deviation from the protocol)** (`evaluation_test/test_results.md`)
- H1 SUPPORTED: C3-hop vs the Delhi-only control, WBGT all days -0.127 [-0.180, -0.075], Holm p < 0.001 (development
  -0.125: replicated).
- H2 not supported: C3-hop vs control, WBGT extreme days -0.081 [-0.261, +0.100], p 0.337 (development -0.255; below the
  test's detectable effect ~0.45, as stated in advance).
- H3 SUPPORTED: C3-hop vs ridge_hw5, WBGT extreme days -0.718 [-0.895, -0.540], Holm p < 0.001 (development -0.847).
- H4 not supported: LSTM + Rg vs control, WBGT all days -0.042 [-0.080, -0.004], p 0.033, Holm p 0.066 (development
  -0.044: the effect size replicated, but it is not significant after the multiple-testing correction).
- Exploratory: Tmax C3-hop vs control all days -0.127 [-0.209, -0.045]; graph structure still no claim (C2 / C3-pool
  vs C3-hop ~0); C3-hop + Rg ~0 (both families); Rz side-check 2019-2021 -0.004 [-0.040, +0.032] (the development
  all-days gain did not replicate); LSTM + Rg worse on WBGT extreme days (+0.176, p 0.041); C3-hop gains largest at
  leads 1-3; hot-day Brier skill Tmax +0.28 (control +0.23), WBGT +0.07 (+0.07); 80% range coverage 0.77-0.78 all
  days, WBGT extreme 0.29. Test-period maxima: Tmax 45.3 / 44.3 / 46.2 °C (2019 / 2022 / 2024 windows).

**Post-test exploration 1: pretrained time-series foundation models. SPECIFIED 2026-10-20 (user approved), before
any of them was run; exploratory (the frozen protocol and its one test run are unchanged)**
- Question: do general-purpose pretrained forecasters (Amazon Chronos, Google TimesFM), used zero-shot on Delhi's
  own history, beat our models? Reported whatever the outcome.
- Packages (user approved; venv only, no pip cache): chronos-forecasting 2.3.2, timesfm 3.0.2 (+23 dependencies; no
  existing package changed). Weights from Hugging Face, cached inside the project (`.hf_cache/`, gitignored).
- Setting: for every test window of C3-hop (2019-01-01 .. 2026-09-06, both families), the model gets the target's
  daily series (Tmax, or physical WBGT daily max) for the 1,024 days before the query date, nothing else; it
  forecasts 5 days; the point forecast is its median (0.5 quantile). The daily series is rebuilt from ERA5 truth
  already in the committed test outputs (predictions_test/) plus the pre-2019 data, so the lock is not touched;
  the code lives in `post_test/`, outside the locked directories.
- Scores: RMSE all days and observed heatwave days (the same strata as C3-hop's test frame), compared with C3-hop
  and the Delhi-only LSTM (paired cluster-jackknife, year x season; the foundation models are one deterministic
  run, ours the 10-seed mean of per-seed RMSE), plus bias; climatology / persistence from the test registry.
- Caveat stated with the results: their pretraining corpora may contain ERA5-derived or other weather series from
  the test years (cut-off dates not fully documented).
- **Foundation-model result 2026-10-21** (`evaluation_post_test/foundation_baselines.md`; weights 1.7 GB in .hf_cache/,
  nothing written to the user's own cache). C3-hop beats both on every comparison, both families: WBGT all days vs
  Chronos-2 -0.129 [-0.207, -0.051], vs TimesFM 3.0 -0.137 [-0.216, -0.059]; WBGT heatwave days -0.799 / -0.806;
  Tmax all days -0.175 / -0.177; Tmax heatwave days -0.795 / -0.747 (all p <= 0.002). The Delhi-only LSTM ties them on
  all days (WBGT -0.002 / -0.010; Tmax -0.048 / -0.050, n.s.) and beats them on heatwave days (WBGT -0.72 / -0.73; Tmax
  -0.55 / -0.51). RMSE: Chronos-2 WBGT 2.387 / heatwave 3.822, TimesFM 3.0 2.396 / 3.828; Tmax 2.409 / 3.101 and
  2.411 / 3.053; their heatwave-day cold bias -3.5 (WBGT), -2.5 to -2.7 (Tmax). Exploratory, post-test.

**Forecast-to-IMD alert mapping: B + D. DECIDED 2026-10-22 (user)**
- B (IMD-style, adjusted for area-average data): heat-wave day = Tmax >= 40 °C and >= 3.0 °C above the day-of-year normal,
  or Tmax >= 45 °C; severe day = >= 4.5 °C above normal (the labels-v2 rule the models were trained and tested on).
  Colours follow IMD's persistence rules (IMD heat-wave FAQ p.8) on the recent observed days + the 5 forecast days:
  Yellow = heat-wave conditions on 2 days; Orange = severe for 2 days or heat wave for >= 4 days; Red = severe for > 2
  days or > 6 heat-wave/severe days in total; Green = otherwise. Our single area average stands in for IMD's ">= 2
  stations". Always labelled "IMD-style, adjusted for area-average data; not an official IMD warning". Why not IMD's
  numbers as written (A): ERA5's 9-cell mean damps peaks (46-year max 46.6 °C, so IMD's 47 °C severe rule can never
  fire; IMD's 4.5 °C rule gives 13 spells in 2007-2018 vs 34 with 3.0 °C).
- D: the calibrated hot-day probability (Tmax Brier skill +0.28 on the test) is shown next to each day's colour.
- Kept in reserve (C, if mentors want exact IMD equivalence): quantile-map the ERA5 box Tmax to a Delhi station record
  (NOAA GSOD, Safdarjung), fitted on 1980-2018, then apply IMD's rules verbatim; also usable to check B's alert frequency.
- WBGT: no IMD WBGT tiers; humid heat stays a separate, labelled note (decision 2026-10-05).

**Critical reviews and fixes: 2026-10-22 reviews, fixes APPROVED and applied 2026-10-25 (user: "go ahead with all from a to d")**
- Four read-only reviews (methodology, docs/reproducibility, code quality/tests, hygiene/security) found no critical
  problem, no leakage and sound statistics; H1-H4 recompute exactly from the committed forecasts.
- Nothing under the six locked folders was changed (the test lock still opens). Issues there are disclosed in
  `docs/KNOWN_DEVIATIONS.md` instead of fixed.
- Claims corrected in the README, `docs/PROJECT_GUIDE.md`, `docs/MODELS_EXPLAINED.md` and the glossary: H2's null is
  not only a power problem (realised detectable effect ~0.25 °C vs the development gain -0.26); H3 is ~96% a smaller
  cold bias against a ridge with fewer inputs; C3-hop ties the plain ridge on all days; ranges and warnings fail on
  heatwave days (80% range holds 29% WBGT / 54% Tmax; WBGT yes/no warning hit rate 9%); H4's gain is ~50% bias;
  H1 compares tuned vs untuned; the graph and foundation-model wording softened.
- New post-test, descriptive only and written after the test (not pre-registered): `post_test/test_period_extras.py`
  (10-seed ensemble, hit rate / false-alarm ratio / ETS, H2 realised MDE, bias shares, H1/H3/H4 without ERA5T).
  It cannot change the confirmatory result.
- Safeguards added outside the locked folders: tests of all four lock conditions on a throwaway repo (the refusal
  tests no longer skip when the tag exists); `data/TEST_MANIFEST.json` + tests fingerprint the test outputs and the
  data they read; CI fetches tags and runs `test_lock.require()`.
- Dependencies: post-test packages moved to `requirements-post-test{,.lock}.txt` (no version changed); Hugging Face
  revisions pinned to the commits used on 2026-10-21 (both repos unchanged since before then).
