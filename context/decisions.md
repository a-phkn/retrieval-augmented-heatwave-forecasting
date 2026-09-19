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