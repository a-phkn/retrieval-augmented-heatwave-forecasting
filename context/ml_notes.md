# ML Notes

Accumulated ML knowledge — models tried, hyperparameters, seeds, and actual
(not projected) results. Update after every training run.

## Models implemented
- **Persistence baseline** (`training/baselines.py`)
- **Climatology baseline** (`training/baselines.py`, reuses P1's `clim_mean_t_max` column)
- **LSTM baseline** (`models/lstm.py` + `training/train_lstm.py`) — trained, 5 seeds, all converged.
  **Uses weighted MSE loss (hot_weight=10), not plain MSE** — see "Why
  weighted loss" below for how this was decided.

## Baselines
- **Persistence:** forecast day t+k = value at day t (last observed T_max),
  repeated for all 5 forecast days.
- **Climatology:** forecast day t+k = train-only ±7-day-of-year climatology
  mean for that calendar day (same `clim_mean_t_max` column already computed
  by P1 — reuse it, don't recompute).

## LSTM baseline — current canonical spec
- 2-layer LSTM, hidden size 64, dropout 0.2.
- MLP head: 64 → 32 → 5.
- Optimizer: Adam, lr=1e-3. Batch size: 64.
- Up to 100 epochs, early stopping patience 10 on val loss.
- **Loss: weighted MSE on z-normalized target** —
  `loss = mean(weight * (pred-target)^2)`, `weight = 1 + 9*hot_mask`
  (`hot_weight=10`), where `hot_mask` flags forecast-day-instances that are
  actually "hot" per the project's own train-only-climatology + 1.5σ
  definition. This replaced plain MSE (see rationale below).
- 5 seeds; report de-normalized MAE/RMSE, mean ± std across seeds.
- **IMPORTANT for Step 5:** the future retrieval-augmented model must use
  this same weighted loss (hot_weight=10) to keep the "same training
  procedure, retrieval on/off only" comparison valid. See
  `context/decisions.md`.

## Results log

### Sanity baselines (`evaluation/sanity_baselines.json`), run 2026-09-17
| split | persistence MAE / RMSE | climatology MAE / RMSE | n |
|---|---|---|---|
| train | 1.888 / 2.555 | 2.060 / 2.654 | 13,131 |
| val   | 1.950 / 2.615 | 1.834 / 2.344 | 1,092 |
| test  | 2.053 / 2.772 | 2.243 / 2.870 | 2,802 |

Note: climatology beats persistence on val but not train/test — expected,
since persistence tracks short-term autocorrelation while climatology tracks
the seasonal mean; neither dominates universally.

### Why weighted loss: the plain-MSE result that motivated the switch

The first LSTM trained (plain MSE, otherwise identical spec) had excellent
overall fit but a specific, serious failure mode, discovered by adding a
heatwave-day detection metric (predicted/actual T_max thresholded against
train-only climatology mean+1.5σ, same rule that defines "hot day" elsewhere
in the pipeline) on top of the usual MAE/RMSE:

| | MAE | RMSE | R² | Precision | Recall | F1 | Bias on hot days |
|---|---|---|---|---|---|---|---|
| Plain MSE (superseded) | 1.553±0.013 | 2.028±0.007 | 0.894±0.001 | 0.643 | 0.048 | 0.089 | −2.716°C |
| Persistence (reference) | 1.950 | 2.615 | — | 0.222 | 0.227 | 0.225 | — |

Bias diagnostic showed why: mean bias (pred−actual) was **−2.716°C on actual
hot days** vs. **−0.025°C on normal days** (essentially unbiased). Plain MSE
is dominated by the ~95% of normal days, so the MSE-optimal function shrinks
extreme predictions toward the mean. Hot days clear the detection threshold
by only ~0.6°C on average, so a −2.7°C systematic undershoot was enough to
collapse recall to 4.8% — **worse than persistence's 22.7%**, despite the
LSTM winning decisively on every aggregate metric. This was the strongest
evidence for the project's central hypothesis (a plain forecaster fails
exactly where it matters most) — but the user reasonably asked whether this
could be fixed rather than just documented, which led to the weighted-loss
change below.

### Current canonical LSTM baseline (weighted MSE, hot_weight=10)

`evaluation/baseline_lstm/val_metrics.json` +
`evaluation/baseline_lstm/val_extended_metrics.json`, run 2026-09-17.
5 seeds {0,1,2,3,4}, all converged via early stopping (13–20 epochs),
~107s total training time on CPU.

**Regression metrics (5-seed mean ± std):**

| MAE | RMSE | R² | MAPE |
|---|---|---|---|
| 1.660 ± 0.025 | 2.223 ± 0.035 | 0.872 ± 0.004 | 5.64% ± 0.10% |

**Per-horizon (MAE grows monotonically with lead time — sanity-check passed):**

| Day-ahead | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| MAE | 1.22 | 1.54 | 1.74 | 1.87 | 1.93 |
| RMSE | 1.62 | 2.05 | 2.33 | 2.47 | 2.53 |

**Heatwave-day detection** (pooled across 5 seeds, n=27,300 day-instances,
1,320 actually hot):

| | Pred. hot | Pred. not hot |
|---|---|---|
| **Actual hot** | TP=363 | FN=957 |
| **Actual not hot** | FP=678 | TN=25,302 |

**Full comparison, same detection framing, val split:**

| Method | MAE (overall) | Precision | Recall | F1 | Bias on hot days |
|---|---|---|---|---|---|
| Persistence | 1.950 | 0.222 | 0.227 | 0.225 | — |
| Plain-MSE LSTM (superseded) | 1.553 | 0.643 | 0.048 | 0.089 | −2.716°C |
| **Weighted-MSE LSTM (canonical)** | 1.660 | 0.349 | **0.275** | **0.307** | −1.555°C |

Bias on normal days shifted slightly to **+0.497°C** (from −0.025°C) — a
side effect of pushing predictions up to catch more hot days; still small
relative to the ~1.55°C hot-day bias it's trading against.

**Interpretation:** the 10x loss weight moved recall from 4.8% → 27.5% (now
above persistence) and F1 from 0.089 → 0.307 (3.4x), at the cost of a modest
drop in overall fit (MAE 1.553→1.660, R² 0.894→0.872). Given the project is
specifically about heatwave forecasting, this trade was judged worth it
(user's explicit call) — this is now the single canonical baseline model,
superseding the plain-MSE version, which was retired (not kept as a parallel
model).

**Still not fixed:** bias on hot days is smaller but still negative
(−1.555°C) — there's headroom left. `hot_weight=10` was a deliberate,
moderate first guess (true inverse-frequency weighting would be ~21.7x), not
tuned/swept.

**Not yet done:** test-split evaluation (per project convention, test is
touched once at the end, not per-milestone) — that's Step 6, owned jointly
with P3's retrieval system once it exists.

### Stratified RMSE — the actual headline result for this baseline (val split, run 2026-09-18)

Added after the "what metric actually matters for this project" discussion
(see `decisions.md`). Stratification: normal (anomaly ≤1.0σ), unusual
(1.0–1.5σ, or >1.5σ but not part of a qualifying ≥3-day episode), extreme
(>1.5σ AND part of a qualifying episode in `event_catalogue.parquet`).

| | Normal (n=4575) | Unusual (n=731) | **Extreme (n=154)** |
|---|---|---|---|
| Persistence | 2.614 | 2.705 | 2.179 |
| Climatology | 1.940 | 3.486 | **5.031** |
| **LSTM (canonical)** | **2.285** | **1.919** | **1.599** |

**This is a clean sweep** — the LSTM beats persistence on all three strata,
not just extremes, and beats climatology everywhere except the "normal"
stratum (where climatology's job is trivially easy — predict the seasonal
mean on a normal day). Climatology's collapse on "extreme" (5.03°C, 3x
worse than the LSTM, worse than persistence too) is a strong illustration
that a method with a fine-looking global MAE (1.83, actually better than
persistence's 1.95) can be completely blind to what matters — good panel to
keep front-and-center in any write-up.

**On the metric this project is being staked on:** LSTM's extreme-stratum
RMSE (1.599) is ~27% better than persistence's (2.179). This is the number
retrieval-augmentation needs to beat later, not recall — see the "which
metric matters" discussion in `decisions.md` for why detection recall was
downgraded from headline status.

**Caveat before calling this a paper result:** this is a val-split point
estimate with no confidence interval yet, and n=154 extreme instances is
thin. The bootstrap CI machinery already exists (built for the
canonical-vs-dual-head comparison, see below) — running it for
LSTM-vs-persistence on the extreme stratum is a cheap, worthwhile next step
before this number is defensible in a paper's results table. Test-split
numbers (not yet touched) are what actually go in a paper regardless.

### Metric pruning — narrowed to 4 metrics, others deliberately dropped

Following a discussion of what actually matters for this project's eval
(full reasoning in `decisions.md`), `training/evaluate_lstm.py` and
`training/visualize_results.py` were both rewritten to compute/plot exactly:
1. **Global MAE/RMSE** — credibility floor only, not a differentiator
2. **Stratified RMSE (normal/unusual/extreme)** — the headline metric
3. **Detection precision/recall/F2** (F2, not F1 — recall matters more than
   precision for this use case, so F1's equal weighting was actively
   misleading)
4. **Bias diagnostic (hot vs. normal day bias)** — explains why #3 looks
   the way it does

**Deliberately dropped:** R² and MAPE (redundant with RMSE for this
purpose), per-horizon (day 1-5) breakdown (a sanity check, not a
decision-relevant metric), plain accuracy (misleading under this class
imbalance — this is literally how the original 4.8%-recall problem was
discovered, since 95%+ accuracy was hiding it), F1 (replaced by F2), and the
raw confusion-matrix panel (folded into the precision/recall/F2 numbers).
`evaluation/baseline_lstm/val_extended_metrics.json`'s schema changed to
`1_global_accuracy` / `2_stratified_rmse` / `3_detection` /
`4_bias_diagnostic` keys to match. Both rewritten files were handed to the
user directly (not packaged in a zip) since only these two files changed.

### Dual-head model (focal-loss classification head) — explored, NOT adopted

Built in response to an explicit ask to push heatwave-day recall to
75-80%+. Architecture: shared LSTM encoder, two independent heads — an
unweighted-MSE regression head (freed from any detection duty) and a
separate classification head trained with sigmoid focal loss on the
project's own hot-day labels, with the decision threshold tuned post-hoc on
a validation precision-recall curve to hit an exact target recall.

**It worked exactly as designed** for its stated goal: 75% recall
achieved (confirmed at multiple thresholds, e.g. 80%→14.9% precision,
90%→12.2% precision), and regression MAE came back down to 1.566 (close to
the original best-ever plain-MSE result of 1.553, better than canonical's
1.660) — because the regression head no longer needed to hedge for
detection purposes at all.

**But it lost on the metric that actually matters.** Once evaluated on
stratified RMSE (built specifically to check this), the dual-head model's
regression head was **worse than canonical on both unusual (2.525 vs 1.918)
and extreme (2.646 vs 1.598) RMSE** — worse than even persistence on
extreme. Bootstrap 95% CI on the extreme-stratum RMSE difference (canonical
− dual-head) = −0.998°C, CI [−1.190, −0.808], **excludes zero**: canonical
is statistically, decisively better on the metric this project's
conclusions rest on.

**Why:** freeing the regression head from any detection duty meant it also
lost any *incentive* to be accurate on hot days specifically — it reverted
to the same tail-shrinkage behavior the original plain-MSE model had.
Fixing detection recall and fixing continuous-forecast accuracy on extreme
days turned out to be different problems; this architecture solved one and
inadvertently made the other worse. Genuinely useful negative result, not a
wasted detour — it's the reason the "which metric matters" conversation
happened and the reason detection recall got demoted from headline status.

**Decision:** not adopted. Canonical (`models/baseline_lstm/`, weighted-MSE
single-head) remains the baseline. The dual-head code (`models/lstm.py`'s
`DualHeadLSTMForecaster`, `training/losses.py`, `training/train_lstm_dualhead.py`,
`training/compare_models.py`) exists only in the working sandbox that
produced these numbers — it was shared with the user as inline code
snippets, not as files, and was never applied to their actual project. If a
future session wants to reconstruct it (e.g. to try a version where BOTH
heads use weighted/focal losses, which nothing has tested yet), the snippets
are in this conversation's history; the architecture idea (decoupled
detection head) may still be worth revisiting combined with a properly
weighted regression head, since neither piece alone was fully correct.

## Open questions / things to check once retrieval starts
- How sensitive is the extreme-stratum result to K (roadmap sweep {1,3,5,10,20})?
- Does the statistical-feature representation retrieve genuinely different
  analogues than a learned encoder would? (Step 7 ablation)
- Does retrieval degrade if restricted to the last 15 years only
  (non-stationarity check, Step 7)?
- **The actual bar retrieval has to clear:** extreme-stratum RMSE of 1.599
  (canonical LSTM, val). Recall/F2 are secondary diagnostics, not the target
  — see the dual-head write-up above for why optimizing recall alone can
  actively point you at a worse model.
- Untested idea worth revisiting: a dual-head model where BOTH heads use
  weighted/focal losses (not just the classification head) might get
  detection gains without the extreme-stratum regression regression the
  pure-unweighted version showed. Nobody has tried this yet.
- Is `hot_weight=10` well-tuned? Not swept — a quick sweep (e.g. 5/10/15/20)
  before committing further would be cheap.