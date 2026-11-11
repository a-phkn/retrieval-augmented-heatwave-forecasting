# Decision Log: retrieval improvements (branch `retrieval-improvements`)

Why this branch: the mentor's feedback (meeting 2026-11-09, relayed 2026-11-11). Our retrieval never helped, while the
papers we based it on report gains: RAFT (Han et al. 2025, arXiv 2505.04163) and resolution-aware retrieval (Deznabi,
Kumar & Fiterau 2025, arXiv 2510.16695). She asked us to find out *why* (the inner workings, not only "the analogues
are too cold on heatwave days"), to look at other types of retrieval, and to try to make retrieval work. The plan
(Step 1 explains, Step 2 tries to improve) was approved by the user on 2026-11-11.

Rules for this file:
- An entry is **PRE-REGISTERED** once committed and pushed, before anything it describes is run. GitHub's push time is
  the timestamp.
- Frozen entries are never edited. A later change is a new dated **DEVIATION** line under the entry; results are added
  as dated **RESULT** lines.
- Everything computed is reported, including results that go against us.
- The user allowed (2026-11-11) skipping a step that does not look promising. A skipped step gets a dated **SKIPPED**
  line with the numbers that decided it.

**Common to every entry below. PRE-REGISTERED 2026-11-11 (plan approved by the user).**
- Data: development folds only. f1-f4 have validation blocks 2007-2009, 2010-2012, 2013-2015 and 2016-2018; each fold
  trains on all earlier years from 1980. Nothing from 2019 onwards is read. A later look at the test years would be its
  own entry, labelled post-test and descriptive. Nothing here changes the frozen models, H1-H4 or the advisory.
- These validation blocks already chose the project's models (G3, G-R*, ...). Results here are exploratory evidence
  about mechanisms, not confirmation. The clean test of anything new is a prospective season.
- Code: new files only, in `retrieval_study/` (results in `retrieval_study/results/`, tests in
  `tests/test_retrieval_study_*.py`). No file under training/ models/ pipeline/ retrieval/ evaluation/ configs/
  changes, so the test lock stays valid. New code may import the frozen modules. No new dependencies.
- Families: Tmax (`t_max`, labels v2) and WBGT (`wbgt_lj_max`, labels wbgt), as in the project.
- Statistics: the paired cluster jackknife (`evaluation/compare_v2.compare`), with year x season clusters on the
  target date. The 4 folds' validation blocks are pooled: 36 clusters on all days, 12 on heatwave days. "Helps" means
  the 95% CI of the RMSE difference is below 0. Strata: all days, and the extreme stratum (heatwave days by the
  family's label rule).

**RS-1: Why retrieval did not help here. PRE-REGISTERED 2026-11-11 (plan approved by the user).**
- Question: is the reference method (RAFT) any better on our data? Does retrieval help only when it carries
  information the base model lacks?
- **RS-1a/b, the testbed** (`retrieval_study/testbed.py`):
  - Model: ridge regression (multi-output, standardised columns, unpenalised intercept). It predicts the next 5 days
    of the target minus its last input day.
    - Ridge strength from {0.1, 1, 10, 100, 1000} and, for RAFT, the number of neighbours m from {1, 5, 10, 20}. Both
      are chosen together by all-days RMSE on the fold's last 2 training years, after fitting on the earlier training
      windows whose forecast ends before them.
    - The model is then refitted on all training windows (as the project's ridge baselines are) and scored on the
      validation block.
    - RAFT trains a linear model with Adam on the squared error. A ridge reaches the same family of forecasts (any
      linear map of the inputs) in closed form. Stated difference.
  - Bases (what the model knows), from the window's input days only:
    - **B0, own history only:** the target's 14 input days minus the last one (RAFT's input normalisation).
    - **B1, + season:** B0, plus sine and cosine of the first forecast day's day of year, plus the target's normal
      (the fold's training-year climatology) on each of the 5 forecast days minus the last input value.
    - **B2, + all Delhi inputs:** B1 plus every Delhi input channel of the project's models (the 10 base features and
      the target's own channels) on all 14 input days.
    - **B3, + region:** B2 plus the regional readings Rg matches on. These are standardised anomalies on the last 3
      input days of Tmax at the 27 upstream points and at Delhi; for WBGT also the 27 points' dew point and Delhi's
      relative humidity.
  - Retrieval arms, added to each base as extra inputs:
    - **none.**
    - **RAFT, as published** (Han et al. 2025, Sec. 3; `retrieval_study/raft.py`):
      - Key: the target's 14 input days. Similarity: Pearson correlation. Values: the 5 days after each candidate's
        window minus that window's last value.
      - The top m are weighted by softmax(rho / 0.1).
      - Retrieval is done separately on windows average-pooled over p = 1, 2 and 4 days. Keys are pooled from the
        window's end (14, 7 and 3 values); values from the first forecast day (5, 2 and 1 values). The 8 values enter
        the ridge.
      - Pool: the fold's training windows. A candidate whose 19-day window overlaps the query's is excluded (RAFT's
        rule). For validation queries every candidate lies in the past.
    - **RAFT-rand:** the same machinery with m random eligible candidates (RAFT's m; 10 seeds).
    - **R0:** the project's retrieval exactly as in the runs (`FoldRetriever`, mode "sim", K = 5). It matches on v1's
      17 window features by cosine similarity, with the chronological, episode and spacing rules. Values: the mean
      raw level of the 5 analogues' next 5 days minus the query's last input value. The roughly 60 earliest training
      windows (early 1980) have fewer than 5 analogues and are left out of R0's fitting.
    - **R0-rand:** the project's random control (mode "rand", K = 5, 10 seeds).
  - Comparisons: each arm vs none, and each real arm vs its random version, per base, family and stratum.
- **RS-1c, RAFT's add-on mode on our networks:**
  - For C3-hop and the Delhi-only LSTM of each family (`{A1prime,A2Lr}_hw5_C3hop`, `{A1prime,A2Lr}_hw5`), the network
    forecast is the 10-seed ensemble mean.
  - A ridge maps RAFT's 8 values to the network's error (truth minus forecast). It is fitted on the networks' saved
    forecasts for the last 2 training years, which the networks never trained on
    (`predictions_v2/inner/`). Its strength and m are chosen leave-one-year-out within those 2 years.
  - The correction is added to the validation forecasts. RAFT-rand gets the same setting.
  - Comparisons: corrected vs network alone, and corrected vs RAFT-rand-corrected.
- **RS-1d, diagnostics (descriptive):**
  - (i) Does a similar past mean a similar future?
    - For each validation query, 200 random eligible candidates (fixed seed).
    - Spearman correlation, pooled over the pairs, between key similarity and value similarity. Value similarity is
      minus the RMSE between the candidate's 5-day change and the query's actual 5-day change.
    - Computed for the RAFT key (Pearson of the 14 days) and the R0 key (cosine of the 17 features), on all queries
      and on queries whose 5 target days include a heatwave day.
    - RAFT reports 0.60 on its ETTh1 data. Its exact definition is not given, so the comparison is rough.
  - (ii) Can the key find the right analogues?
    - The oracle analogues are the 5 eligible candidates whose actual 5-day changes are closest to the query's
      (RMSE).
    - Reported: their median percentile rank under each key (50% = no better than chance, 100% = ranked first), and
      the error of the oracle forecast (last value plus the oracle mean change) next to the real retrieval's.
  - (iii) "Cold days", on heatwave-day rows (extreme stratum), by lead:
    - the bias of the R0 analogues' mean raw level against the truth;
    - the bias of RAFT's change-based forecast (last value plus the weighted change, p = 1, the m chosen on B2);
    - the gap between the query's last input value and the mean of its R0 analogues' last input values;
    - the share of rows where no eligible candidate's raw future at that lead is as hot as the truth.
- **Predictions (written now):**
  - **P1.** RAFT does not help B2 or B3 on all days (CI includes 0 or is above it) and does not beat RAFT-rand there,
    in either family. On B0 and B1 it may help a little; this is uncertain, because RAFT's key ignores both the time
    of year and the temperature level.
  - **P2 (the information principle).**
    - R0 helps B0 on all days, with a CI below 0, and beats R0-rand there: its analogues carry the time of year and
      the level that B0 lacks.
    - Its gain shrinks from B0 to B1, and on B2 and B3 its CI includes 0.
  - **P3.** On B2 and B3 every retrieval arm's all-days change is smaller than 0.01 °C in size.
  - **P4.** The RAFT add-on gives no all-days gain for either network (CI includes 0).
  - **P5.** The key-value Spearman correlation is well below RAFT's 0.60 for both keys, and lower on heatwave-day
    queries.
  - **P6.** The oracle analogues' median percentile rank under either key is below 90%, and not higher on
    heatwave-day queries.
  - **P7.** On heatwave days the R0 analogues' mean raw level is more than 1 °C below the truth, averaged over
    leads. RAFT's change-based forecast is less cold.
- **Reading rule:** the information principle counts as supported if P2 holds in both families. P1 and P3-P7 are
  descriptive. If P2 fails, the explanation offered to the mentor has to change, and that is reported as it comes out.
- Outputs: `retrieval_study/results/step1.{md,json}`.
- **RESULT 2026-11-11** (`retrieval_study/results/step1.md`). All-days numbers unless stated. Δ = RMSE change; negative
  = retrieval helps.
  - **P2, the information principle, holds in both families.**
    - Our own retrieval (R0) cuts the error of the model that knows only the target's last 14 days (B0): Tmax by
      0.148 °C (CI -0.170 to -0.126), WBGT by 0.143 °C. It beats random retrieval by 0.13 °C in both families.
    - Give that model the season directly (B1) and the gain is gone for Tmax (+0.003, CI includes 0). For WBGT a
      trace remains (-0.009, CI below 0), equal to its margin over random. R0's features include humidity, which B1
      still lacks.
    - With all Delhi inputs (B2) or the region as well (B3), R0 changes nothing: |Δ| ≤ 0.001 °C in both families.
  - **P1: RAFT as published adds nothing at any level, B0 included**: |Δ| ≤ 0.003 °C in both families, and it is
    never better than its random control. Its key ignores the level, and the shape of the last 14 days says nothing
    usable about the next 5 days' changes.
  - **P3 holds** (every retrieval on B2 and B3: |Δ| ≤ 0.001 °C).
  - **P4, RAFT's add-on mode on our networks:**
    - It fails as written for the Delhi-only WBGT LSTM: -0.041 °C on all days (CI -0.080 to -0.002).
    - But the random-retrieval add-on does the same: the difference is +0.001. In every fold the add-on chose the
      strongest ridge penalty (1000), which shrinks the retrieved values to almost nothing. What is left is a
      constant shift, a bias correction learned on the inner years. It is not retrieval.
    - Every add-on makes heatwave days worse (+0.15 to +0.51 °C), because it corrects towards the average, as the
      earlier analogue correction did.
  - **P5 holds. A similar past barely means a similar future here.** The key-value correlations are 0.02 to 0.08
    against RAFT's 0.60. On heatwave queries they are -0.04 to +0.05.
  - **P6 holds.**
    - The 5 past windows whose next 5 days really matched the query's (the oracle analogues) rank only at the 51st
      to 60th percentile under either key, against 50 for chance; on heatwave queries, 51st to 59th.
    - The pool does contain them: the oracle's own forecast has an RMSE of 0.29 °C (Tmax) and 0.37 °C (WBGT). Nothing
      in the last 14 days points to them.
  - **P7 holds, and it changes the "cold days" explanation.**
    - On heatwave days the R0 analogues' futures are 1.6 to 3.6 °C (Tmax, lead 1 to 5) and 4.4 to 4.9 °C (WBGT)
      colder than the truth. RAFT's change-based version is less cold (0.6 to 3.0 °C, and 2.9 to 4.2 °C).
    - The pool is not short of hot days: every heatwave-day value is matched or beaten by some past future (0% of
      rows without one).
    - The analogues also do not start much cooler: the query's last value is above its analogues' by 0.25 °C (Tmax)
      and 1.35 °C (WBGT) for heatwave-bound queries.
    - So the cold bias comes mainly from what follows: windows that look like today were usually followed by ordinary
      days. Nothing in the last 14 days at Delhi tells the coming heatwaves apart (regression to the mean). It is not
      that hot examples are missing.
  - **Reading.**
    - Retrieval adds what the model's inputs lack, and nothing else.
    - Our models' inputs already contain everything Delhi-only analogues could add: the season, the level, all
      Delhi variables and the region.
    - The reference method's key carries no information about the next 5 days' changes in our data.
    - The RS-1e positive control (hourly) tests whether this is about the setting or our code.

**RS-1e: RAFT in its own kind of setting, on our data (a positive control). PRE-REGISTERED 2026-11-11, written while
RS-1 was running and before any of its results were seen.**
- Why: if RAFT gives nothing on our daily task, a sceptic could blame our implementation. RAFT's gains come from
  high-frequency data with daily cycles. A July 2026 study (Cakiroglu et al., arXiv 2609.20193, abstract) finds
  retrieval's benefit depends on the input window relative to the dominant cycle. So the same code is run where those
  conditions hold: hourly temperature with a short window.
- Data: Delhi's hourly 2 m temperature, the 9-cell mean, IST hours (`pipeline.hourly_features.load_cell_hourly`).
  Rows from 2019 on are dropped right after reading and never used. Training 1980-2015, validation 2016-2018 (fold
  f4's years); the settings are chosen on 2014-2015 after fitting on the earlier years.
- Queries: issue times every 6 hours (00, 06, 12 and 18 IST). Training queries are 15,000 issue times drawn at random
  (fixed seed) from 1980-2013 plus every issue time in 2014-2015. Validation queries are every issue time in
  2016-2018.
- Candidates: windows starting every 3 hours that end within 1980-2015. A candidate whose window overlaps the
  query's is excluded.
- Window S = 12 or 96 hours; horizon 48 hours (the next 48 hourly values).
- Bases: **H0**, the S hours minus the last one (RAFT's input normalisation); **H1**, H0 plus sine and cosine of the
  issue hour and of the day of year (the phase given directly).
- Arms: none; RAFT as in RS-1 (Pearson; top m from {1, 5, 10, 20}; softmax(rho / 0.1); periods of 1, 2 and 4 hours,
  keys pooled from the end and values from the start: 48 + 24 + 12 values); RAFT-rand (3 seeds; the same m).
- Model and selection as in RS-1a. Score: RMSE over the 48 hourly values.
- Clusters: year x season of the issue date. There are 9 on validation, so the CIs are fragile and the result is
  descriptive.
- **Predictions (written now):**
  - **P8.** At S = 12 hours, RAFT helps H0 (CI below 0) and beats RAFT-rand.
  - **P9.** RAFT's gain on H0 is at least twice as large at S = 12 as at S = 96. On H1 it is smaller than on H0 at
    the same S.
- Reading: if P8 holds, our RAFT code works where the published conditions hold, and the daily null result is about
  the setting, not the code. If P8 fails, that is reported, and the RS-1 explanation has to lean on RS-1a-d alone.
- Code: `retrieval_study/hourly.py`; outputs `retrieval_study/results/step1e.{md,json}`.
- **RESULT 2026-11-11** (`retrieval_study/results/step1e.md`). 9 clusters, so the CIs are fragile; descriptive.
  RMSE over the next 48 hours.
  - **P8 holds: our RAFT code works where the published conditions hold.**
    - With a 12-hour window, RAFT cuts the error of the window-only model (H0) from 1.978 to 1.833 °C (-0.145, CI
      -0.194 to -0.097; 7%). It beats its random control by 0.106 °C (CI -0.137 to -0.075).
    - Random retrieval also helps a little (-0.039 °C): any past windows add the average shape of the daily cycle,
      which a straight-line model fed 12 hours cannot produce.
  - **P9 holds.**
    - With a 96-hour window, RAFT's gain on H0 is 0.029 °C (CI -0.046 to -0.012), five times smaller. Random retrieval
      adds nothing there.
    - Given the phase directly (H1), RAFT's gain shrinks: from 0.145 to 0.079 °C at 12 hours, and from 0.029 to
      0.019 °C at 96 hours. The sine and cosine of the hour are not the whole daily shape, so part of the gain
      remains.
  - **Reading.**
    - On the same data source, RAFT gives the kind of gain its paper reports when the input window does not contain
      the cycle that repeats. The gain shrinks as the window covers the cycle, and again when the model is told the
      phase.
    - On our daily task, the cycle that matters (the season) is given to the model directly, so there is nothing
      left for RAFT to supply (RS-1a: |Δ| ≤ 0.003 °C).
    - This agrees with Cakiroglu et al. 2026 (arXiv 2609.20193; we read its abstract only): retrieval's benefit
      follows the input window relative to the dominant cycle.

**RS-2 (written after RS-1's results, as its own entry before anything in it runs).** Candidates named now, in no
order. A candidate runs only if RS-1 gives a reason to expect it to help.
- RAFT-style retrieval inside C3-hop.
- A cross-site analogue pool: the 27 upstream points' own histories, to find more examples as hot as Delhi's
  heatwaves.
- Retrieval keyed on the 500 hPa pattern (1980-2018 data exist; Rz).
- Retrieval for forecast ranges: neighbours' past errors.
- A learned similarity.

**RS-2: Screens for retrieval that could help our best model. PRE-REGISTERED 2026-11-11, after RS-1's results and
before any screen was run.**
- **SKIPPED, with RS-1's numbers** (the user allowed skipping unpromising steps):
  - **RAFT-style retrieval inside C3-hop.** RAFT added nothing at any information level (|Δ| ≤ 0.003 °C) and never
    beat its random control. Its add-on to C3-hop was no better than random.
  - **A cross-site analogue pool.** It would fix a shortage of hot examples, and RS-1 found none: 0% of heatwave-day
    values lack an equally hot past future. The failure is that the key cannot pick out the right analogues (oracle
    ranks 51st to 60th percentile). A larger pool does not change that.
  - **A learned similarity on the same inputs.** On B2 and B3 every retrieval added nothing (|Δ| ≤ 0.001 °C). A
    learned key over the same information cannot add information the model does not already use.
- **Run: the two candidates that bring information C3-hop does not see, and retrieval for ranges.**
  - **Rz**, the 500 hPa pattern (the project's own retrieval, `FoldRetriever` mode "synoptic", K = 5, 1980-2018 data).
    - Signal: the mean standardised anomaly of the analogues' next 5 days. Random control: R0-rand, as in the runs.
    - Direct version: the 20 principal-component scores of the same 3-day 500 hPa anomaly fields that Rz matches on.
  - **LW, season-to-date analogues.**
    - Key: the target's standardised anomaly over the last 60 days. Distance: RMS. The m = 10 nearest training
      windows that do not overlap the query.
    - Signal: their mean standardised anomaly over the next 5 days. Random control: 10 random eligible windows (10
      seeds).
    - Direct version: the mean standardised anomaly over days -60 to -46, -45 to -31 and -30 to -15.
  - **KR, retrieval for ranges.** 80% ranges around C3-hop's 10-seed forecast, from its inner-block errors.
    - "knn" uses the 50 inner-block situations most similar to the query (R0's key) and the 10th and 90th
      percentiles of their errors per lead.
    - "global" uses all inner-block errors. "random" uses 50 random inner-block situations (10 seeds).
- **Two screens for Rz and LW, in both families:**
  - **Information screen** (the RS-1 testbed):
    - Arms: B3 alone; B3 + the signal; B3 + its random control; B3 + the direct version.
    - Ridge as in RS-1a; scored on the 4 validation blocks.
    - B3 stands in for what C3-hop sees; C3-hop sees more upstream variables, so this screen is generous.
  - **Relevance screen** (the add-on, `retrieval_study/screens.py`):
    - A ridge maps the signal to C3-hop's error. It is fitted on C3-hop's inner-block forecasts, with its strength
      chosen leave-one-year-out; the random control gets the same strength.
    - RS-1c showed that the add-on's constant shift alone changes the scores. So the decisive comparison is against
      the random control.
- **Pass rule (fixed now):**
  - A signal passes when both of these hold:
    - In the information screen, B3 + signal beats B3 alone and B3 + random, with both CIs below 0, either on all
      days, or on heatwave days without being worse on all days by more than 0.005 °C.
    - In the relevance screen, + signal beats + random, with the CI below 0, on the same stratum.
  - A signal that passes gets an end-to-end C3-hop run, written as its own entry (RS-3) before it runs. If none
    passes, step 2 ends here.
  - KR passes if its interval score (Gneiting & Raftery 2007) on heatwave days is below that of both the global
    ranges and the random-50 ranges (paired cluster jackknife on the mean score, CIs below 0), and its all-days
    interval score is not more than 2% above the global ranges'. Coverage and mean width are reported.
- **Predictions (written now):**
  - **P10.** Rz adds a little information on B3 for WBGT (the earlier screen found about -0.004 °C) and nothing for
    Tmax, and nothing on heatwave days. The direct version adds at least as much as Rz.
  - **P11.** LW adds nothing on B3 (|Δ| < 0.005 °C) in either family.
  - **P12.** KR covers heatwave days better than the global ranges. Whether its interval score is better is open.
- Outputs: `retrieval_study/results/step2.{md,json}`.
- **RESULT 2026-11-11** (`retrieval_study/results/step2.md`).
  - **Rz passes for WBGT and fails for Tmax.**
    - WBGT information screen: -0.011 °C on all days against B3 alone and against its random control (both CIs
      below 0). On heatwave days -0.034 °C (CI includes 0).
    - WBGT relevance screen: it beats the random add-on by 0.012 °C on all days (CI -0.020 to -0.005) and by 0.080 °C
      on heatwave days (CI -0.155 to -0.005).
    - As in RS-1c, every add-on is worse than C3-hop alone on heatwave days (+0.35 °C), because of its constant shift.
    - Tmax: -0.004 °C against B3 (CI includes 0) and no relevance.
  - **The information beats retrieval of it.** The direct version (the 20 PC scores of the 500 hPa fields) helps B3
    far more than Rz does:
    - Tmax: -0.038 °C on all days and -0.120 °C on heatwave days (both CIs below 0);
    - WBGT: -0.046 °C on all days.
    - Retrieval delivers about a quarter of that for WBGT and none of it for Tmax. Rz is worse than the direct
      version by 0.034 °C (Tmax) and 0.036 °C (WBGT) on all days, both CIs above 0.
  - **P10 holds:** a little for WBGT, nothing for Tmax, nothing on heatwave days, and the direct version adds at least
    as much.
  - **P11 holds. LW adds nothing** (|Δ| < 0.005 °C, every CI includes 0), and neither does its direct version.
    Weather beyond the last 14 days is not missing information.
  - **P12 fails, the wrong way. KR's ranges are worse on heatwave days.**
    - They hold the truth less often than the plain ranges: Tmax 70% vs 79%, WBGT 15% vs 41%.
    - The interval score is worse by 1.3 and 8.0 (CIs above 0).
    - The most similar past situations were mostly ordinary days with small errors, so the ranges narrow exactly
      where they should widen. This is the same regression to the mean as for the forecasts.
  - **Reading.**
    - The only retrieval that helps our best model is keyed on information the model lacks: the upper air. It helps
      only on ordinary WBGT days.
    - Giving the model that information directly does more, including on Tmax heatwave days in the linear screen.
    - Retrieval is a weak channel for information, not a source of it.
- **DEVIATION 2026-11-11 (written after the RS-2 result):** RS-3 is not run.
  - The end-to-end run the pass rule calls for, C3-hop + Rz for WBGT, already exists. It is `A2Lr_hw5_C3hop_Rz`,
    trained 2026-10-18 under its own pre-registered rule (`evaluation_v2/rz_gate.md`), with the same retrieval
    (`FoldRetriever` mode "synoptic", K = 5) and the same backbone.
  - Its development result stands in for RS-3:
    - all days -0.023 °C (CI -0.041 to -0.005); against random retrieval -0.020 °C (CI -0.040 to -0.000);
    - heatwave days +0.004 °C (CI -0.120 to +0.128).
  - Training it again would repeat that run. For Tmax, nothing passed.

**RS-4: C3-hop with the 500 hPa pattern as a direct input (not retrieval). PRE-REGISTERED 2026-11-11, after RS-2 and
before any run (user approved).**
- **Why.** RS-2's linear screen found the upper-air information worth more given directly than through retrieval:
  -0.038 °C (Tmax) and -0.046 °C (WBGT) on all days, and -0.120 °C on Tmax heatwave days. This tests it in our best
  model. It is the information principle's next step, not a retrieval method.
- **Model.** C3-hop exactly as frozen (`configs/{A1prime,A2Lr}_hw5_C3hop.json`), trained by the frozen
  `training.train_unified.train_one_seed_graph`.
  - Unchanged: graph, hops, hidden size, learning rate, dropout, hot-day weight 5, early stopping on the last 2
    training years, seeds 0-9.
  - The only change: 10 extra Delhi input channels on each of the 14 input days. They are that day's 500 hPa anomaly
    field scored on its first 10 principal components.
  - The fields come from `retrieval.fold_retrieval.synoptic_daily`: South Asia 10-45° N, 40-100° E, 12 UTC,
    standardised anomalies against the fold's training-year climatology, weighted by sqrt(cos lat).
  - The PCA uses the fold's training days only. Scores are divided by their training-day SD.
- **Runs:** 10 seeds x 4 folds, both families (`{A1prime,A2Lr}_hw5_C3hop_Z500d`). Predictions go in
  `retrieval_study/predictions/`; checkpoints in `retrieval_study/checkpoints/` (not committed).
- **Comparison:**
  - against C3-hop's saved forecasts (same seeds, folds and windows), paired cluster jackknife with 10 seeds each, on
    all days and heatwave days;
  - for WBGT, also against C3-hop + Rz (descriptive).
- **Rule:** the direct input helps if its all-days CI is below 0. Heatwave days are reported with their CI (12
  clusters).
- **Control (conditional):**
  - If it helps, the same run is repeated with each day's 500 hPa scores taken from a random other day (within the
    fold's years, within ±30 days of the same day of year). This rules out "more inputs" as the cause.
  - The direct input must then also beat that control on all days.
- **Predictions:**
  - It helps on all days in both families, but by less than in the linear screen (-0.01 to -0.04 °C), because C3-hop
    already sees 27 upstream points.
  - Heatwave days: no detectable gain for WBGT; open for Tmax.
- **DEVIATION 2026-11-11 (a clarification of the control, written before the control ran and before the main runs
  finished):**
  - Each year's dates take the scores of the same calendar dates in another of the fold's training years. There is
    one random donor year per year, never the year itself, with a fixed seed; 29 February takes 28 February.
  - Why this form of "a random other day": each 14-day sequence stays a coherent upper-air sequence, and the season
    is kept. Only the link to the right day is broken.
  - Code: `retrieval_study/z500_direct.py --control` (`*_Z500shuf`).
  - It runs for a family only if that family's main comparison helps (CI below 0 on all days).

**RS-5: retrieval keyed on a global model's forecast, an analogue ensemble (Tmax only). PRE-REGISTERED 2026-11-11, after
RS-2 and before any forecast value was downloaded (user approved).**
- **Why.** This is the textbook case where retrieval works in weather forecasting (the analogue ensemble, Delle Monache
  et al. 2013). Its key is a numerical forecast, which carries information about the future that past observations do
  not. By RS-1's principle it should work here, unlike same-site analogues.
- **Data:**
  - ECMWF IFS HRES operational forecasts from WeatherBench 2
    (`datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr`, 1.5°), 2 m temperature only. The
    store has no surface humidity or radiation, so there is no WBGT.
  - Runs started 12 UTC on q - 1 (the GraphCast yardstick's timing), 2016-01-01 12 UTC to 2018-12-26 12 UTC. Lead
    hours 12 to 126, every 6 hours.
  - About 3,300 chunks (~2 GB), streamed. Only the Delhi-box means are kept, in
    `retrieval_study/data/hres_2016_2018_delhi_t2m.parquet` (committed).
  - Nothing from 2019 on is downloaded.
- **Box:** the area-weighted mean over the 1.5° cells overlapping the 9-cell Delhi box
  (`evaluation.graphcast_yardstick.overlap_weights`).
- **Queries:** fold f4's validation windows with query dates 2016-01-02 to 2018-12-27 (2016-01-01 has no 12 UTC run
  the day before), leads 1-5.
- **Key:** the four 6-hourly HRES values (00, 06, 12 and 18 UTC) on the target day, standardised per lead over the
  archive.
- **Archive, leave-one-year-out:** the same lead's forecasts for target days in the other two years, within ±45 days
  of the same day of year. Each member's value is the ERA5 9-cell Tmax of its target day.
- **Forecast:** the mean value of the K = 20 nearest members (Euclidean). Random control: 20 random archive members (10
  seeds).
- **Compared with, on the same rows:**
  - C3-hop (10-seed ensemble mean, fold f4);
  - HRES's own daily maximum after the yardstick's per-month correction (fitted on ERA5 1980-2015);
  - the random control.
- **Diagnostic:** the key-value Spearman correlation of the forecast key, as in RS-1d.
- **Statistics:** paired cluster jackknife over year x season clusters. There are 9 on all days (fragile) and at most 3
  on heatwave days (no CI; descriptive). This is one fold and 3 years, so RS-5 is descriptive.
- **Predictions:**
  - The analogue ensemble beats C3-hop and its random control on all days (CIs below 0).
  - Its key-value correlation is far above RS-1's 0.02-0.08.
  - On heatwave days it is less cold than C3-hop (descriptive).
  - Against corrected HRES: open.
