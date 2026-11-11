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
