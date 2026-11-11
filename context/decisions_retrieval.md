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

**RS-2 (written after RS-1's results, as its own entry before anything in it runs).** Candidates named now, in no
order. A candidate runs only if RS-1 gives a reason to expect it to help.
- RAFT-style retrieval inside C3-hop.
- A cross-site analogue pool: the 27 upstream points' own histories, to find more examples as hot as Delhi's
  heatwaves.
- Retrieval keyed on the 500 hPa pattern (1980-2018 data exist; Rz).
- Retrieval for forecast ranges: neighbours' past errors.
- A learned similarity.
