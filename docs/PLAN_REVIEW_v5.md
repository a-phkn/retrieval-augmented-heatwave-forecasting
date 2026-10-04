# Review of Execution Plan v4 + Migration Guide + Updated Methodology — and a revised 8-week plan

## Context

You asked for a critical feasibility review of three documents (Updated Methodology, Execution Plan v4, Migration & File-Reuse Guide) against the actual repo, plus improvements and a full list of setbacks. Constraints you gave: **2 people (roles P1–P4 still assigned), 8-week deadline, output = paper + capstone**, DSTGNN preferred but open to an alternative ("physics-based" welcome) pending a team discussion.

Everything below was checked against the code and the committed parquet files, not taken from the PDFs. Numbers marked **[measured]** come from analyses I ran on `datasets/all_daily.parquet` and `evaluation/*.json` this session.

---

## 1. Verdict in one paragraph

Plan v4 is **scientifically careful but not feasible as written**. It was sized for 4 people × 12 weeks (~48 person-weeks); you have 2 × 8 (~16), so about one third. More importantly, three problems are invisible in the PDFs and would undermine the results even with unlimited time: (1) under the plan's own labels v2, **the validation split contains only 5 hot days / 1 episode** [measured], so every "tune on validation" step (flag threshold, gates, conformal, hot_weight) breaks; (2) **the current LSTM is worse than climatology** on global RMSE and on normal days, and the headline metric (extreme-stratum RMSE on *observed* extremes) rewards a warm bias; (3) the plan's spatial story (N9 → N42 ERA5-Land, UHI) rests on data that **has almost no independent spatial signal**. The good parts are worth keeping: the protocol (frozen index, one change per run, cluster bootstrap, test lock), the WBGT pivot, the SARAF reading, and the hybrid rule-engine + verifier design for Phase 6. There is also a real finding hiding in the data that makes a stronger paper than the plan expects: **humid heat (WBGT) is trending in Delhi but Tmax is not** [measured]. That gives the non-stationarity hypothesis a built-in negative control.

---

## 2. What I verified (evidence)

| Claim in plan/guide | Reality in repo | Status |
|---|---|---|
| Raw hourly JSON holds T and RH for 9 cells → Tier-A WBGT needs no new variables | `download_era5.py` requests hourly `temperature_2m, relative_humidity_2m, wind_speed_10m, surface_pressure` in `Asia/Kolkata` tz | ✅ true, **but `data/raw/` does not exist locally**, so it has to be re-downloaded |
| Checkpoints, FAISS index, analogue tables must be regenerated | No `models/*/seed_*`, no `retrieval/faiss_index.bin`/`candidates.parquet`/`analogues_top20.parquet` | ✅ true |
| 17,025 windows, 13,131/1,092/2,802 split | matches `forecast_windows.parquet`/docs | ✅ |
| Extreme RMSE: LSTM 1.285 vs RA 1.231 (val) | matches `evaluation/*/val_extended_metrics.json` | ✅ |
| "Weighted LSTM beats persistence" | true, **but loses to climatology**: global RMSE 2.401 vs **2.344**, normal-stratum 2.524 vs **1.940** | ⚠️ the plan never mentions this |
| `train_lstm.py` docstring says hw=10 | yes (constant is 20.0) | ✅ (G1 doc drift is real) |
| `context/progress.md` / `build_plan.md` stale | still say hw=10 and "Step 5 NOT STARTED" | ✅ |
| SARAF (Zhou et al., arXiv 2606.04135) | paper exists; the plan's summary of s̄, MMR, λ(s̄), σ(s̄), time bonus is accurate | ✅ |
| Env: Python 3.12, deps installed | local Python is **3.13.2**, **no venv**, `faiss` not installed; `requirements.txt` is unpinned (5 packages) | ⚠️ |

**Data findings [measured] that change the plan:**

- **Label scarcity.** Mar 15–Jul 31, Tmax ≥ 40 °C & anomaly ≥ 4.5 °C (the plan's IMD-style rule): train 204 days, **val 5 days / 1 episode**, test 46 / 11. Pooled over the four rolling folds (2007–18): only 59 days / 13 episodes.
  With **anomaly ≥ 3.0 °C**: pooled folds **154 days / 34 episodes**, test **131 / 27**. That is workable.
- **Trends (seasonal means, Sen slope per decade, 1980–2025):** Tmax **+0.09 (CI −0.14…+0.31), not significant**; Tmin +0.24 (sig.); RH mean **+1.9 %** (sig.); WBGT proxy from daily means **+0.45 °C (CI 0.33…0.56)**.
  So "old analogues are outdated" is plausible for WBGT and implausible for Tmax. That makes Tmax a natural negative control.
- ERA5 area-mean damping: max Tmax over 46 yrs = 46.6 °C; only 36 days ≥ 45 °C. Station-based IMD thresholds don't transfer directly.

---

## 3. Critical issues, ranked (each with its fix)

**B = blocks or invalidates results; M = weakens claims.**

1. **[B] Validation is empty of extremes under labels v2.** Flag-threshold tuning, G-D2/G-D3 gates, conformal per stratum, and dry/humid bins are all specified "on validation".
   *Fix:* make **rolling-origin out-of-fold (OOF) predictions over 2007–2018 the development evidence** for every selection and gate (12 years, not 3). Choose the label rule by a **pre-registered minimum-sample rule**, not by matching known events: e.g. "≥ 25 episodes in pooled folds".
   That gives: primary hot day = Tmax ≥ 40 & anomaly ≥ 3.0 (documented reason: ERA5 area-mean damps station peaks); IMD 4.5/6.5 kept as the "severe" sub-stratum; add IMD's absolute ≥ 45 °C criterion, which the plan omits. WBGT label = seasonal percentile, picked by the same rule once hourly WBGT exists.

2. **[B] The headline metric is gameable, and the baseline fails an absolute floor.** Stratifying on *observed* extremes is the **forecaster's dilemma** (Lerch, Thorarinsdottir, Ravazzolo & Gneiting 2017, *Statistical Science* 32(1)): a model biased warm wins it. That is exactly what hot_weight=20 does (+0.88 °C bias on normal days; worse than climatology).
   *Fix:*
   - (a) **Absolute credibility floor:** MSE skill score vs climatology > 0 globally. The parent-relative "+5 %" rule is not enough.
   - (b) Also report the **forecast-conditioned** stratum (days the model *predicts* extreme) and Brier score / reliability for the flag. Optionally add a threshold-weighted score.
   - (c) New rung **A2r: predict the anomaly relative to train climatology (residual target)**. This is a cheap architecture change that usually recovers climatology-level skill on normal days, so hot_weight no longer has to buy extreme skill with a global warm bias.

3. **[B] A2 changes two factors at once** (target Tmax→WBGT *and* labels v1→v2, and the labels also drive the loss weight mask). That breaks the plan's own one-change rule.
   *Fix:* insert **A1′ = Tmax, labels v2** between A1 and A2.

4. **[B] Rolling folds have a hidden engineering cost.** Climatology, anomaly channels, normalisation stats, hot labels, retrieval features and the FAISS index are all fixed to 1980–2015 in the current parquet and code. Using them for fold val=2007–09 leaks the validation block into the features.
   *Fix:* a `fold` abstraction that recomputes all train-only quantities per fold (`training/data.py`, `retrieval/features.py`, `build_index.py`, `precompute_analogues.py`). Budget ~3–4 days. It is the backbone of everything else.

5. **[B] Missing controls for retrieval.** R0 vs A2 mixes "retrieval" with "extra parameters and an attention module". SARAF itself uses a random-retrieval control.
   *Fix:* add **R0-rand** (same model, K eligible analogues sampled at random) and a non-neural **analogue-ensemble baseline** (Delle Monache et al. 2013, *MWR* 141: mean of analogue futures). Also add SARAF's **future-similarity retention diagnostic** (Spearman of input-similarity vs future-similarity) on the current analogues in Week 1. If analogue futures don't track the truth, no fusion design can help, and you learn that in a day.

6. **[M→B for the spatial claims] N9/N42 carry little independent spatial information.** The 9 ERA5 cells span ~75 km of flat plain. ERA5-Land's 2 m temperature is ERA5 forcing run through a land-surface model with no urban physics (Muñoz-Sabater et al. 2021, *ESSD*). So N42 "propagation" would be largely interpolation, and UHI is not represented. Phase 4A/4B and the UHI claim are the weakest part of the plan. See §4 for the fix (a regional, upstream graph).

7. **[M] Factual corrections to Plan v4 before anything is written into the paper:**
   - "35 °C WBGT is the human-tolerance limit": the 35 °C limit is **wet-bulb temperature**, not WBGT (Sherwood & Huber 2010, *PNAS*). Vecellio et al. 2022 (*J. Appl. Physiol.*) suggest ~31 °C wet-bulb. For WBGT, cite ISO 7243 categories. The conclusion "45 °C WBGT is implausible" still stands.
   - The plan calls the BoM formula "shade-based". BoM describes it as an approximation that ignores actual radiation and wind. **Verify the wording** before tiering on it; Lemke & Kjellstrom 2012 (*Industrial Health* 50) compare the methods.
   - "Liljegren WBGT with thermofeel": verify which WBGT method thermofeel implements. A Liljegren implementation is in Kong & Huber 2022 (*Earth's Future*).
   - The "AC set-point 24 °C" example traces to the BEE/Ministry of Power **default-setting rule for room ACs**, not a Delhi demand-response protocol. Quote it accurately or drop it.

8. **[M] ERA5T revision on re-download.** The last ~3 months before 2026-09-06 were preliminary ERA5T when first pulled and may have changed since.
   *Fix:* keep the committed `all_daily.parquet` as v1 truth. Diff the re-downloaded overlap, document any change, and set `END_DATE` fixed (2026-09-06).

9. **[M] Statistical power.** With 11–27 test episodes and a seed-SD of 0.12–0.16 °C, effects of ~0.05 °C (the current RA gain) are undetectable, and Holm correction across H1–H5 makes it worse.
   *Fix:* in Week 1, run a **minimum-detectable-effect simulation** with `stats.py` on the existing predictions. Pre-register 2 primary hypotheses (not 5) and keep the rest exploratory.

10. **[M] Conformal under drift.** The WBGT trend breaks the exchangeability assumption.
    *Fix:* calibrate on the most recent OOF years, or use adaptive conformal inference (Gibbs & Candès 2021, NeurIPS).

11. **[L] Compute is not the bottleneck; people are.** The LSTM takes ~20 s per seed. An N9 DSTGNN is roughly 9× that, so a few minutes per seed on CPU. Use local CPU first and Colab only for many-seed sweeps and the LLM. Every upload/download cycle costs human time.

---

## 4. Architecture: honest review and options for the team discussion

**Honest review of DSTGNN-as-primary (plan v4 version, local N9→N42).** The engineering spec and the gates (G-D0…G-D4) are good. The problem is the data: 9 near-identical reanalysis cells over one city, and an aggregate target that is the mean of those same cells. Expected outcome: C4 ≈ C2 (shared per-node LSTM), so the graph adds parameters without signal. The plan already admits this ("honest expectation"). That means ~3 weeks of one person's time with a likely null result, and a null is hard to sell as "the best solution".

**Physics point that changes the picture:** Delhi heatwaves are synoptic. Hot, dry air is **advected from the north-west (Thar/Rajasthan/Pakistan)** under anticyclones over 1–3 days, and soil moisture amplifies them. The signal a graph *can* exploit lives at ~100–1,000 km scale, not ~25 km.

| Option | What it is | Expected skill gain | Risk | Effort (2 ppl, 8 wks) | Paper story |
|---|---|---|---|---|---|
| **1. Regional "upstream" DSTGNN (recommended if DSTGNN must stay)** | Same DSTGNN code (§9.2 of v4). Nodes = ~20–30 ERA5 points on a ~1.5–2° lattice over NW India/Pakistan, plus the Delhi node. Daily features only for upstream nodes; the advective edges (wind-direction gated) now carry real meaning. Target unchanged (Delhi aggregate). | Moderate. Upstream heat and dryness lead Delhi by 1–3 days, which matches the 5-day horizon. | Data download volume (Open-Meteo quota) and a new G-D0 check. | Medium. The downloader pattern is reused; daily-only variables for upstream nodes keep it light. | "Graph learns where Delhi's heat comes from." Testable, physically grounded, and the adjacency is interpretable (advective weights align with wind). |
| **2. Physics-guided LSTM (recommended if you can switch)** | (a) **Physics-structured output head:** predict Tmax and vapour pressure/dewpoint at time of max, then compute WBGT/HI *inside the model* with the BoM/Rothfusz formulas (differentiable). (b) **Physics-informed predictors:** soil moisture (Open-Meteo ERA5 has soil-moisture variables; verify) and, if obtainable, z500/T850 (ARCO-ERA5 Zarr from Colab, or CDS). | Highest per unit of effort. Guarantees Tmax/WBGT consistency and gives the dry vs humid error decomposition directly (your Phase 5 question). | Low | Low–medium | "Physically consistent heat-stress forecasting." Pairs naturally with the WBGT-trend finding. |
| 3. Plan v4 local N9 DSTGNN | As specified, gated | Likely ≈ C2 | Low technical, high scientific | Medium | Weak; defensible only as a null result plus node-level skill |
| "Fancy" add-on: foundation-model benchmark | Compare against ECMWF HRES / GraphCast / Pangu forecasts from **WeatherBench 2** (Rasp et al. 2024) at the Delhi grid points, for the years they cover | n/a (an external yardstick: they will beat any local-history model at days 1–5) | Low | Low (a few days) | Honest positioning: retrieval and heat-stress tiering add value *on top of* NWP-class skill, rather than competing with it |

**My recommendation:** **Option 1 + Option 2a together.** Keep the DSTGNN name and code, but on the regional upstream graph, with the physics-structured WBGT head as one rung. You get a DSTGNN with a real chance of beating the LSTM, plus a physics-based component, without doubling scope. Add the WeatherBench 2 benchmark only if Week 7 has slack.

**Controls stay as in v4, with one addition:** an **LSTM given the same upstream features flattened**. Without it, any DSTGNN gain could be "more data" rather than "graph". Keep v4's 20-config budget and the BB* decision date, moved to end of Week 5. The fallback is the physics-guided LSTM (Option 2), not iTransformer: it is cheaper and better suited to L=14 and 13k windows. **Drop PatchTST, iTransformer, N42, and 4B UHI.**

---

### 4.1 What happens to the existing trained LSTM baseline?

- **It stays exactly as it is, as A1** (Tmax, labels v1, hot_weight=20). The code and recipe don't change.
- **Its checkpoints aren't in the repo** (they're gitignored), so Week 1 regenerates them with the existing `python -m training.train_lstm` (~2 min on CPU). This reproduces the same model; it isn't a new one. The regenerated checkpoints are then hashed and frozen as the legacy reference.
- **A1 can't be the control for the new models.** The physics-guided LSTM and the DSTGNN predict WBGT with labels v2, evaluated on rolling folds. A fair comparison needs an LSTM trained on the same target, labels and folds. So the same architecture and recipe is trained a few more times:
  - **A1′** (Tmax, labels v2)
  - **A2** (WBGT, labels v2)
  - **A2r** (anomaly target, which fixes the "worse than climatology" problem)
- Each of these takes minutes on CPU, once per fold.
- The physics-guided LSTM (2a) and the DSTGNN (Option 1) are **additional** models compared against A2 (or A2r). They don't replace A1.

### 4.2 Where retrieval (RAG) sits — it is still the core of the project

- **Retrieval stays central.** Primary hypothesis H-A is "retrieval improves WBGT extreme forecasts", so the project's name still holds.
- **Two separate retrieval systems:**
  1. **Forecast retrieval (analogue RAG):** the FAISS analogue search plus the attention fusion (`models/retrieval_lstm.py`). It's upgraded through the ladder R0 → R4 (time-aligned, MMR diversity, drift gate, climate-shift adjustment).
  2. **Document retrieval for Phase 6:** BM25 + dense search over the approved IMD/NDMA/WHO corpus, feeding the constrained LLM advisory.
- **How analogue retrieval plugs into each model:**
  - The ladder is developed on the LSTM control (A2/A2r) because it is cheap and fast to iterate on.
  - The **best rung is then attached to the final backbone**: the DSTGNN (run G-R*) and/or the physics-guided LSTM.
  - This gives a 2×2 comparison: {LSTM, final backbone} × {no retrieval, best retrieval}. It tests whether retrieval helps regardless of architecture.
- **Nothing in Options 1/2 replaces retrieval.** They only change the backbone that retrieval feeds into.

## 5. Revised scope for 2 people / 8 weeks

**Role mapping:** **Person A = P1 + P3** (data, labels, retrieval, diagnostics). **Person B = P2 + P4** (models, DSTGNN, advisory/LLM, demo). Data feeds retrieval; models feed the advisory, so each person owns one dependency chain.

**Keep (scientific spine):** Phase 0 stats and freeze · Tier-A WBGT + HI from hourly · labels v2 (sample-size rule) + acceptance test · rolling-fold infrastructure · A0/A1/A1′/A2/A2r · derived binary flag (F1, FAR, POFD at an OOF-fixed threshold) · compact retrieval ladder · DSTGNN (Option 1) + controls · strata, dry/humid and analogue-age analyses on OOF + test · test lock · Phase 6 rule engine + verifier + constrained LLM.

**Compact retrieval ladder (on the LSTM):** R0 (RA-v1), **R0-rand**, **AnEn baseline**, R1 time-aligned bonus, R2 MMR, R3 one pre-registered drift gate (ADF p>0.05 OR KPSS p<0.05; report firing rate; the power check is a 1-day task), R4 climate-shift adjustment. Run each on both WBGT and Tmax (negative control). Transplant the best rung onto the DSTGNN (G-R*).

**Cut (with reason):** N42 / ERA5-Land (no independent signal) · 4B UHI head (a separate project) · Tier-B Liljegren (keep only a 2-year spot-check if CDS arrives) · iTransformer/PatchTST · R3b/R5/R6 (unpowered) · look-back 56 · dynamic-NDVI side experiment · learned flag head H1 · GPT-4o comparator (optional) · React dashboard (replace with a single static results page or notebook demo) · remote sensing as model input. **NDVI/NDBI/LST via GEE survives only as an optional HVI layer for Phase 6 zone naming** (stretch).

---

## 6. Week-by-week plan (8 weeks; no slack, so the cut order in §8 is pre-agreed)

| Wk | Person A (P1+P3) | Person B (P2+P4) | Gate at end of week |
|---|---|---|---|
| 1 | Create venv (py 3.12 if available, else 3.13); pin `requirements.txt`. Back up `evaluation/`, `predictions/` → `archive_v1/`. Edit downloader copy: fixed `END_DATE=2026-09-06`, add `wind_direction_10m`, `dew_point_2m`, radiation, soil moisture → `data/raw/era5_v2/`; start re-download (quota: ~2–3 days). Rebuild FAISS + analogues; **premise checks**: trend report (WBGT vs Tmax), future-similarity retention, AnEn baseline. | `evaluation/stats.py` (cluster bootstrap by episode-year, DM-HAC lag 4) + calibration tests; extend `evaluate_lstm.py` to save per-window predictions; reproduce A1 locally (record timings vs `archive_v1`); manifest + `splits/window_index_v1.parquet` + `git tag v1-frozen`; **MDE simulation**; fix doc drift in `context/`. Start a 1-page corpus source list. | G0: tests pass, A1 reproduces, CIs for LSTM vs RA, MDE known, premise report written |
| 2 | `pipeline/hourly_features.py` (WBGT-BoM, HI per cell-hour → daily max → 9-cell mean); `pipeline/labels_v2.py` with sample-size rule + `tests/test_labels_v2.py` (flags Jun-2019, Apr–May 2022, May–Jun 2024 per IMD bulletins; nothing in Jan–Feb). | **Fold abstraction** (per-fold climatology / normalisation / labels / index) + `training/train_unified.py` + `models/backbones.py` (encode + head); `registry/runs.csv`. | G1: labels pass acceptance; fold OOF pipeline runs A1 end-to-end |
| 3 | Regional upstream node download (Option 1) or skip; G-D0 spatial/lead-lag signal check. Retrieval R0, R0-rand, R1 on folds (WBGT + Tmax). | A1′, A2, A2r on folds; hot_weight re-sweep (OOF); derived flag. DSTGNN skeleton + **G-D1 unit tests**. | G2: control chosen (A2 or A2r); G-D1 pass |
| 4 | R2, R3 (power check first), R4; mechanism metrics (redundancy, age, season mismatch). | C2, C3, C4 + flattened-upstream LSTM control; G-D2 trainability, G-D4 compute. Physics head rung (2a). | G3: best retrieval rung picked on OOF |
| 5 | Analogue-age analysis (regression with controls + DB-scope intervention) on OOF; dry/humid split (calendar + RH-based). | G-D3 non-inferiority on OOF; **BB* decision (hard date)**; G-R* transplant. Advisory: action library schema, rule engine, verifier + tests (no LLM). | G4: BB* recorded in `docs/PREREGISTRATION.md` |
| 6 | Diagnostics tables/figures from saved OOF predictions; conformal (recent-years or adaptive); write `docs/PREREGISTRATION.md` (2 primary hypotheses, configs + hashes). | Constrained LLM (open-weights 7–8B, 4-bit, llama.cpp/vLLM + grammar/JSON schema on Colab) with mocked forecasts; adversarial test set. | G5: **test lock** (tagged commit) |
| 7 | **One test run, 10 seeds** (Colab if needed); evaluate locally; case studies (2019, 2022, 2024). | Advisory on real forecasts + conformal bands; faithfulness/coverage eval (2 raters, 100 advisories → reduce to 50 if short). | G6: results frozen |
| 8 | Paper: methods, results, limitations. | Demo page, reproducibility pack (README with expected numbers, Colab notebooks, manifest). | Deliverables |

---

## 7. Setbacks to expect (risk register)

| # | Setback | Likelihood / impact | Early signal | Mitigation |
|---|---|---|---|---|
| 1 | Open-Meteo quota/rate limits slow the re-download (≈5k monthly requests × weight >1; extra variables push past 10 vars) | High / medium | 429s on day 1 | Start Day 1; resumable script; split variables into a second pass; regional nodes daily-only; ARCO-ERA5 via Colab as backup |
| 2 | Re-downloaded data ≠ committed v1 (ERA5T revisions, API changes) | Medium / medium | Overlap diff ≠ 0 | v1 parquet stays truth; document the diff; v2 built only from new raw |
| 3 | A1 doesn't reproduce (CPU vs GPU, torch versions, Python 3.13) | Medium / low | Metrics off by > noise | Log versions; accept a documented tolerance; regenerated checkpoints become the hashed baseline |
| 4 | Labels v2 miss a known event, or the sample-size rule conflicts with the acceptance test | Medium / high | `test_labels_v2` fails | Adjust only the percentile/anomaly rule globally (documented); never per event |
| 5 | Retrieval still ≈ no-retrieval | High / medium | R0 ≈ R0-rand in Week 3 | Frame as "retrieval quality and recency control" (v4 D8); mechanism metrics + WBGT/Tmax contrast make a null publishable |
| 6 | Drift gate is inert (ADF always rejects) | High / low | Firing rate ≈ 0 in the power check | Report as a finding; R4 (climate-shift adjustment) carries the drift claim |
| 7 | DSTGNN ≈ C2 or the flattened-upstream control | Medium / high | G-D0 shows weak lead-lag | Pre-registered fallback to the physics-guided LSTM by end of Week 5; no extra tuning |
| 8 | Fold infrastructure takes longer than 4 days | Medium / high | Not done by end of Week 2 | Fallback: 2 folds (2010–12, 2013–15) + val, still pooled OOF |
| 9 | Colab disconnects / GPU unavailable | Medium / low | — | Most jobs are CPU-feasible; Drive checkpoints for long runs; seed-resume pattern already exists |
| 10 | Phase 6 sources missing (Delhi demand-response) | High / medium | Week 1 source list empty for energy | Health-only advisory; action allowed only with a verbatim quote |
| 11 | LLM verifier rejects too much or the model can't follow the schema | Medium / medium | High abstain rate on mocks | Grammar-constrained decoding; smaller action enum; rule-template fallback is itself an ablation arm |
| 12 | Test-set peeking under deadline pressure | Medium / catastrophic | Anyone runs `split=='test'` before G5 | Test-lock tag + a code guard that refuses test eval without the lock file |
| 13 | ERA5 humidity trend is partly a reanalysis artefact | Medium / medium for the paper | — | Caveat it; cross-check against IMD/CPCB stations if obtainable; Tmax control limits the damage |
| 14 | Overwriting v1 outputs (`train_lstm`, `prepare_datasets.py` write into v1 paths and regenerate a test file) | Medium / high | — | Back up first; v2 scripts write only to `*_v2/` paths (as the Guide says) |
| 15 | Bus factor: 2 people, 8 weeks, illness or exams | Medium / high | — | The cut order below is agreed in advance; weekly gates are visible in `context/progress.md` |

---

## 8. Cut order if behind schedule (agree with teammate now)

1. WeatherBench 2 benchmark → 2. GEE/HVI layer → 3. LLM model comparison (keep one model) → 4. Physics head rung → 5. R4 → 6. regional graph (fall back to N9, reported as a null)

**Never cut:** fold OOF evaluation, stats toolkit, labels acceptance test, R0-rand control, test lock, Phase 6 verifier.

---

## 9. Decisions (recorded 2026-10-04)

1. **Architecture — DECIDED:** Option 1 (regional upstream DSTGNN) + 2a (physics-structured WBGT head). The fallback is the physics-guided LSTM (Option 2), decided by the end-of-Week-5 BB* gate.
2. **Label rule — DECIDED:**
   - Season Mar 15–Jul 31.
   - Primary hot day: Tmax ≥ 40 °C and anomaly ≥ 3.0 °C vs per-fold train climatology, or Tmax ≥ 45 °C (IMD's absolute criterion).
   - Episode: ≥ 2 consecutive hot days.
   - Severe sub-stratum: anomaly ≥ 4.5 °C (IMD); "very severe" ≥ 6.5 °C, reported descriptively only.
   - WBGT label: seasonal percentile from train years, chosen by the pre-registered rule "≥ 25 episodes in the pooled folds".
   - Labels v1 are kept for continuity.
3. **Headline metric — DECIDED:** extreme-stratum RMSE (primary), gated by two requirements:
   - an absolute floor: MSE skill score vs climatology > 0 globally;
   - the forecast-conditioned extreme stratum, reported alongside.
4. **Primary hypotheses — DECIDED.** Holm correction applies across these two only; everything else is exploratory.
   - H-A: the best retrieval rung beats no retrieval on WBGT extreme-stratum RMSE (OOF + test).
   - H-B: BB* is non-inferior to the LSTM control (upper CI bound of ΔRMSE ≤ +0.15 °C extreme, global MAE ≤ +5 %) and better on extremes.
5. **Phase 6 energy-grid scope — DECIDED: health-only fallback.**
   - The advisory covers health actions only, unless citable energy-grid sources are verified by end of Week 2.
   - If sources are found, energy actions enter the action library with verbatim quotes, like any other action.
   - The source search uses `docs/PHASE6_SOURCE_RESEARCH_BRIEF.md`.

---

## 10. What happens after you approve this plan

1. Save this review into the repo as `docs/PLAN_REVIEW_v5.md` (the only file change at first) so your teammate can read it.
2. Week 1 work starts only after your OK on each multi-file step (per your CLAUDE.md): venv + pinned requirements → backup + freeze/manifest → `evaluation/stats.py` with tests → per-window predictions → premise-check notebook/script.

**Verification for each step:**
- `python -m pytest tests/ -v` (expect the existing 9 to pass, plus new stats-calibration and manifest tests).
- A1 metrics reproduce `archive_v1/evaluation/baseline_lstm/val_extended_metrics.json` within a documented tolerance.
- `stats.py`: false-positive rate ≈ 5 % on a synthetic null; detects a planted effect.
- The labels-v2 acceptance test passes.
- Every run has a `registry/runs.csv` row with data and config hashes.
- No test-split evaluation before the G5 lock tag.

**Critical files:**
- **Reuse:** `training/data.py` (`build_split_target_stratum`, normalisation), `retrieval/query.py` (`eligibility_mask`, `dedup_max_per_episode`), `retrieval/precompute_analogues.py`, `models/lstm.py`, `models/retrieval_lstm.py` (`encode()` already separated), `training/train_retrieval_lstm.py` (seed-resume), `Step5_Colab_Minimal.ipynb` (bundle pattern).
- **New:** `evaluation/stats.py`, `pipeline/hourly_features.py`, `pipeline/labels_v2.py`, `training/train_unified.py`, `models/backbones.py`, `models/dstgnn.py`, `data/graph.py`, `retrieval/saraf.py`, `advisory/*`.
