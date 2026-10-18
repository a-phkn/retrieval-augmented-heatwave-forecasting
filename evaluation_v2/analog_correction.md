# Option 2: analog error correction of C3-hop (rule written 2026-10-18, before it was run)

Out of fold 2007-2018, 10 seeds x 4 folds, paired cluster-jackknife; Δ = AnEn-corr minus comparison (negative = AnEn-corr better). Honest past errors only (each fold's early-stopping years + earlier folds' validation years). K = 20 analogs within ±45 days of the time of year.

## Decision

| Family | (1) vs raw, all | (2) vs raw, extreme | (3) vs random, extreme | (4) vs linear, extreme | Result |
|---|---|---|---|---|---|
| Tmax | +0.005 [-0.040, +0.050] (p 0.827) | +0.729 [+0.584, +0.874] (p <0.001) | +0.328 [+0.137, +0.519] (p 0.003) | +0.388 [+0.243, +0.532] (p <0.001) | fail (True, False, False, False) |
| WBGT (physical) | -0.032 [-0.080, +0.015] (p 0.172) | +0.901 [+0.753, +1.050] (p <0.001) | +0.141 [+0.022, +0.259] (p 0.023) | +0.137 [+0.008, +0.266] (p 0.039) | fail (True, False, False, False) |

## Scores and forecaster's-dilemma checks

| Family | Variant | RMSE all | RMSE extreme | Bias all | Bias extreme | Forecast hot days per seed | RMSE / bias on forecast hot days |
|---|---|---|---|---|---|---|---|
| Tmax | raw | 2.065 | 1.827 | +0.15 | -1.25 | 681 | 2.26 / +0.89 |
| Tmax | anen | 2.070 | 2.555 | +0.02 | -2.09 | 306 | 1.64 / -0.01 |
| Tmax | random | 2.087 | 2.227 | +0.06 | -1.66 | 473 | 2.06 / +0.55 |
| Tmax | linear | 2.039 | 2.168 | +0.06 | -1.67 | 430 | 1.81 / +0.26 |
| WBGT (physical) | raw | 2.248 | 3.019 | +0.38 | -2.69 | 201 | 2.54 / +1.81 |
| WBGT (physical) | anen | 2.216 | 3.920 | +0.01 | -3.68 | 12 | 1.98 / +1.22 |
| WBGT (physical) | random | 2.229 | 3.779 | +0.02 | -3.49 | 50 | 2.31 / +1.61 |
| WBGT (physical) | linear | 2.174 | 3.783 | -0.00 | -3.55 | 20 | 1.82 / +1.15 |

Also (all days): Tmax: AnEn vs random -0.017 [-0.046, +0.012] (p 0.243), vs linear +0.032 [+0.001, +0.062] (p 0.042), linear vs raw -0.027 [-0.055, +0.002] (p 0.064); WBGT (physical): AnEn vs random -0.014 [-0.035, +0.007] (p 0.198), vs linear +0.041 [+0.021, +0.062] (p <0.001), linear vs raw -0.074 [-0.113, -0.035] (p <0.001)
Extreme days, controls vs raw: Tmax: linear +0.341 [+0.224, +0.458] (p <0.001), random +0.401 [+0.271, +0.531] (p <0.001); WBGT (physical): linear +0.764 [+0.665, +0.864] (p <0.001), random +0.761 [+0.631, +0.890] (p <0.001)
