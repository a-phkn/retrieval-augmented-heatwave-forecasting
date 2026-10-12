# Physics head and R2 (rules written 2026-10-07, before either ran)

Out of fold 2007-2018, 10 seeds x 4 folds, paired cluster-jackknife; Δ = rung minus comparison (negative = the rung is better).

## Physics head (PH_lstm): **DROPPED** under the keep rule
(dropped only if the all-days CI upper end is >= +0.05 °C for WBGT or Tmax)

Training: 0 skipped (non-finite) batches; all losses finite: True.

| Target | RMSE head / direct | Δ all days | Extreme RMSE head / direct | Δ extreme | Extreme bias head / direct | Fails rule |
|---|---|---|---|---|---|---|
| WBGT (physical) vs `A2Lr_hw5` | 2.415 / 2.373 | +0.041 [+0.001, +0.082] (p 0.046) | 3.364 / 3.274 | +0.090 [-0.062, +0.241] (p 0.223) | -3.06 / -3.02 | True |
| Tmax vs `A1prime_hw5` | 2.308 / 2.175 | +0.132 [+0.077, +0.187] (p <0.001) | 2.180 / 2.153 | +0.027 [-0.125, +0.179] (p 0.700) | -1.73 / -1.71 | True |

## R2 (regional matching + MMR diversity) under G3

| Family | All RMSE | Extreme RMSE | Δ vs control, all | Δ vs control, extreme | Δ vs R0-rand, extreme | Δ vs Rg, all | G3 (1, 2, 3) |
|---|---|---|---|---|---|---|---|
| Tmax | 2.166 | 2.162 | -0.010 [-0.036, +0.016] (p 0.441) | +0.008 [-0.106, +0.122] (p 0.875) | +0.006 [-0.139, +0.151] (p 0.926) | -0.007 [-0.029, +0.015] (p 0.534) | FAIL (True, False, False) |
| WBGT (physical) | 2.326 | 3.138 | -0.048 [-0.074, -0.021] (p <0.001) | -0.136 [-0.300, +0.029] (p 0.098) | -0.092 [-0.259, +0.076] (p 0.258) | -0.004 [-0.024, +0.016] (p 0.688) | FAIL (True, False, False) |

R2 was pre-registered and screened after G3 (decisions.md 2026-10-07); like Rg, it is reported as a post-G3 rung. Extreme-day results are dominated by bias (see `calibration_check.md`).
