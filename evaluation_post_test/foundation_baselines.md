# Pretrained time-series foundation models vs our models, test years 2019-2026 (post-test exploration)

Specified 2026-10-20 before running (context/decisions.md). Zero-shot: each foundation model sees only the target's own last 1024 days and forecasts 5 days (median). RMSE in °C (lower is better): ours = mean of 10 seeds' RMSE; foundation models = one deterministic run. Δ = first-named minus the foundation model (negative = ours better); paired cluster-jackknife, year x season. Caveat: their pretraining data may include weather series from these years.

## WBGT (physical)

| Model | RMSE all days | RMSE heatwave days | Bias heatwave days |
|---|---|---|---|
| C3-hop | 2.258 | 3.023 | -2.66 |
| Delhi-only LSTM | 2.386 | 3.103 | -2.75 |
| climatology | 2.568 | 4.749 |  |
| persistence | 3.043 | 3.775 |  |
| Chronos-2 (Amazon) | 2.387 | 3.822 | -3.54 |
| TimesFM 3.0 (Google) | 2.396 | 3.828 | -3.54 |

| Comparison | Δ RMSE (95% CI) | p |
|---|---|---|
| C3-hop vs Chronos-2 (Amazon) (all) | -0.129 [-0.207, -0.051] | 0.002 |
| C3-hop vs Chronos-2 (Amazon) (extreme) | -0.799 [-1.099, -0.499] | <0.001 |
| LSTM vs Chronos-2 (Amazon) (all) | -0.002 [-0.078, +0.075] | 0.964 |
| LSTM vs Chronos-2 (Amazon) (extreme) | -0.719 [-1.016, -0.421] | <0.001 |
| C3-hop vs TimesFM 3.0 (Google) (all) | -0.137 [-0.216, -0.059] | 0.001 |
| C3-hop vs TimesFM 3.0 (Google) (extreme) | -0.806 [-1.169, -0.442] | <0.001 |
| LSTM vs TimesFM 3.0 (Google) (all) | -0.010 [-0.088, +0.068] | 0.797 |
| LSTM vs TimesFM 3.0 (Google) (extreme) | -0.725 [-1.096, -0.354] | 0.002 |

## Tmax

| Model | RMSE all days | RMSE heatwave days | Bias heatwave days |
|---|---|---|---|
| C3-hop | 2.234 | 2.307 | -1.92 |
| Delhi-only LSTM | 2.361 | 2.548 | -2.02 |
| climatology | 2.868 | 4.447 |  |
| persistence | 2.772 | 2.555 |  |
| Chronos-2 (Amazon) | 2.409 | 3.101 | -2.70 |
| TimesFM 3.0 (Google) | 2.411 | 3.053 | -2.53 |

| Comparison | Δ RMSE (95% CI) | p |
|---|---|---|
| C3-hop vs Chronos-2 (Amazon) (all) | -0.175 [-0.260, -0.091] | <0.001 |
| C3-hop vs Chronos-2 (Amazon) (extreme) | -0.795 [-1.129, -0.460] | <0.001 |
| LSTM vs Chronos-2 (Amazon) (all) | -0.048 [-0.099, +0.003] | 0.062 |
| LSTM vs Chronos-2 (Amazon) (extreme) | -0.553 [-1.068, -0.038] | 0.039 |
| C3-hop vs TimesFM 3.0 (Google) (all) | -0.177 [-0.254, -0.100] | <0.001 |
| C3-hop vs TimesFM 3.0 (Google) (extreme) | -0.747 [-1.116, -0.377] | 0.002 |
| LSTM vs TimesFM 3.0 (Google) (all) | -0.050 [-0.105, +0.005] | 0.075 |
| LSTM vs TimesFM 3.0 (Google) (extreme) | -0.505 [-1.014, +0.004] | 0.051 |

