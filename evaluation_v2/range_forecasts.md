# Range forecasts (specified 2026-10-18, before they were computed)

Out of fold 2007-2018. Centre = recalibrated 10-seed ensemble mean; spread from the fold's inner-block residuals (Gaussian, or the empirical residuals). CRPS in °C (lower is better); CRPSS = skill vs a climatology forecast. Coverage: share of days whose truth falls inside the 80% / 50% interval (ideal 0.80 / 0.50).

## Tmax

| Model | Days | CRPS Gaussian | CRPS empirical | CRPS climatology | CRPSS (Gauss / emp) | 80% coverage (Gauss / emp) | 50% coverage (Gauss / emp) |
|---|---|---|---|---|---|---|---|
| C3-hop | all | 1.082 | 1.073 | 1.427 | +0.242 / +0.248 | 0.82 / 0.79 | 0.55 / 0.50 |
| C3-hop | extreme | 1.224 | 1.156 | 2.928 | +0.582 / +0.605 | 0.78 / 0.68 | 0.38 / 0.37 |
| control | all | 1.169 | 1.160 | 1.427 | +0.181 / +0.187 | 0.82 / 0.79 | 0.53 / 0.49 |
| control | extreme | 1.582 | 1.511 | 2.928 | +0.460 / +0.484 | 0.65 / 0.54 | 0.25 / 0.23 |

| Test (sqrt-CRPS, paired cluster-jackknife) | Δ (95% CI) | p |
|---|---|---|
| C3-hop vs control, crps_gauss (all) | -0.0409 [-0.050, -0.032] | <0.001 |
| C3-hop vs control, crps_gauss (extreme) | -0.1513 [-0.192, -0.111] | <0.001 |
| C3-hop vs control, crps_emp (all) | -0.0409 [-0.050, -0.032] | <0.001 |
| C3-hop vs control, crps_emp (extreme) | -0.1541 [-0.199, -0.109] | <0.001 |

## WBGT (physical)

| Model | Days | CRPS Gaussian | CRPS empirical | CRPS climatology | CRPSS (Gauss / emp) | 80% coverage (Gauss / emp) | 50% coverage (Gauss / emp) |
|---|---|---|---|---|---|---|---|
| C3-hop | all | 1.191 | 1.189 | 1.379 | +0.136 / +0.137 | 0.81 / 0.79 | 0.52 / 0.49 |
| C3-hop | extreme | 2.577 | 2.653 | 3.539 | +0.272 / +0.250 | 0.23 / 0.25 | 0.03 / 0.02 |
| control | all | 1.270 | 1.269 | 1.379 | +0.079 / +0.080 | 0.80 / 0.79 | 0.53 / 0.49 |
| control | extreme | 2.812 | 2.886 | 3.539 | +0.205 / +0.184 | 0.21 / 0.24 | 0.01 / 0.01 |

| Test (sqrt-CRPS, paired cluster-jackknife) | Δ (95% CI) | p |
|---|---|---|
| C3-hop vs control, crps_gauss (all) | -0.0354 [-0.043, -0.028] | <0.001 |
| C3-hop vs control, crps_gauss (extreme) | -0.0716 [-0.116, -0.027] | 0.004 |
| C3-hop vs control, crps_emp (all) | -0.0359 [-0.043, -0.029] | <0.001 |
| C3-hop vs control, crps_emp (extreme) | -0.0701 [-0.113, -0.027] | 0.004 |

