# Graph tuning round: selection on the inner 2-year block

Mean all-days RMSE (°C) on each fold's last 2 training years, 3 seeds, averaged over the 4 folds. No 2007-2018 validation number was read. Selected = lowest (ties: earlier in the grid).

## A1prime_hw5 C3: selected `A1prime_hw5_C3_h64_lr1e-3_d0`

| Combination | Inner RMSE |
|---|---|
| A1prime_hw5_C3_h32_lr1e-3_d0 | 2.1828 |
| A1prime_hw5_C3_h32_lr1e-3_d0.2 | 2.1661 |
| A1prime_hw5_C3_h32_lr3e-4_d0 | 2.1699 |
| A1prime_hw5_C3_h32_lr3e-4_d0.2 | 2.1874 |
| A1prime_hw5_C3_h64_lr1e-3_d0 **(selected)** | 2.1606 |
| A1prime_hw5_C3_h64_lr1e-3_d0.2 | 2.1770 |
| A1prime_hw5_C3_h64_lr3e-4_d0 | 2.1729 |
| A1prime_hw5_C3_h64_lr3e-4_d0.2 | 2.1809 |

## A1prime_hw5 C3pool: selected `A1prime_hw5_C3pool_h64_lr1e-3_d0.2`

| Combination | Inner RMSE |
|---|---|
| A1prime_hw5_C3pool_h32_lr1e-3_d0 | 2.0575 |
| A1prime_hw5_C3pool_h32_lr1e-3_d0.2 | 2.0643 |
| A1prime_hw5_C3pool_h32_lr3e-4_d0 | 2.0593 |
| A1prime_hw5_C3pool_h32_lr3e-4_d0.2 | 2.0934 |
| A1prime_hw5_C3pool_h64_lr1e-3_d0 | 2.0676 |
| A1prime_hw5_C3pool_h64_lr1e-3_d0.2 **(selected)** | 2.0547 |
| A1prime_hw5_C3pool_h64_lr3e-4_d0 | 2.0591 |
| A1prime_hw5_C3pool_h64_lr3e-4_d0.2 | 2.0679 |

## A1prime_hw5 U1: selected `A1prime_hw5_U1_h128_lr1e-3_d0`

| Combination | Inner RMSE |
|---|---|
| A1prime_hw5_U1_h64_lr1e-3_d0 | 2.1699 |
| A1prime_hw5_U1_h64_lr1e-3_d0.2 | 2.2130 |
| A1prime_hw5_U1_h64_lr3e-4_d0 | 2.2509 |
| A1prime_hw5_U1_h64_lr3e-4_d0.2 | 2.2547 |
| A1prime_hw5_U1_h128_lr1e-3_d0 **(selected)** | 2.1447 |
| A1prime_hw5_U1_h128_lr1e-3_d0.2 | 2.1660 |
| A1prime_hw5_U1_h128_lr3e-4_d0 | 2.2001 |
| A1prime_hw5_U1_h128_lr3e-4_d0.2 | 2.2440 |

## A2Lr_hw5 C3: selected `A2Lr_hw5_C3_h32_lr1e-3_d0`

| Combination | Inner RMSE |
|---|---|
| A2Lr_hw5_C3_h32_lr1e-3_d0 **(selected)** | 2.3158 |
| A2Lr_hw5_C3_h32_lr1e-3_d0.2 | 2.3396 |
| A2Lr_hw5_C3_h32_lr3e-4_d0 | 2.3288 |
| A2Lr_hw5_C3_h32_lr3e-4_d0.2 | 2.3493 |
| A2Lr_hw5_C3_h64_lr1e-3_d0 | 2.3221 |
| A2Lr_hw5_C3_h64_lr1e-3_d0.2 | 2.3420 |
| A2Lr_hw5_C3_h64_lr3e-4_d0 | 2.3234 |
| A2Lr_hw5_C3_h64_lr3e-4_d0.2 | 2.3471 |

## A2Lr_hw5 C3pool: selected `A2Lr_hw5_C3pool_h64_lr3e-4_d0`

| Combination | Inner RMSE |
|---|---|
| A2Lr_hw5_C3pool_h32_lr1e-3_d0 | 2.2196 |
| A2Lr_hw5_C3pool_h32_lr1e-3_d0.2 | 2.2187 |
| A2Lr_hw5_C3pool_h32_lr3e-4_d0 | 2.2223 |
| A2Lr_hw5_C3pool_h32_lr3e-4_d0.2 | 2.2337 |
| A2Lr_hw5_C3pool_h64_lr1e-3_d0 | 2.2236 |
| A2Lr_hw5_C3pool_h64_lr1e-3_d0.2 | 2.2295 |
| A2Lr_hw5_C3pool_h64_lr3e-4_d0 **(selected)** | 2.2172 |
| A2Lr_hw5_C3pool_h64_lr3e-4_d0.2 | 2.2375 |

## A2Lr_hw5 U1: selected `A2Lr_hw5_U1_h128_lr1e-3_d0.2`

| Combination | Inner RMSE |
|---|---|
| A2Lr_hw5_U1_h64_lr1e-3_d0 | 2.2311 |
| A2Lr_hw5_U1_h64_lr1e-3_d0.2 | 2.2313 |
| A2Lr_hw5_U1_h64_lr3e-4_d0 | 2.2822 |
| A2Lr_hw5_U1_h64_lr3e-4_d0.2 | 2.2654 |
| A2Lr_hw5_U1_h128_lr1e-3_d0 | 2.1973 |
| A2Lr_hw5_U1_h128_lr1e-3_d0.2 **(selected)** | 2.1957 |
| A2Lr_hw5_U1_h128_lr3e-4_d0 | 2.2351 |
| A2Lr_hw5_U1_h128_lr3e-4_d0.2 | 2.2265 |
