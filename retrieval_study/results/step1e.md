# Retrieval study, step 1e: RAFT in its own kind of setting (hourly, positive control)

Pre-registered in `context/decisions_retrieval.md` (RS-1e) before it was run. Delhi's hourly 2 m temperature (9-cell mean); train 1980-2015, validate 2016-2018; queries every 6 hours; forecast = the next 48 hourly values. RMSE in °C over the 48 values. Δ = RMSE of the second minus the first (negative = retrieval helps). 95% CIs: paired cluster jackknife, 9 year x season clusters (fragile; descriptive).

## Window 12 hours

Queries: 15000 fitting, 2913 for choosing settings, 4377 validation; 105173 candidate windows.

| Base | RMSE none | RMSE RAFT | RAFT vs none | RAFT-rand vs none | RAFT vs RAFT-rand | RAFT's m |
|---|---|---|---|---|---|---|
| H0 (window only) | 1.9782 | 1.8329 | -0.1453 [-0.1939, -0.0967] **(CI < 0)** | -0.0392 [-0.0592, -0.0193] **(CI < 0)** | -0.1061 [-0.1373, -0.0748] **(CI < 0)** | 20 |
| H1 (window + phase) | 1.8962 | 1.8176 | -0.0786 [-0.1120, -0.0453] **(CI < 0)** | -0.0151 [-0.0253, -0.0049] **(CI < 0)** | -0.0635 [-0.0874, -0.0397] **(CI < 0)** | 20 |

## Window 96 hours

Queries: 15000 fitting, 2913 for choosing settings, 4377 validation; 105145 candidate windows.

| Base | RMSE none | RMSE RAFT | RAFT vs none | RAFT-rand vs none | RAFT vs RAFT-rand | RAFT's m |
|---|---|---|---|---|---|---|
| H0 (window only) | 1.6737 | 1.6447 | -0.0290 [-0.0459, -0.0121] **(CI < 0)** | -0.0005 [-0.0035, +0.0025] | -0.0285 [-0.0454, -0.0116] **(CI < 0)** | 20 |
| H1 (window + phase) | 1.6587 | 1.6395 | -0.0191 [-0.0366, -0.0017] **(CI < 0)** | +0.0015 [-0.0000, +0.0030] | -0.0206 [-0.0374, -0.0038] **(CI < 0)** | 20 |

## The predictions (automatic reading)

| Prediction | Holds |
|---|---|
| P8 | yes |
| P9 (gain at 12 h at least twice the gain at 96 h) | yes |
| P9 (smaller gain once the phase is given) | yes |
