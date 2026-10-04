"""
Analogue-ensemble (AnEn) baselines on the validation split (plan v5, Week 1 premise
checks). Model-free: no training, only averaging what retrieved analogues did next.

Why: RA-v1's attention is nearly uniform (it effectively averages its 5 analogues),
so the key control is whether a plain average of analogue outcomes does as well, and
whether the retriever's choice of analogues beats random past windows. The analogue
ensemble is a standard meteorological baseline (Delle Monache et al. 2013, Mon. Wea.
Rev. 141).

  anen        forecast(day) = clim_mean(day) + clim_std(day) * mean_k z_k(lead)
              z_k(lead) = standardised Tmax anomaly of analogue k on its own lead day,
              analogues = the SAME top-5 RA-v1 uses (retrieval/analogues_top20.parquet).
  anen_random same formula with 5 random eligible windows (train split) per query,
              20 independent draws stored as seeds 0-19.

Standardised anomalies (anomaly / clim_std) make analogues from slightly different
calendar dates comparable, as in the retrieval features. All climatology is train-only
(datasets/all_daily.parquet). Output: predictions_v1/val/{anen,anen_random}.parquet in
the predict_v1 format. Test split never read.

Run from repo root:  python -m evaluation.anen_v1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.predict_v1 import ANALOGUES_PATH, K, LEADS, OUT_DIR, REPO_ROOT, long_frame
from training.data import _load_daily, _load_windows, build_split_arrays, build_split_target_stratum

N_RANDOM_DRAWS = 20
RANDOM_SEED = 20261004


def _z_outcomes(daily: pd.DataFrame, start_dates: np.ndarray) -> np.ndarray:
    """(n, LEADS) standardised Tmax anomalies on start_date + 0..4 days."""
    z = (daily["t_max_anomaly"] / daily["clim_std_t_max"]).to_numpy()
    pos = daily.index.get_indexer(pd.DatetimeIndex(start_dates))
    if (pos < 0).any():
        raise ValueError("analogue dates not found in all_daily")
    return z[pos[:, None] + np.arange(LEADS)[None, :]]


def _forecast(daily: pd.DataFrame, query_dates: pd.Series, mean_z: np.ndarray) -> np.ndarray:
    """clim_mean + clim_std * mean_z on each query's 5 target days."""
    pos = daily.index.get_indexer(pd.DatetimeIndex(query_dates))[:, None] + np.arange(LEADS)[None, :]
    return daily["clim_mean_t_max"].to_numpy()[pos] + daily["clim_std_t_max"].to_numpy()[pos] * mean_z


def run(split: str = "val") -> dict[str, pd.DataFrame]:
    if split != "val":
        raise ValueError("AnEn premise check is defined on the val split only")
    daily = _load_daily()  # indexed by date, contiguous
    windows = _load_windows()
    _, y_raw, query_dates = build_split_arrays(split, daily, windows)
    stratum = build_split_target_stratum(split, daily, windows)

    # Retrieved analogues: the same top-K RA-v1 uses, in rank order.
    an = pd.read_parquet(ANALOGUES_PATH)
    an = an[(an["rank"] <= K) & an["query_date"].isin(query_dates)]
    dates = an.pivot(index="query_date", columns="rank", values="analogue_query_date").reindex(query_dates)
    if dates.isna().any().any():
        raise ValueError("some val queries have fewer than K analogues")
    z = np.stack([_z_outcomes(daily, dates[k].to_numpy()) for k in range(1, K + 1)], axis=1)  # (n, K, LEADS)
    anen = long_frame("anen", 0, query_dates, _forecast(daily, query_dates, z.mean(axis=1)), y_raw, stratum)

    # Random eligible analogues: val queries may use any train window (split rule;
    # the 19-day rule is then automatically satisfied).
    pool = windows.loc[windows["split"] == "train", "query_date"].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    parts = []
    for draw in range(N_RANDOM_DRAWS):
        pick = pool[rng.integers(0, len(pool), size=(len(query_dates), K))]
        zr = np.stack([_z_outcomes(daily, pick[:, k]) for k in range(K)], axis=1)
        parts.append(long_frame("anen_random", draw, query_dates, _forecast(daily, query_dates, zr.mean(axis=1)), y_raw, stratum))
    anen_random = pd.concat(parts, ignore_index=True)

    out = OUT_DIR / split
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("anen", anen), ("anen_random", anen_random)):
        df.to_parquet(out / f"{name}.parquet", index=False)
        rmse = df.groupby("seed")["error"].apply(lambda e: float(np.sqrt(np.mean(e**2)))).mean()
        print(f"  [{split}] {name:12s} rows={len(df):7d} mean RMSE={rmse:.4f} -> {(out / f'{name}.parquet').relative_to(REPO_ROOT)}")
    return {"anen": anen, "anen_random": anen_random}


if __name__ == "__main__":
    run()
