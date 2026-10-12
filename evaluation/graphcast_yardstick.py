"""
External yardstick for the paper (user decision 2026-10-12: "only for paper purposes"; the
backbone stays C3-hop): how far is our local model from a global pre-trained GNN weather model?

GraphCast (Lam et al. 2023, Science 382: 1416), ERA5-initialised forecasts published by
WeatherBench 2 (Rasp et al. 2024; public bucket gs://weatherbench2, read anonymously), 0.7°
conservative regridding (512x256), 2018, read chunk by chunk over HTTPS. Tmax only: GraphCast has no 2 m dew point or radiation,
so WBGT cannot be computed.

Design (fixed before any GraphCast number was looked at):
- Information timing: for a window whose first forecast day is q, GraphCast starts at 12 UTC
  (17:30 IST) on q - 1, so it has no information our models lack (their inputs end on q - 1).
- Area: area-weighted mean over the 0.7° cells overlapping the 9-cell Delhi box
  (28.125-28.875 N, 76.875-77.625 E), the same area as our Tmax target.
- Daily maximum: GraphCast gives 2 m temperature every 6 h (00/06/12/18 UTC = 05:30/11:30/17:30/
  23:30 IST), which misses the ~14:00-15:00 IST peak. A per-calendar-month linear map
  "true daily max = a + b x 6-hourly-sampled max" is fitted on OUR ERA5 hourly data, 1980-2015
  (fold f4's training years; 2018 is not used), and applied to GraphCast. The same map applied
  to ERA5's own 6-hourly samples in 2018 gives the method's floor: the error a perfect 6-hourly
  forecast would still have.
- Windows: fold f4's validation windows with query dates in 2018 (targets end before 2019; the
  locked test period is never read). Compared with our models' forecasts on exactly those windows.
- Reported descriptively (RMSE and bias, all days, extreme days, by lead): one year has only 3
  year x season clusters, too few for the paired cluster test.

Run from repo root:  python -m evaluation.graphcast_yardstick
Writes data/external/graphcast_2018_delhi_t2m.parquet (the small extract, cached) and
evaluation_v2/graphcast_yardstick.{md,json}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.compare_v2 import load_run, rmse_by
from evaluation.predict_v1 import long_frame
from evaluation.retrieval_information_check import OUT_DIR
from pipeline.hourly_features import load_cell_hourly
from training.folds import FORECAST_DAYS, TEST_START, build_fold

REPO_ROOT = Path(__file__).resolve().parents[1]
GC_URL = ("https://storage.googleapis.com/weatherbench2/datasets/graphcast/2018/"
          "date_range_2017-11-16_2019-02-01_12_hours-512x256_equiangular_conservative.zarr")
EXTRACT = REPO_ROOT / "data" / "external" / "graphcast_2018_delhi_t2m.parquet"
BOX = (28.125, 28.875, 76.875, 77.625)  # lat_min, lat_max, lon_min, lon_max of the 9 ERA5 cells
SAMPLE_HOURS_IST = ((5, 6), (11, 12), (17, 18), (23, 24))  # IST hours bracketing 00/06/12/18 UTC (:30)
FIT_YEARS = (1980, 2015)
YEAR = 2018
FAMILY_RUNS = {"control (A1prime_hw5)": "A1prime_hw5", "C3-hop": "A1prime_hw5_C3hop", "C2": "A1prime_hw5_C2",
               "U1 (tuned)": "A1prime_hw5_U1_tuned", "ridge": "A1prime_hw5_ridge"}


# ------------------------------------------------------------------ geometry and timing


def overlap_weights(cell_edges_lat: np.ndarray, cell_edges_lon: np.ndarray, box=BOX) -> np.ndarray:
    """(n_lat, n_lon) fraction of the box covered by each grid cell (edges ascending; sums to 1)."""
    la = np.clip(np.minimum(cell_edges_lat[1:], box[1]) - np.maximum(cell_edges_lat[:-1], box[0]), 0, None)
    lo = np.clip(np.minimum(cell_edges_lon[1:], box[3]) - np.maximum(cell_edges_lon[:-1], box[2]), 0, None)
    w = la[:, None] * lo[None, :]
    return w / w.sum()


def init_and_leads(query_date: pd.Timestamp, lead: int) -> tuple[pd.Timestamp, list[pd.Timedelta]]:
    """GraphCast start (12 UTC on q - 1) and the 4 lead times hitting 00/06/12/18 UTC of the target
    day D = q + lead - 1 (the IST calendar day D contains those four UTC times)."""
    init = query_date - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
    target = query_date + pd.Timedelta(days=lead - 1)
    return init, [target + pd.Timedelta(hours=h) - init for h in (0, 6, 12, 18)]


# ------------------------------------------------------------------ sampling correction (ERA5, local)


def era5_daily_true_and_sampled() -> pd.DataFrame:
    """Per IST day: 'true' = mean over the 9 cells of each cell's daily max (the Tmax target's
    definition); 'sampled' = the same with only the 4 six-hourly times (each the mean of the two
    IST hours bracketing XX:30)."""
    true, samp = [], []
    for cell in range(1, 10):
        h = load_cell_hourly(cell).set_index("time")["t"]
        true.append(h.groupby(h.index.normalize()).max())
        true[-1] = true[-1][true[-1].index < TEST_START]
        samp.append(sampled_daily_max(h))
    out = pd.DataFrame({"true": pd.concat(true, axis=1).mean(axis=1), "sampled": pd.concat(samp, axis=1).mean(axis=1)})
    return out[out.index < TEST_START].dropna()


def sampled_daily_max(h: pd.Series) -> pd.Series:
    """Daily max over the 4 times XX:30 IST for XX in SAMPLE_HOURS_IST's first hours (= 00/06/12/18
    UTC), each estimated as the mean of the two bracketing hourly values (XX:00 and XX+1:00)."""
    half = (h + h.shift(-1)) / 2  # value at XX:30 = mean of XX:00 and XX+1:00
    s = half[np.isin(half.index.hour, [a for a, _ in SAMPLE_HOURS_IST])].dropna()
    return s.groupby(s.index.normalize()).max()


def fit_correction(daily: pd.DataFrame, years=FIT_YEARS) -> dict[int, tuple[float, float]]:
    """Per calendar month: (a, b) of true = a + b x sampled, least squares on the given years."""
    sel = daily[(daily.index.year >= years[0]) & (daily.index.year <= years[1])]
    out = {}
    for m, g in sel.groupby(sel.index.month):
        b, a = np.polyfit(g["sampled"].to_numpy(), g["true"].to_numpy(), 1)
        out[int(m)] = (float(a), float(b))
    return out


def apply_correction(sampled: np.ndarray, months: np.ndarray, coef: dict[int, tuple[float, float]]) -> np.ndarray:
    a = np.array([coef[int(m)][0] for m in months])
    b = np.array([coef[int(m)][1] for m in months])
    return a + b * sampled


# ------------------------------------------------------------------ GraphCast extract (network, cached)


def chunk_plan(inits: list[pd.Timestamp], lead_hours: list[int], times: pd.DatetimeIndex,
               file_leads: np.ndarray, lead_chunk: int) -> dict[tuple[int, int], list[tuple[pd.Timestamp, int, int]]]:
    """Which zarr chunks to download: {(time index, lead chunk): [(init, lead_hours, index in chunk)]}."""
    plan: dict[tuple[int, int], list] = {}
    for init in inits:
        ti = times.get_loc(init)
        for h in lead_hours:
            li = int(np.flatnonzero(file_leads == h)[0])
            plan.setdefault((ti, li // lead_chunk), []).append((init, h, li % lead_chunk))
    return plan


def fetch_graphcast(query_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Box-mean 2 m temperature (°C) per (init, lead_hours), for the inits and leads the windows
    need; only the needed chunks are downloaded (plain HTTPS: the gcsfs client hung on this machine),
    cached in EXTRACT."""
    if EXTRACT.exists():
        return pd.read_parquet(EXTRACT)
    from concurrent.futures import ThreadPoolExecutor

    import numcodecs
    import requests

    sess = requests.Session()

    def get(path: str) -> bytes:
        for attempt in range(5):
            try:
                r = sess.get(f"{GC_URL}/{path}", timeout=120)
                r.raise_for_status()
                return r.content
            except requests.RequestException:
                if attempt == 4:
                    raise
        raise AssertionError

    meta = json.loads(get(".zmetadata"))["metadata"]

    def coord(name: str) -> np.ndarray:
        z = meta[f"{name}/.zarray"]
        return np.frombuffer(numcodecs.Blosc().decode(get(f"{name}/0")), dtype=z["dtype"])

    za, attrs = meta["2m_temperature/.zarray"], meta["2m_temperature/.zattrs"]
    if attrs["_ARRAY_DIMENSIONS"] != ["time", "prediction_timedelta", "longitude", "latitude"] or za["chunks"][0] != 1:
        raise ValueError(f"unexpected GraphCast layout: {attrs['_ARRAY_DIMENSIONS']} chunks {za['chunks']}")
    if meta["time/.zattrs"]["units"] != "hours since 2017-11-16" or meta["prediction_timedelta/.zattrs"]["units"] != "hours":
        raise ValueError("unexpected time units in the GraphCast file")
    times = pd.DatetimeIndex(pd.Timestamp("2017-11-16") + pd.to_timedelta(coord("time"), unit="h"))
    file_leads = coord("prediction_timedelta")
    lat, lon = coord("latitude"), coord("longitude")
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
        raise ValueError("expected ascending latitude and longitude in the GraphCast grid")
    dlat, dlon = float(np.diff(lat).mean()), float(np.diff(lon).mean())
    ilat = np.flatnonzero((lat + dlat / 2 > BOX[0]) & (lat - dlat / 2 < BOX[1]))
    ilon = np.flatnonzero((lon + dlon / 2 > BOX[2]) & (lon - dlon / 2 < BOX[3]))
    w = overlap_weights(np.r_[lat[ilat] - dlat / 2, lat[ilat[-1]] + dlat / 2],
                        np.r_[lon[ilon] - dlon / 2, lon[ilon[-1]] + dlon / 2])  # (n_lat, n_lon)

    inits = sorted({init_and_leads(q, 1)[0] for q in query_dates})
    lead_hours = sorted({int(lt.total_seconds() // 3600) for q in query_dates
                         for L in range(1, FORECAST_DAYS + 1) for lt in init_and_leads(q, L)[1]})
    plan = chunk_plan(inits, lead_hours, times, file_leads, za["chunks"][1])

    def box_means(key: tuple[int, int]) -> list[tuple[pd.Timestamp, int, float]]:
        ti, ci = key
        raw = numcodecs.Blosc().decode(get(f"2m_temperature/{ti}.{ci}.0.0"))
        a = np.frombuffer(raw, dtype=za["dtype"]).reshape(za["chunks"])[0]  # (lead, lon, lat)
        return [(init, h, float((a[li][np.ix_(ilon, ilat)].T * w).sum()) - 273.15) for init, h, li in plan[key]]

    rows, keys = [], sorted(plan)
    print(f"GraphCast: {len(inits)} inits, {len(keys)} chunks to download", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, part in enumerate(pool.map(box_means, keys), 1):
            rows += part
            if i % 50 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)} chunks", flush=True)
    out = pd.DataFrame(rows, columns=["init", "lead_hours", "t2m"]).sort_values(["init", "lead_hours"])
    if out["t2m"].isna().any():
        raise ValueError("missing GraphCast values in the extract")
    EXTRACT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(EXTRACT, index=False)
    return out


# ------------------------------------------------------------------ evaluation


def windows_2018() -> tuple:
    """Fold f4's validation windows with query dates in YEAR (Tmax, labels v2)."""
    val = build_fold("f4", "t_max", "v2").val
    keep = pd.DatetimeIndex(val.query_dates).year == YEAR
    return val.subset(keep)


def graphcast_frame(val, gc: pd.DataFrame, coef) -> pd.DataFrame:
    g = gc.set_index(["init", "lead_hours"])["t2m"]
    q = pd.DatetimeIndex(val.query_dates)
    sampled = np.empty((len(q), FORECAST_DAYS))
    for i, qd in enumerate(q):
        for L in range(1, FORECAST_DAYS + 1):
            init, lts = init_and_leads(qd, L)
            sampled[i, L - 1] = max(g.loc[(init, int(lt.total_seconds() // 3600))] for lt in lts)
    months = np.array([[(qd + pd.Timedelta(days=L)).month for L in range(FORECAST_DAYS)] for qd in q])
    pred = apply_correction(sampled.ravel(), months.ravel(), coef).reshape(sampled.shape)
    return long_frame("graphcast", 0, val.query_dates, pred, val.y_raw, val.stratum)


def floor_frame(val, daily: pd.DataFrame, coef) -> pd.DataFrame:
    """A perfect 6-hourly forecast (ERA5's own samples) put through the same correction."""
    q = pd.DatetimeIndex(val.query_dates)
    tgt = np.array([[qd + pd.Timedelta(days=L) for L in range(FORECAST_DAYS)] for qd in q])
    s = daily["sampled"].reindex(pd.DatetimeIndex(tgt.ravel())).to_numpy()
    pred = apply_correction(s, np.array([t.month for t in tgt.ravel()]), coef).reshape(tgt.shape)
    return long_frame("perfect_6h_floor", 0, val.query_dates, pred, val.y_raw, val.stratum)


def summarise(df: pd.DataFrame) -> dict:
    ext = df[df["stratum"] == "extreme"]
    return {"rmse_all": rmse_by(df), "rmse_extreme": rmse_by(ext) if len(ext) else float("nan"),
            "bias_all": float(df["error"].mean()), "bias_extreme": float(ext["error"].mean()) if len(ext) else float("nan"),
            "by_lead": [rmse_by(df[df["lead"] == L]) for L in range(1, FORECAST_DAYS + 1)], "n_extreme": int(len(ext))}


def main() -> None:
    val = windows_2018()
    daily = era5_daily_true_and_sampled()
    coef = fit_correction(daily)
    gc = fetch_graphcast(pd.DatetimeIndex(val.query_dates))
    keys = pd.DataFrame({"query_date": np.repeat(pd.DatetimeIndex(val.query_dates), FORECAST_DAYS),
                         "lead": np.tile(np.arange(1, FORECAST_DAYS + 1), len(val.query_dates))})
    res = {"GraphCast (corrected)": summarise(graphcast_frame(val, gc, coef)),
           "perfect 6-hourly forecast (method floor)": summarise(floor_frame(val, daily, coef))}
    for name, rid in FAMILY_RUNS.items():
        df = load_run(rid)
        if df is not None:
            res[name] = summarise(df.merge(keys, on=["query_date", "lead"], how="inner"))
    write_report(res, coef, len(val.query_dates))
    (OUT_DIR / "graphcast_yardstick.json").write_text(json.dumps({"results": res, "correction": coef}, indent=2,
                                                                 default=float) + "\n", encoding="utf-8")
    print({k: round(v["rmse_all"], 3) for k, v in res.items()})


def write_report(res: dict, coef: dict, n_windows: int) -> None:
    L = ["# External yardstick: GraphCast vs our models, Delhi Tmax, 2018 (paper only)", "",
         f"{n_windows} forecast windows (query dates in 2018, fold f4 validation block), 5 leads. GraphCast: "
         "WeatherBench 2 ERA5-initialised forecasts, 0.7° grid, started 12 UTC the day before the first forecast "
         "day; daily maximum from its 6-hourly values via a per-month correction fitted on our ERA5 1980-2015. "
         "Descriptive only (one year = 3 clusters). Our models: mean of per-seed RMSE (10 seeds), same windows. "
         "GraphCast starts from the full global atmosphere; ours from 28 points' daily data, so this measures the "
         "gap to a global AI weather model, not a like-for-like contest. Tmax only (no WBGT from GraphCast).", "",
         "| Model | RMSE all | RMSE extreme | Bias all | Bias extreme | Lead 1 | 2 | 3 | 4 | 5 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        L.append(f"| {name} | {r['rmse_all']:.3f} | {r['rmse_extreme']:.3f} | {r['bias_all']:+.2f} | "
                 f"{r['bias_extreme']:+.2f} | " + " | ".join(f"{x:.3f}" for x in r["by_lead"]) + " |")
    n_ext = next(iter(res.values()))["n_extreme"]
    L += ["", f"Extreme-day rows: {n_ext} (window x lead). The method floor is the error a perfect 6-hourly forecast "
          "would still have after the sampling correction; GraphCast is not expected to do better than it. With only "
          f"{n_ext} extreme rows, the extreme-day column is too noisy to rank models."]
    (OUT_DIR / "graphcast_yardstick.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
