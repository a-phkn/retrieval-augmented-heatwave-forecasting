"""
H-A (PROTOCOL_V2): do Delhi's fixed work windows miss unsafe working hours on alert days?

Conventions (PROTOCOL_V2 Sections 1.2, 1.4): hour-ending stamps; a window "a-b" (24 h clock) covers stamps
a < h <= b, so 12-4 pm covers 13, 14, 15, 16. Unsafe hour: 9-cell mean hourly WBGT >= limit at a working
stamp 06..20. Alert days: S2-Yellow, Safdarjung station Tmax >= 40 C, Mar 15-Sep 30; days without a station
value, or with a TMAX quality flag, are excluded and counted. Sensitivities: S1 (IMD, 2 consecutive days at
Safdarjung), S3 (HAP departure >= +4 C), and the ERA5 9-cell Tmax proxy, which has no gaps.

Primary rule: the 4-hour window that covers the most unsafe hours on 1980-2006 alert days ("best past window")
is compared with W1 (12-4 pm) on 2007-2018 alert days. Supported if the delete-one-cluster jackknife 95% CI of
coverage(best) - coverage(W1) is above 0. Clusters: year x season blocks (evaluation.stats.make_cluster_ids);
jackknife SE centred on the replicate mean (CV3J), t with G-1 df, as evaluation/stats.py.
Ties in the window search go to the earliest start (stated in advance).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from evaluation.stats import MIN_CLUSTERS, make_cluster_ids
from v2.exposure import PRIMARY_LIMIT, WORK_HOURS

WINDOWS = {"W1 12-4 pm (HAP p.155)": (12, 16), "W2 1-5 pm (HAP p.81)": (13, 17), "W3 12-3 pm (IMD public)": (12, 15)}
SEASON = ("03-15", "09-30")
DEV_END, SEARCH_END = 2018, 2006
CANDIDATES = tuple((s, s + 4) for s in range(WORK_HOURS[0] - 1, WORK_HOURS[-1] - 3))  # stamps 06..20 inside


def covers(window: tuple[int, int], hours: np.ndarray) -> np.ndarray:
    a, b = window
    return (hours > a) & (hours <= b)


def _season_index(years=range(1980, DEV_END + 1)) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([d for y in years for d in pd.date_range(f"{y}-{SEASON[0]}", f"{y}-{SEASON[1]}")])


def _station_tmax(station: pd.DataFrame) -> pd.Series:
    """Station Tmax by date with quality-flagged values set to NaN (PROTOCOL_V2 H-A)."""
    s = station.assign(date=pd.to_datetime(station["date"])).set_index("date").sort_index()
    t = s["tmax_c"].astype(float)
    if "tmax_qflag" in s:
        t = t.where(s["tmax_qflag"].isna())
    return t


def _classify(tmax: pd.Series, alert: pd.Series) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Restrict to season days 1980-2018; days whose rule cannot be evaluated (NaN) are excluded and counted."""
    idx = _season_index()
    a = alert.reindex(idx)
    usable = a.notna()
    missing = (~usable).groupby(idx.year).sum().rename("unclassified_days").to_frame()
    return idx[usable & a.fillna(False).astype(bool)], missing


def doy_normal(tmax: pd.Series, years=(1981, 2010), smooth_days: int = 31) -> pd.Series:
    """Day-of-year mean over `years`, circular centred moving average (PROTOCOL_V2 1.3: 1981-2010, 31-day)."""
    t = tmax[(tmax.index.year >= years[0]) & (tmax.index.year <= years[1])].dropna()
    m = t.groupby(t.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    h = smooth_days // 2
    ext = pd.concat([m.iloc[-h:], m, m.iloc[:h]])
    return ext.rolling(smooth_days, center=True).mean().iloc[h:-h].set_axis(range(1, 367))


def departure(tmax: pd.Series, normal: pd.Series) -> pd.Series:
    return tmax - normal.reindex(tmax.index.dayofyear).to_numpy()


def s2_alert_days(station: pd.DataFrame, threshold: float = 40.0) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """S2-Yellow (primary): station Tmax >= 40 C. station: date, tmax_c, tmax_qflag."""
    t = _station_tmax(station)
    return _classify(t, (t >= threshold).where(t.notna()))


def s1_alert_days(station: pd.DataFrame, normal_years=(1981, 2010)) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """S1, IMD heat wave at Safdarjung only: criterion (Tmax >= 40 and departure >= 4.5, or Tmax >= 45) met on
    the day and the day before; declared on the second day (PROTOCOL_V2 1.3, single-station approximation)."""
    t = _station_tmax(station)
    t = t.reindex(pd.date_range(t.index.min(), t.index.max()))
    dep = departure(t, doy_normal(t, normal_years))
    met = (((t >= 40.0) & (dep >= 4.5)) | (t >= 45.0)).where(t.notna())
    today, prev = met.astype(float), met.shift(1).astype(float)    # 1 met, 0 not met, NaN unknown
    alert = pd.Series(np.nan, index=t.index)
    alert[(today == 1) & (prev == 1)] = 1.0
    alert[(today == 0) | (prev == 0)] = 0.0                        # one known non-criterion day settles "no"
    return _classify(t, alert.map({1.0: True, 0.0: False}))


def s3_alert_days(station: pd.DataFrame, normal_years=(1981, 2010)) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """S3, Delhi HAP departure table, Orange or Red: departure >= +4 C above normal Tmax."""
    t = _station_tmax(station)
    dep = departure(t, doy_normal(t, normal_years))
    return _classify(t, (dep >= 4.0).where(t.notna()))


def era5_proxy_alert_days(cell_tmax: pd.Series) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Sensitivity (PROTOCOL_V2 H-A): the go/no-go note's rule on the 9-cell ERA5 Tmax (no gaps):
    Mar 15-Jul 31, (Tmax >= 40 and anomaly >= 3) or Tmax >= 45, anomaly vs the 1980-2018 day-of-year mean,
    31-day smoothed. cell_tmax: daily 9-cell Tmax indexed by date."""
    t = cell_tmax.sort_index()
    anom = departure(t, doy_normal(t, (1980, DEV_END)))
    md = t.index.month * 100 + t.index.day
    alert = (((t >= 40.0) & (anom >= 3.0)) | (t >= 45.0)) & (md >= 315) & (md <= 731)
    return _classify(t, alert.where(t.notna()))


def unsafe_hour_table(hourly_mean: pd.DataFrame, alert_days, limit: float = PRIMARY_LIMIT) -> pd.DataFrame:
    """hourly_mean: time (IST), wbgt (9-cell mean). One row per unsafe working hour on an alert day."""
    h = hourly_mean.assign(date=hourly_mean["time"].dt.normalize(), hour=hourly_mean["time"].dt.hour)
    h = h[h["date"].isin(pd.DatetimeIndex(alert_days)) & h["hour"].isin(WORK_HOURS) & (h["wbgt"] >= limit)]
    return h[["date", "hour"]].reset_index(drop=True)


def best_window(unsafe: pd.DataFrame) -> tuple[int, int]:
    hours = unsafe["hour"].to_numpy()
    counts = [int(covers(w, hours).sum()) for w in CANDIDATES]
    return CANDIDATES[int(np.argmax(counts))]  # argmax returns the first maximum: earliest start


def coverage(unsafe: pd.DataFrame, window) -> float:
    return float(covers(window, unsafe["hour"].to_numpy()).mean()) if len(unsafe) else float("nan")


@dataclass(frozen=True)
class JackknifeResult:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    n_clusters: int
    n_unsafe_hours: int


def jackknife(unsafe: pd.DataFrame, stat, level: float = 0.95) -> JackknifeResult:
    """Delete-one-cluster jackknife of a statistic of the unsafe-hour table (CV3J centring, t(G-1))."""
    clusters = make_cluster_ids(unsafe["date"])
    ids = np.unique(clusters)
    est = stat(unsafe)
    G = len(ids)
    if G < MIN_CLUSTERS:
        return JackknifeResult(est, np.nan, np.nan, np.nan, G, len(unsafe))
    reps = np.array([stat(unsafe[clusters != g]) for g in ids])
    se = float(np.sqrt((G - 1) / G * np.sum((reps - reps.mean()) ** 2)))
    q = sp_stats.t.ppf(0.5 + level / 2, G - 1)
    return JackknifeResult(est, se, est - q * se, est + q * se, G, len(unsafe))


def h_a(hourly_mean: pd.DataFrame, station: pd.DataFrame | None, limit: float = PRIMARY_LIMIT,
        alerts: tuple[pd.DatetimeIndex, pd.DataFrame] | None = None) -> dict:
    """Primary population S2-Yellow from `station`; pass `alerts` (days, unclassified) for S1/S3/ERA5 proxy."""
    alert, missing = alerts if alerts is not None else s2_alert_days(station)
    unsafe = unsafe_hour_table(hourly_mean[hourly_mean["time"].dt.year <= DEV_END], alert, limit)
    search = unsafe[unsafe["date"].dt.year <= SEARCH_END]
    hold = unsafe[unsafe["date"].dt.year > SEARCH_END]
    best = best_window(search)
    w1 = WINDOWS["W1 12-4 pm (HAP p.155)"]
    diff = jackknife(hold, lambda u: coverage(u, best) - coverage(u, w1))
    out = {"limit": limit, "alert_days": int(len(alert)), "unclassified_days": int(missing["unclassified_days"].sum()),
           "best_past_window": f"{best[0]}-{best[1]}", "primary_diff": asdict(diff),
           "supported": bool(np.isfinite(diff.ci_low) and diff.ci_low > 0)}
    for name, w in WINDOWS.items():
        out[f"share_outside {name}"] = asdict(jackknife(unsafe, lambda u, w=w: 1.0 - coverage(u, w)))
    return out
