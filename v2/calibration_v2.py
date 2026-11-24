"""
H-C (heat-conditioned conformal ranges) and E5 (rest-hours trigger by conformal risk control), PROTOCOL_V2
Section 3 H-C and amendment A2 (Section 12). Pure functions on forecast tables; v2/run_calibration_v2.py loads
the data.

Forecast tables (format of predictions_v2/<run>/<fold>.parquet and predictions_v2/inner/<run>/<fold>.parquet):
seed, lead, target_date, pred, actual. Per the v1 convention the metric is the mean of per-seed metrics: every
statistic below is computed per seed and averaged; for the jackknife, the per-row seed average is the unit, so
the jackknife is over the same mean.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from evaluation.stats import MIN_CLUSTERS, make_cluster_ids
from v2.risk_control import crc_threshold

LEVEL = 0.80
BAND = (0.75, 0.85)
HOT_QUANTILE = 0.90            # forecast-hot = top decile of in-season inner-block forecasts
ALPHA, TOLERANCE = 0.20, 0.05  # E5: H0 miss rate >= ALPHA + TOLERANCE
MAX_FALSE_ALARM = 0.40         # E5 second part (A2.5): fire rate on non-event days (base-rate free)
SEASON = (315, 930)            # Mar 15 - Sep 30, as month*100+day


def in_season(dates) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    md = d.month * 100 + d.day
    return np.asarray((md >= SEASON[0]) & (md <= SEASON[1]))


def conformal_offsets(resid: np.ndarray, level: float = LEVEL) -> tuple[float, float]:
    """Split-conformal two-sided offsets with the finite-sample correction: the ceil((n+1)(1+level)/2)-th and
    floor((n+1)(1-level)/2)-th order statistics of actual - pred."""
    r = np.sort(np.asarray(resid, float))
    n = len(r)
    hi_k = int(np.ceil((n + 1) * (1 + level) / 2 - 1e-9))   # tolerance: (1-0.8)/2 is 0.0999.. in floats
    lo_k = int(np.floor((n + 1) * (1 - level) / 2 + 1e-9))
    hi = r[min(hi_k, n) - 1] if hi_k <= n else np.inf
    lo = r[lo_k - 1] if lo_k >= 1 else -np.inf
    return float(lo), float(hi)


def mondrian_intervals(inner: pd.DataFrame, val: pd.DataFrame, level: float = LEVEL) -> pd.DataFrame:
    """Per (seed, lead): hot threshold = HOT_QUANTILE of in-season inner forecasts; separate conformal offsets
    for hot and other inner days; applied to val by the same forecast threshold. Adds lo, hi, hot, hit."""
    out = []
    for (seed, lead), v in val.groupby(["seed", "lead"]):
        c = inner[(inner["seed"] == seed) & (inner["lead"] == lead)]
        if c.empty:
            raise ValueError(f"no inner forecasts for seed {seed}, lead {lead}")
        thr = np.quantile(c.loc[in_season(c["target_date"]), "pred"], HOT_QUANTILE)
        c_hot = c["pred"].to_numpy() >= thr
        r = (c["actual"] - c["pred"]).to_numpy()
        off = {True: conformal_offsets(r[c_hot], level), False: conformal_offsets(r[~c_hot], level)}
        hot = v["pred"].to_numpy() >= thr
        lo = np.where(hot, off[True][0], off[False][0]) + v["pred"].to_numpy()
        hi = np.where(hot, off[True][1], off[False][1]) + v["pred"].to_numpy()
        out.append(v.assign(lo=lo, hi=hi, hot=hot, hit=(v["actual"] >= lo) & (v["actual"] <= hi)))
    return pd.concat(out, ignore_index=True)


@dataclass(frozen=True)
class Est:
    estimate: float
    se: float
    n_clusters: int
    n: int


def _jack(rows: pd.DataFrame, col: str) -> Est:
    """rows: one per target date (seed-averaged `col`); delete-one-cluster jackknife of the mean, CV3J."""
    cl = make_cluster_ids(rows["target_date"])
    ids = np.unique(cl)
    est = float(rows[col].mean())
    G = len(ids)
    if G < MIN_CLUSTERS:
        return Est(est, float("nan"), G, len(rows))
    reps = np.array([rows.loc[cl != g, col].mean() for g in ids])
    return Est(est, float(np.sqrt((G - 1) / G * np.sum((reps - reps.mean()) ** 2))), G, len(rows))


def _p_greater(e: Est, value: float) -> float:
    """One-sided p for H1: mean > value."""
    if not np.isfinite(e.se) or e.se == 0:
        return float("nan")
    return float(sp_stats.t.sf((e.estimate - value) / e.se, e.n_clusters - 1))


def h_c(folds: list[tuple[pd.DataFrame, pd.DataFrame]], lead: int = 1) -> dict:
    """folds: (inner, val) per fold; each fold is calibrated on its own inner block, val rows are pooled."""
    iv = pd.concat([mondrian_intervals(i[i["lead"] == lead], v[v["lead"] == lead]) for i, v in folds])
    g = iv.groupby("target_date")
    per_date = pd.DataFrame({"hit": g["hit"].mean(), "hot": g["hot"].mean(),
                             "width": g["hi"].mean() - g["lo"].mean()}).reset_index()
    res, ps = {"lead": lead}, []
    for name, rows in (("all_days", per_date), ("forecast_hot", per_date[per_date["hot"] >= 0.5])):
        e = _jack(rows, "hit")
        p_lo = _p_greater(e, BAND[0])                                            # coverage > 0.75
        p_hi = _p_greater(Est(-e.estimate, e.se, e.n_clusters, e.n), -BAND[1])  # coverage < 0.85
        ps += [p_lo, p_hi]
        res[name] = {**asdict(e), "p_above_0.75": p_lo, "p_below_0.85": p_hi,
                     "mean_width": float(rows["width"].mean())}
    res["p_tost"] = float(max(ps)) if np.all(np.isfinite(ps)) else float("nan")
    return res


def e5(folds: list[tuple[pd.DataFrame, pd.DataFrame]], event: pd.Series, lead: int = 1,
       alpha: float = ALPHA, severe: pd.Series | None = None) -> dict:
    """event: bool Series by date (observed rest-hours >= R). Score = pred (daily max WBGT forecast). Per fold and
    seed: lambda by CRC on that fold's in-season inner days; trigger on val when pred >= lambda; pooled.
    severe (optional, bool by date): a stricter subset of event days; its miss rate is reported (secondary)."""
    ev = event.astype(bool)
    rows, lambdas = [], {}
    for k, (inner, val) in enumerate(folds):
        for seed, v in val[val["lead"] == lead].groupby("seed"):
            c = inner[(inner["seed"] == seed) & (inner["lead"] == lead)]
            c = c[in_season(c["target_date"]) & c["target_date"].isin(ev.index)]
            lam = crc_threshold(c["pred"].to_numpy(), ev.reindex(c["target_date"]).to_numpy(), alpha)
            lambdas[f"fold{k}/seed{seed}"] = lam
            v = v[in_season(v["target_date"]) & v["target_date"].isin(ev.index)]
            rows.append(v.assign(fire=v["pred"].to_numpy() >= lam, event=ev.reindex(v["target_date"]).to_numpy()))
    t = pd.concat(rows, ignore_index=True)
    d = t.groupby("target_date").agg(fire=("fire", "mean"), event=("event", "first")).reset_index()
    events = d[d["event"]].assign(miss=lambda x: 1.0 - x["fire"])
    e = _jack(events, "miss")
    p = _p_greater(Est(-e.estimate, e.se, e.n_clusters, e.n), -(alpha + TOLERANCE))   # H1: miss < alpha + tol
    years = pd.DatetimeIndex(d["target_date"]).year
    base = float(d["event"].mean())
    fire_rate = float(d["fire"].mean())
    precision = float((d["fire"] * d["event"]).sum() / max(1e-12, d["fire"].sum()))
    lift = precision / base if base > 0 else float("nan")
    far = float(d.loc[~d["event"], "fire"].mean()) if (~d["event"]).any() else float("nan")
    sev = {}
    if severe is not None:
        sv = events[events["target_date"].isin(severe[severe.astype(bool)].index)]
        sev = {"severe_event_days": int(len(sv)), "severe_miss_rate": asdict(_jack(sv, "miss")) if len(sv) else None}
    return {"lead": lead, "alpha": alpha, "miss_rate": asdict(e), "p_miss_below_alpha_plus_tol": p,
            "base_rate": base, "fire_rate": fire_rate, "precision": precision, "lift": lift,
            "false_alarm_rate": far,
            "peirce_skill": float(1.0 - e.estimate - far),
            "false_alarm_ok": bool(far <= MAX_FALSE_ALARM),
            "triggers_per_season": float(d["fire"].sum() / max(1, len(np.unique(years)))),
            "event_days": int(len(events)), "days": int(len(d)), "lambdas": lambdas, **sev}


def holm(pvals: dict[str, float], fwer: float = 0.05) -> dict[str, dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": p, "p_holm": adj, "reject": adj < fwer}
    return out
