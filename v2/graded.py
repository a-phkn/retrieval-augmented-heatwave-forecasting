"""
PROTOCOL_V2 amendment A1 (24 Nov 2026, written after the H-A result; exploratory, outside the Holm family):
H-A and H-E re-expressed with required rest-hours (v2.work_rest) instead of the binary unsafe hour/day.

H-A1: on S2 alert days, the share of required rest-hours that fall outside each fixed window, and the same
      best-past-window test as H-A with hours weighted by their rest share.
H-E1: per insurance trigger and ERA5 cell, the share of in-period rest-hours that fall on days that do not
      count toward a payout; and across cell-seasons, Spearman rank correlation between season rest-hours and
      the number of days that count toward a payout (descriptive, no test).
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from v2.exposure import WORK_HOURS
from v2.insurance_triggers import Trigger, mark_trigger
from v2.window_test import CANDIDATES, DEV_END, SEARCH_END, WINDOWS, covers, jackknife
from v2.work_rest import rest_share


def rest_hour_table(hourly_mean: pd.DataFrame, alert_days, work: str, level: str, col: str = "wbgt") -> pd.DataFrame:
    """One row per working hour with a positive rest share on an alert day: date, hour, weight."""
    h = hourly_mean.assign(date=hourly_mean["time"].dt.normalize(), hour=hourly_mean["time"].dt.hour)
    h = h[h["date"].isin(pd.DatetimeIndex(alert_days)) & h["hour"].isin(WORK_HOURS)]
    h = h.assign(weight=rest_share(h[col], work, level))
    return h.loc[h["weight"] > 0, ["date", "hour", "weight"]].reset_index(drop=True)


def wcoverage(u: pd.DataFrame, window) -> float:
    tot = u["weight"].sum()
    return float(u["weight"][covers(window, u["hour"].to_numpy())].sum() / tot) if tot > 0 else float("nan")


def best_window_w(u: pd.DataFrame) -> tuple[int, int]:
    return CANDIDATES[int(np.argmax([wcoverage(u, w) for w in CANDIDATES]))]


def h_a1(hourly_mean, alert_days, work: str, level: str, col: str = "wbgt") -> dict:
    u = rest_hour_table(hourly_mean[hourly_mean["time"].dt.year <= DEV_END], alert_days, work, level, col)
    search, hold = u[u["date"].dt.year <= SEARCH_END], u[u["date"].dt.year > SEARCH_END]
    best = best_window_w(search)
    w1 = WINDOWS["W1 12-4 pm (HAP p.155)"]
    diff = jackknife(hold, lambda x: wcoverage(x, best) - wcoverage(x, w1))
    n_days = len(pd.DatetimeIndex(alert_days))
    out = {"work": work, "level": level, "wbgt": col, "alert_days": n_days,
           "rest_hours_per_alert_day": float(u["weight"].sum() / n_days) if n_days else float("nan"),
           "best_past_window": f"{best[0]}-{best[1]}", "diff_best_minus_W1": asdict(diff)}
    for name, w in WINDOWS.items():
        out[f"share_rest_outside {name}"] = asdict(jackknife(u, lambda x, w=w: 1.0 - wcoverage(x, w)))
    return out


def audit_graded(daily_rest: pd.DataFrame, trig: Trigger) -> dict:
    """daily_rest: cell, date, tmax, rest_hours (v2.work_rest.daily_rest)."""
    d = mark_trigger(daily_rest, trig)
    p = d[d["in_period"]]
    s = p.groupby(["cell", "season"]).agg(rest=("rest_hours", "sum"), paid_days=("counts", "sum"),
                                          paid=("season_paid", "first"))
    rho = sp_stats.spearmanr(s["rest"], s["paid_days"]).statistic if s["paid_days"].nunique() > 1 else float("nan")
    top = s["rest"] >= s["rest"].quantile(0.75)
    return {"trigger": trig.name, "cell_seasons": int(len(s)),
            "rest_hours_per_season": float(s["rest"].mean()),
            "share_rest_hours_on_days_not_counted": float(p.loc[~p["counts"], "rest_hours"].sum() / p["rest_hours"].sum()),
            "share_seasons_paid": float(s["paid"].mean()),
            "share_top_quartile_rest_seasons_unpaid": float((~s.loc[top, "paid"]).mean()),
            "spearman_season_rest_vs_paid_days": float(rho)}
