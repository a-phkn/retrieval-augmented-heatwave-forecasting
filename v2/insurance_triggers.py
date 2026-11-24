"""
H-E (PROTOCOL_V2): how often do India's temperature-based heat-insurance triggers miss unsafe working days?
Descriptive only: a trigger built from WBGT would catch WBGT-unsafe days by construction, so no
"WBGT beats temperature" test is made. The paper calls the result exposure basis risk, never income loss.

Triggers are applied per ERA5 cell to the daily maximum 2 m temperature, as the schemes use ERA5.
"Exceeding" in the sources is read as strictly greater than. [CHECK] all values: they come from a programme
report and news (PROTOCOL_V2 H-E), not the contracts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Trigger:
    name: str
    city: str
    threshold: float          # deg C, daily maximum temperature, strictly exceeded
    rule: str                 # "season_count": pays if >= n_days hot days in the period (not necessarily consecutive)
                              # "consecutive": each run of >= n_days consecutive hot days is an event
    n_days: int
    start: str                # "MM-DD", risk period inclusive
    end: str
    source: str


TRIGGERS = (
    Trigger("VimoSEWA 43.2/7d", "ahmedabad", 43.2, "season_count", 7, "04-01", "07-31",
            "VimoSEWA x SCF report, Dec 2025, pp. 4, 21-22"),
    Trigger("SEWA 43.6/2c", "ahmedabad", 43.6, "consecutive", 2, "04-01", "07-31",
            "NPR, 16 Jul 2025 (risk period not stated; [CHECK], VimoSEWA's used)"),
    Trigger("SEWA 44.1/2c", "ahmedabad", 44.1, "consecutive", 2, "04-01", "07-31",
            "NPR, 16 Jul 2025 (risk period not stated; [CHECK], VimoSEWA's used)"),
    Trigger("MHT 45.27/1d", "delhi", 45.27, "consecutive", 1, "05-01", "07-31", "Down To Earth, May 2026 (day rule not stated: 1 day assumed, [CHECK])"),
    Trigger("MHT 46.53/1d", "delhi", 46.53, "consecutive", 1, "05-01", "07-31", "Down To Earth, May 2026 (day rule not stated: 1 day assumed, [CHECK])"),
    Trigger("MHT 47/1d", "delhi", 47.0, "consecutive", 1, "05-01", "07-31", "Down To Earth, May 2026 (day rule not stated: 1 day assumed, [CHECK])"),
)


def in_period(dates: pd.Series, start: str, end: str) -> pd.Series:
    md = dates.dt.strftime("%m-%d")
    return (md >= start) & (md <= end)


def _runs_at_least(flags: np.ndarray, k: int) -> np.ndarray:
    """True for every element that belongs to a run of >= k consecutive True values."""
    out = np.zeros(len(flags), dtype=bool)
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            if j - i >= k:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def mark_trigger(daily: pd.DataFrame, trig: Trigger) -> pd.DataFrame:
    """daily: one row per (cell, date) with tmax; consecutive calendar days assumed within a season.
    Adds: in_period, hot (tmax > threshold, in period), counts (the day counts toward a payout), and
    season_paid (the cell-season pays at least once)."""
    d = daily.sort_values(["cell", "date"]).reset_index(drop=True)
    d["in_period"] = in_period(d["date"], trig.start, trig.end)
    d["hot"] = d["in_period"] & (d["tmax"] > trig.threshold)
    d["season"] = d["date"].dt.year
    counts = np.zeros(len(d), dtype=bool)
    paid = pd.Series(False, index=d.index)
    for _, g in d.groupby(["cell", "season"]):
        idx = g.index.to_numpy()
        if trig.rule == "season_count":
            pays = g["hot"].sum() >= trig.n_days
            c = g["hot"].to_numpy() & pays
        elif trig.rule == "consecutive":
            # a calendar gap breaks a run
            full = g.set_index("date")["hot"].asfreq("D", fill_value=False)
            r = pd.Series(_runs_at_least(full.to_numpy(), trig.n_days), index=full.index)
            c = r.reindex(g["date"]).to_numpy()
            pays = bool(c.any())
        else:
            raise ValueError(trig.rule)
        counts[idx] = c
        paid.loc[idx] = pays
    d["counts"] = counts
    d["season_paid"] = paid
    return d


def audit(daily: pd.DataFrame, trig: Trigger) -> dict:
    """daily needs cell, date, tmax, unsafe_day, unsafe_hours (v2.exposure.daily_exposure). Descriptive
    exposure basis risk inside the trigger's risk period."""
    d = mark_trigger(daily, trig)
    p = d[d["in_period"]]
    unsafe = p[p["unsafe_day"]]
    missed = unsafe[~unsafe["counts"]]
    seasons = p.groupby(["cell", "season"]).agg(paid=("season_paid", "first"), unsafe_hours=("unsafe_hours", "sum"))
    median_uh = seasons["unsafe_hours"].median()
    return {
        "trigger": trig.name, "source": trig.source,
        "cell_days_in_period": int(len(p)), "unsafe_days": int(len(unsafe)),
        "share_unsafe_days_not_counted": float(len(missed) / len(unsafe)) if len(unsafe) else float("nan"),
        "unsafe_hours_on_missed_days": int(missed["unsafe_hours"].sum()),
        "share_counted_days_not_unsafe": float((p["counts"] & ~p["unsafe_day"]).sum() / max(1, p["counts"].sum())),
        "cell_seasons": int(len(seasons)), "share_seasons_paid": float(seasons["paid"].mean()),
        "share_seasons_unpaid_above_median_unsafe": float(((~seasons["paid"]) & (seasons["unsafe_hours"] > median_uh)).mean()),
    }


def audit_by_decade(daily: pd.DataFrame, trig: Trigger) -> pd.DataFrame:
    rows = []
    for dec, g in daily.groupby((daily["date"].dt.year // 10) * 10):
        rows.append({"decade": int(dec), **audit(g, trig)})
    return pd.DataFrame(rows)
