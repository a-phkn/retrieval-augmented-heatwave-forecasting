"""
The rule engine: forecast -> alert colour per day -> which library actions apply (mapping B + D, decided 2026-10-22;
the exact rule written in context/decisions.md 2026-10-26 before this code ran).

Deterministic and model-free: the LLM never sets a colour. Input is the last OBSERVED_DAYS observed days and the
FORECAST_DAYS forecast days of Delhi's area-average Tmax with each date's normal (the training-years climatology).
  - heat-wave day: the labels-v2 rule (pipeline/labels_v2.py, unchanged): Tmax >= 40 C and >= 3.0 C above normal,
    or Tmax >= 45 C, inside Mar 15 - Jul 31; severe: a heat-wave day >= 4.5 C above normal;
  - run: consecutive heat-wave days, observed and forecast together; every day of a run gets the run's colour:
      RED     >= 3 consecutive severe days, or >= 7 days
      ORANGE  >= 2 consecutive severe days, or >= 4 days
      YELLOW  >= 2 days
      GREEN   otherwise (a single heat-wave day, or no run), as in IMD's colour table (IMD heat-wave FAQ p.8);
  - D: each forecast day also carries the model's hot-day probability;
  - humid heat: a separate note when the WBGT forecast reaches the WBGT label threshold (Mar 15 - Sep 30). It comes
    from our physical WBGT and is never an IMD category or colour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from advisory.library import IMD_STYLE_LABEL, Action, colour_rank
from pipeline.labels_v2 import ANOMALY_SEVERE_C, hot_days_tmax, in_wbgt_season

OBSERVED_DAYS = 7
FORECAST_DAYS = 5
HEAT_SEASON = ("03-15", "09-30")  # preparedness actions are shown in this window (Tmax and humid-heat seasons)


@dataclass(frozen=True)
class Day:
    date: pd.Timestamp
    lead: int  # 1..5
    tmax: float
    normal: float
    anomaly: float
    heatwave_day: bool
    severe_day: bool
    colour: str
    p_hot: float | None  # D: the hot-day probability for this day, if given
    wbgt: float | None
    humid_heat: bool


@dataclass(frozen=True)
class Assessment:
    days: tuple[Day, ...]
    headline_colour: str  # the highest colour over the 5 forecast days
    label: str
    in_heat_season: bool
    wbgt_threshold: float | None

    @property
    def any_humid_heat(self) -> bool:
        return any(d.humid_heat for d in self.days)


def classify(dates, tmax, normal) -> tuple[np.ndarray, np.ndarray]:
    """(heat-wave day, severe day) for each date, by the labels-v2 rule."""
    tmax, normal = np.asarray(tmax, dtype=np.float64), np.asarray(normal, dtype=np.float64)
    anomaly = tmax - normal
    hw = hot_days_tmax(dates, tmax, anomaly)
    return hw, hw & (anomaly >= ANOMALY_SEVERE_C)


def _longest_streak(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def run_colour(severe_in_run: np.ndarray) -> str:
    """The colour of one run of consecutive heat-wave days, from its days' severe flags."""
    n, s = len(severe_in_run), _longest_streak(np.asarray(severe_in_run, bool))
    if s >= 3 or n >= 7:
        return "RED"
    if s >= 2 or n >= 4:
        return "ORANGE"
    if n >= 2:
        return "YELLOW"
    return "GREEN"


def day_colours(hw: np.ndarray, severe: np.ndarray) -> list[str]:
    """Every day's colour: the colour of the run it belongs to, GREEN outside runs."""
    hw, severe = np.asarray(hw, bool), np.asarray(severe, bool)
    if (severe & ~hw).any():
        raise ValueError("a severe day must be a heat-wave day")
    out, i = ["GREEN"] * len(hw), 0
    while i < len(hw):
        if not hw[i]:
            i += 1
            continue
        j = i
        while j < len(hw) and hw[j]:
            j += 1
        c = run_colour(severe[i:j])
        out[i:j] = [c] * (j - i)
        i = j
    return out


def _check_frames(observed: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    for name, df, n in (("observed", observed, OBSERVED_DAYS), ("forecast", forecast, FORECAST_DAYS)):
        missing = {"date", "tmax", "normal"} - set(df.columns)
        if missing:
            raise ValueError(f"{name} needs columns {sorted(missing)}")
        if len(df) != n:
            raise ValueError(f"{name} must have exactly {n} days, got {len(df)}")
        if df[["tmax", "normal"]].isna().any().any():
            raise ValueError(f"{name} has missing values")
    both = pd.concat([observed[["date", "tmax", "normal"]], forecast[["date", "tmax", "normal"]]], ignore_index=True)
    both["date"] = pd.to_datetime(both["date"])
    if not (both["date"].diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("observed + forecast dates must be consecutive days, observed first")
    if "p_hot" in forecast and not forecast["p_hot"].dropna().between(0, 1).all():
        raise ValueError("p_hot must be a probability in [0, 1]")
    return both


def assess(observed: pd.DataFrame, forecast: pd.DataFrame, wbgt_threshold: float | None = None) -> Assessment:
    """Colours and notes for the 5 forecast days.

    observed: OBSERVED_DAYS rows (date, tmax, normal), the days just before the first forecast day.
    forecast: FORECAST_DAYS rows (date, tmax, normal) and optionally p_hot (D) and wbgt (the humid-heat note,
    used only with wbgt_threshold)."""
    both = _check_frames(observed, forecast)
    hw, severe = classify(both["date"], both["tmax"], both["normal"])
    colours = day_colours(hw, severe)
    f = forecast.reset_index(drop=True)
    days = []
    for k in range(FORECAST_DAYS):
        i = OBSERVED_DAYS + k
        date = both.at[i, "date"]
        wbgt = float(f.at[k, "wbgt"]) if "wbgt" in f and pd.notna(f.at[k, "wbgt"]) else None
        humid = bool(wbgt is not None and wbgt_threshold is not None and wbgt >= wbgt_threshold
                     and in_wbgt_season([date])[0])
        p = float(f.at[k, "p_hot"]) if "p_hot" in f and pd.notna(f.at[k, "p_hot"]) else None
        days.append(Day(date=date, lead=k + 1, tmax=float(both.at[i, "tmax"]), normal=float(both.at[i, "normal"]),
                        anomaly=float(both.at[i, "tmax"] - both.at[i, "normal"]), heatwave_day=bool(hw[i]),
                        severe_day=bool(severe[i]), colour=colours[i], p_hot=p, wbgt=wbgt, humid_heat=humid))
    headline = max((d.colour for d in days), key=colour_rank)
    md = [d.date.strftime("%m-%d") for d in days]
    return Assessment(days=tuple(days), headline_colour=headline, label=IMD_STYLE_LABEL,
                      in_heat_season=any(HEAT_SEASON[0] <= m <= HEAT_SEASON[1] for m in md),
                      wbgt_threshold=wbgt_threshold)


def forecast_actions(colour: str, library: dict[str, Action]) -> list[Action]:
    """Forecast-section actions unlocked at this colour (minimum colour at or below it); none on GREEN."""
    r = colour_rank(colour)
    return sorted((a for a in library.values() if a.section == "forecast" and colour_rank(a.min_colour) <= r),
                  key=lambda a: (-colour_rank(a.min_colour), a.audience, a.action_id))


def preparedness_actions(assessment: Assessment, library: dict[str, Action]) -> list[Action]:
    """Heat-season preparedness (decision 2026-10-26, option 1a): shown in the heat season whatever the colour."""
    if not assessment.in_heat_season:
        return []
    return sorted((a for a in library.values() if a.section == "preparedness"), key=lambda a: (a.audience, a.action_id))


def if_this_happens_actions(library: dict[str, Action]) -> tuple[list[Action], list[Action]]:
    """(for the public, for health professionals): always available, never mixed with the forecast actions."""
    acts = sorted((a for a in library.values() if a.section == "if_this_happens"), key=lambda a: a.action_id)
    return [a for a in acts if not a.for_professionals], [a for a in acts if a.for_professionals]
