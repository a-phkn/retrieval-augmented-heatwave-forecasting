"""
Hourly heat-stress features from the raw Open-Meteo ERA5 JSON (plan v5, Phase 1A).

Target v2 is built from HOURLY data because humidity at the hour of peak heat is far
below the daily mean (Delhi, Jun 2019: RH 32% at the hour of Tmax vs 48% daily mean),
so WBGT/Heat Index computed from daily means misstate heat stress.

Formulas (T in deg C, RH in %):
  vapour pressure  e = RH/100 * 6.105 * exp(17.27 T / (237.7 + T))   [hPa]
  BoM WBGT approximation (Tier-A) = 0.567 T + 0.393 e + 3.94
      Australian Bureau of Meteorology index from T and humidity ONLY -- no radiation or
      wind. It is a temperature-humidity index, not a radiation-aware WBGT: in Delhi it
      exceeds air temperature in cool humid weather and runs ~6 C above a shade WBGT
      (0.7 Tw + 0.3 T) in July; ~80% of its long-term trend comes from the 0.393 e term.
      So: no absolute WBGT thresholds on it, call it "BoM WBGT approximation" in the paper,
      and validate against Tier-B (Liljegren, with radiation and wind from the v2 download).
  Wet-bulb temperature: Stull (2011) empirical formula -- independent humid-heat check.
  Heat Index: NWS algorithm -- simple formula below 80 F, otherwise the Rothfusz (1990)
      regression with the NWS low-humidity and high-humidity adjustments. Hourly values
      exceed the regression's fitted range on the hottest humid days (up to ~58 C); the
      extended heat index (Lu & Romps 2022, J. Appl. Meteor. Climatol.) would be the fix
      if HI becomes a primary target.
  Note: Open-Meteo returns RH as whole percent (+/-0.15 C noise in the BoM index at
      40 C); the v2 dew point allows a more precise humidity later.

Daily aggregation is on the LOCAL (IST) calendar day; the raw JSON is already in
Asia/Kolkata time. Domain aggregate (comparison target) = unweighted mean over the
9 cells of each cell's daily maximum -- the same construction process_weather.py uses
for t_max (max over hours first, then mean over cells; NOT WBGT of mean T and RH).

This module only reads data/raw/; it writes nothing. datasets_v2/ is built in Week 2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "era5_monthly"
N_CELLS = 9
END_DATE = pd.Timestamp("2026-09-06")  # frozen v1 end date (last complete day in v1)


# ---------------------------------------------------------------- formulas


def vapour_pressure_hpa(t_c, rh):
    """Actual vapour pressure (hPa) from air temperature (deg C) and RH (%)."""
    t_c, rh = np.asarray(t_c, dtype=np.float64), np.asarray(rh, dtype=np.float64)
    return rh / 100.0 * 6.105 * np.exp(17.27 * t_c / (237.7 + t_c))


def wbgt_bom(t_c, rh):
    """Australian BoM approximate WBGT (deg C) from T (deg C) and RH (%)."""
    return 0.567 * np.asarray(t_c, dtype=np.float64) + 0.393 * vapour_pressure_hpa(t_c, rh) + 3.94


def wet_bulb_stull(t_c, rh):
    """Wet-bulb temperature (deg C) from T (deg C) and RH (%) at standard pressure,
    Stull (2011, J. Appl. Meteor. Climatol. 50, 2267-2269). Valid for RH 5-99% and
    T -20..50 C (error mostly within +/-0.3 C); used here only as a cross-check."""
    t, rh = np.asarray(t_c, dtype=np.float64), np.asarray(rh, dtype=np.float64)
    return (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh) - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def heat_index_c(t_c, rh):
    """NWS Heat Index (deg C) from T (deg C) and RH (%), vectorised."""
    t_f = np.asarray(t_c, dtype=np.float64) * 9.0 / 5.0 + 32.0
    rh = np.asarray(rh, dtype=np.float64)
    t_f, rh = np.broadcast_arrays(t_f, rh)

    simple = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)
    rothfusz = (
        -42.379 + 2.04901523 * t_f + 10.14333127 * rh - 0.22475541 * t_f * rh
        - 0.00683783 * t_f**2 - 0.05481717 * rh**2 + 0.00122874 * t_f**2 * rh
        + 0.00085282 * t_f * rh**2 - 0.00000199 * t_f**2 * rh**2
    )
    with np.errstate(invalid="ignore"):
        low_rh = (rh < 13) & (t_f >= 80) & (t_f <= 112)
        adj_low = ((13 - rh) / 4.0) * np.sqrt(np.clip((17 - np.abs(t_f - 95.0)) / 17.0, 0, None))
    high_rh = (rh > 85) & (t_f >= 80) & (t_f <= 87)
    adj_high = ((rh - 85) / 10.0) * ((87 - t_f) / 5.0)
    rothfusz = rothfusz - np.where(low_rh, adj_low, 0.0) + np.where(high_rh, adj_high, 0.0)

    use_rothfusz = (simple + t_f) / 2.0 >= 80.0
    hi_f = np.where(use_rothfusz, rothfusz, simple)
    return (hi_f - 32.0) * 5.0 / 9.0


# ---------------------------------------------------------------- loading


def load_cell_hourly(cell: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """All hourly rows for one cell, sorted, de-duplicated, gap-checked.
    Columns: time (local IST, naive), t, rh, wind, pressure."""
    files = sorted((raw_dir / f"cell_{cell}").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no raw files for cell {cell} in {raw_dir}")
    frames = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            h = json.load(f)["hourly"]
        frames.append(pd.DataFrame({
            "time": h["time"], "t": h["temperature_2m"], "rh": h["relative_humidity_2m"],
            "wind": h["wind_speed_10m"], "pressure": h["surface_pressure"],
        }))
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    expected = pd.date_range(df["time"].iloc[0], df["time"].iloc[-1], freq="h")
    if len(expected) != len(df):
        raise ValueError(f"cell {cell}: {len(expected) - len(df)} missing hours between "
                         f"{df['time'].iloc[0]} and {df['time'].iloc[-1]}")
    return df


def daily_cell_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per local calendar day:
      t_max                 daily maximum air temperature
      wbgt_bom_max          max of the BoM T-humidity WBGT approximation (see module doc:
                            NOT a radiation-aware WBGT; runs ~6 C above a shade WBGT in July)
      wbgt_bom_max_hour     hour of that maximum (often late afternoon -- an artefact of a
                            formula without radiation)
      wbgt_bom_mean_12_18   mean of the BoM index 12:00-18:00
      tw_max                max Stull (2011) wet-bulb temperature: physically interpretable
                            humid-heat measure, used to check BoM-derived conclusions
      hi_max                max NWS Heat Index
      rh_at_tmax, e_at_tmax relative humidity (%) and vapour pressure (hPa) at the hour of Tmax
    Only complete days (24 hours with valid T and RH) are kept -- e.g. the raw files hold
    NaN for hours ERA5 had not yet published at download time (from 2026-09-07 05:00), the
    same reason v1 ends on 2026-09-06."""
    h = hourly.dropna(subset=["t", "rh"]).copy()
    h["date"] = h["time"].dt.normalize()
    h["hour"] = h["time"].dt.hour
    h["wbgt"] = wbgt_bom(h["t"], h["rh"])
    h["tw"] = wet_bulb_stull(h["t"], h["rh"])
    h["hi"] = heat_index_c(h["t"], h["rh"])
    h["e"] = vapour_pressure_hpa(h["t"], h["rh"])
    g = h.groupby("date")
    n_hours = g.size()
    i_tmax, i_wbgt = g["t"].idxmax(), g["wbgt"].idxmax()
    afternoon = h[(h["hour"] >= 12) & (h["hour"] <= 18)].groupby("date")["wbgt"].mean()
    out = pd.DataFrame({
        "t_max": g["t"].max(),
        "wbgt_bom_max": g["wbgt"].max(),
        "wbgt_bom_max_hour": h.loc[i_wbgt.values, "hour"].to_numpy(),
        "wbgt_bom_mean_12_18": afternoon,
        "tw_max": g["tw"].max(),
        "hi_max": g["hi"].max(),
        "rh_at_tmax": h.loc[i_tmax.values, "rh"].to_numpy(),
        "e_at_tmax": h.loc[i_tmax.values, "e"].to_numpy(),
    })
    return out[n_hours == 24]


def domain_daily(raw_dir: Path = RAW_DIR, end_date: pd.Timestamp = END_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-cell daily features with a 'cell' column, 9-cell domain mean per day),
    restricted to days complete in every cell and <= end_date. Note: the domain mean of
    wbgt_bom_max_hour averages clock hours across cells -- a diagnostic only, never a
    model feature."""
    per_cell = []
    for cell in range(1, N_CELLS + 1):
        d = daily_cell_features(load_cell_hourly(cell, raw_dir))
        d["cell"] = cell
        per_cell.append(d)
    cells = pd.concat(per_cell).reset_index()
    counts = cells.groupby("date")["cell"].nunique()
    complete = counts[counts == N_CELLS].index
    cells = cells[cells["date"].isin(complete) & (cells["date"] <= end_date)]
    domain = cells.drop(columns="cell").groupby("date").mean()
    return cells.reset_index(drop=True), domain
