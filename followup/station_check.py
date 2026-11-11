"""
P3-3 (pre-registered 2026-11-05, details fixed 2026-11-10 in context/decisions_phase3.md): the Safdarjung station
(GHCN-Daily IN022021900) as a second truth for Tmax. Post-test and descriptive: it changes no model and no
confirmatory result.

(i)   ERA5 9-cell Tmax vs the station, 1980-2018 and 2019-2025, and how often the two heatwave labels agree.
(ii)  The frozen test forecasts (C3-hop, the Delhi-only LSTM, ridge_hw5) vs the station.
(iii) The global models of P3-1 and GraphCast 2018 vs the station. GraphCast 2020 and the P3-2 variants are added
      once P3-2 has run.
(iv)  A per-month linear map station = a + b x ERA5, fitted on 1980-2018, applied to every forecast
      ("station-mapped") and to ERA5 itself.

- Station day: a TMAX value with no quality flag. No gap is filled; a day without a station value is left out.
- Station heatwave day: a single in-season day (15 Mar - 31 Jul) at or above min(45, max(40, normal + 3)) (the label
  rule) or min(45, max(40, normal + 4.5)) (IMD's station criterion). Normal = the station's own day-of-year
  climatology, 1980-2018.
- Every forecast row is scored against both truths on the same rows. Bias = forecast - truth; "ERA5 - station" is
  ERA5's own offset.
- The station file is not in the repository (sources/data/*.csv is gitignored); its SHA-256 is checked.
- Found after the registered results were seen (dating_check, exploratory): from 2011 a station value dated D
  behaves like the larger of the maxima of D - 1 and D, and from 2016 values are missing mostly on days warmer than
  the day before. The 2019-2025 station comparisons are therefore kept as registered but not interpreted.

Run from repo root, after python -m followup.global_models:  python -m followup.station_check
  -> followup/results/station_check.{md,json}
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from followup.global_models import EXTRACT, KEY, PRED_TEST, WARNING_JSON, _delta, _label, _num, align, restrict
from pipeline.climatology import apply_climatology, doy_climatology
from pipeline.labels_v2 import ANOMALY_HOT_C, ANOMALY_SEVERE_C, TMAX_ABSOLUTE_C, TMAX_FLOOR_C, in_season
from training.folds import FORECAST_DAYS

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "followup" / "results"
STATION_CSV = REPO_ROOT / "sources" / "data" / "safdarjung_daily_tidy.csv"
STATION_SHA256 = "a8fe98424ff2829c12601dbbe67de02fe07639cdf68a4ff9fdb4d78c24bbbe1b"  # pre-registered 2026-11-05
FIT_YEARS = (1980, 2018)  # the station normal and the station mapping use these years only
PERIODS = {"1980-2018": (1980, 2018), "2019-2025": (2019, 2025)}
RULES = {"label": ANOMALY_HOT_C, "imd": ANOMALY_SEVERE_C}  # margin above the normal, °C
RULE_NAMES = {"label": "label rule (normal + 3)", "imd": "IMD rule (normal + 4.5)"}
TEST_RUNS = {"C3-hop": "A1prime_hw5_C3hop", "Delhi-only LSTM": "A1prime_hw5", "ridge_hw5": "A1prime_hw5_ridge_hw5"}
PROB_RULES = {"C3-hop": "C3-hop R-prob", "Delhi-only LSTM": "Delhi-only LSTM R-prob"}  # keys of warning_rules.json
STRATA = {"all": "All days", "season": "Heat season (15 Mar - 31 Jul)",
          "st_label": "Station heatwave days, label rule", "st_imd": "Station heatwave days, IMD rule",
          "extreme": "ERA5 heatwave days (the test's extreme stratum)"}
VIEWS = {"era5": ("pred", "actual"), "station": ("pred", "station"), "station_mapped": ("mapped", "station")}
VIEW_NAMES = {"era5": "vs ERA5", "station": "vs station", "station_mapped": "station-mapped vs station"}
TEST_STRATA = ("all", "season", "st_label")
PENDING = ["GraphCast 2020 (P3-2a)", "the P3-2 corrector variants"]
# For the exploratory dating check only; the break years were read off that check's own per-year table.
DATING_BLOCKS = {"1980-2008": (1980, 2008), "2009-2010": (2009, 2010), "2011-2015": (2011, 2015),
                 "2016-2018": (2016, 2018), "2019-2025": (2019, 2025)}


# ------------------------------------------------------------------ the station


def load_station(path: Path = STATION_CSV, sha256: str | None = STATION_SHA256) -> pd.Series:
    """Usable daily TMAX (°C) by date: a value and no quality flag. The file must be the pre-registered one."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: the station file is not in the repository (see "
                                "sources/data/README_DATA.md)")
    if sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"{path.name} is not the pre-registered file (SHA-256 differs)")
    df = pd.read_csv(path, parse_dates=["date"])
    ok = df["tmax_c"].notna() & df["tmax_qflag"].isna()
    s = df.loc[ok].set_index("date")["tmax_c"].astype("float64").rename("station").sort_index()
    if s.index.has_duplicates:
        raise ValueError("the station file has duplicate dates")
    return s


def threshold(normal, dates, margin: float) -> np.ndarray:
    """Heatwave threshold per date: min(45, max(40, normal + margin)) in the heat season, +inf outside it. A day at
    or above it meets (Tmax >= 40 and departure >= margin) or Tmax >= 45."""
    thr = np.minimum(TMAX_ABSOLUTE_C, np.maximum(TMAX_FLOOR_C, np.asarray(normal, dtype=np.float64) + margin))
    return np.where(in_season(dates), thr, np.inf)


def station_frame(station: pd.Series) -> pd.DataFrame:
    """Per usable station day: the value, its normal (the station's own climatology of FIT_YEARS), each rule's
    threshold and heatwave flag, and the season flag."""
    years = station.index.year
    clim = doy_climatology(station, (years >= FIT_YEARS[0]) & (years <= FIT_YEARS[1]))
    normal, _ = apply_climatology(station.index, clim)
    out = pd.DataFrame({"station": station.to_numpy(), "st_normal": normal}, index=station.index.rename("date"))
    for rule, margin in RULES.items():
        out[f"thr_{rule}"] = threshold(normal, out.index, margin)
        out[f"st_{rule}"] = out["station"].to_numpy() >= out[f"thr_{rule}"].to_numpy()
    out["season"] = in_season(out.index)
    return out


def era5_daily() -> pd.DataFrame:
    """ERA5 9-cell Tmax from 1980 to the end of the test period, with the label rule's normal (1980-2018), threshold
    and heatwave flag. Reads the test period, so the caller checks the test lock first."""
    from training.folds import fold_daily

    d, _ = fold_daily("test", "t_max", "v2")
    out = pd.DataFrame({"era5": d["t_max"].astype("float64"), "era5_normal": d["clim_mean_t_max"],
                        "era5_hot": d["hot"].astype(bool)}, index=d.index.rename("date"))
    out["thr_era5"] = threshold(out["era5_normal"], out.index, ANOMALY_HOT_C)
    return out


# ------------------------------------------------------------------ (iv) the station mapping


def fit_mapping(era5: pd.Series, station: pd.Series, years: tuple[int, int] = FIT_YEARS) -> dict[int, dict]:
    """Per calendar month: a, b of station = a + b x ERA5 by least squares over the days of `years` that have both
    values, and the number of days n."""
    both = pd.concat([era5.rename("era5"), station.rename("station")], axis=1, join="inner").dropna()
    both = both[(both.index.year >= years[0]) & (both.index.year <= years[1])]
    out = {}
    for m, g in both.groupby(both.index.month):
        b, a = np.polyfit(g["era5"].to_numpy(), g["station"].to_numpy(), 1)
        out[int(m)] = {"a": float(a), "b": float(b), "n": int(len(g))}
    if sorted(out) != list(range(1, 13)):
        raise ValueError("the station mapping needs days in every calendar month")
    return out


def apply_mapping(values, dates, coef: dict[int, dict]) -> np.ndarray:
    """a + b x value, with the month of each date."""
    months = pd.DatetimeIndex(pd.to_datetime(dates)).month
    a = np.array([coef[int(m)]["a"] for m in months])
    b = np.array([coef[int(m)]["b"] for m in months])
    return a + b * np.asarray(values, dtype=np.float64)


# ------------------------------------------------------------------ forecast rows with both truths


def station_rows(df: pd.DataFrame, st: pd.DataFrame, thr_era5: pd.Series, coef: dict[int, dict]) -> pd.DataFrame:
    """The forecast rows (any number of seeds) whose target day has a station value, in their original order, with
    the station truth, its thresholds and flags, the ERA5 label threshold and the station-mapped forecast. `actual`
    stays the ERA5 truth."""
    t = pd.DatetimeIndex(df["target_date"])
    s = st.reindex(t)
    keep = s["station"].notna().to_numpy()
    out, s = df[keep].reset_index(drop=True), s[keep].reset_index(drop=True)
    thr = thr_era5.reindex(t[keep]).to_numpy(dtype=np.float64)
    if np.isnan(thr).any():
        raise ValueError("a forecast row has no ERA5 label threshold")
    out = out.assign(station=s["station"].to_numpy(), thr_era5=thr, season=s["season"].to_numpy(bool),
                     mapped=apply_mapping(out["pred"], out["target_date"], coef))
    for rule in RULES:
        out[f"thr_{rule}"] = s[f"thr_{rule}"].to_numpy()
        out[f"st_{rule}"] = s[f"st_{rule}"].to_numpy(bool)
    return out


def strata_masks(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"all": np.ones(len(rows), dtype=bool), "season": rows["season"].to_numpy(bool),
            "st_label": rows["st_label"].to_numpy(bool), "st_imd": rows["st_imd"].to_numpy(bool),
            "extreme": (rows["stratum"] == "extreme").to_numpy()}


# ------------------------------------------------------------------ scores


def summarise(rows: pd.DataFrame, pred: str, truth: str) -> dict:
    """RMSE (mean over seeds of the per-seed RMSE; the plain RMSE for one forecast) and bias of pred - truth, per
    stratum; and by lead on all days and on station heatwave days (label rule)."""
    from evaluation.compare_v2 import rmse_by

    r = rows.assign(error=rows[pred].to_numpy() - rows[truth].to_numpy())
    masks = strata_masks(r)

    def one(v: pd.DataFrame) -> dict:
        n = int(len(v) / max(1, v["seed"].nunique()))
        return {"n": n, "clusters": int(v["cluster_id"].nunique()),
                "rmse": rmse_by(v) if n else float("nan"),
                "bias": float(v.groupby("seed")["error"].mean().mean()) if n else float("nan")}

    out = {name: one(r[m]) for name, m in masks.items()}
    for name in ("all", "st_label"):
        out[f"by_lead_{name}"] = [one(r[masks[name] & (r["lead"] == L).to_numpy()]) for L in range(1, FORECAST_DAYS + 1)]
    return out


def three_views(rows: pd.DataFrame) -> dict:
    return {view: summarise(rows, pred, truth) for view, (pred, truth) in VIEWS.items()}


def paired(parent: pd.DataFrame, child: pd.DataFrame) -> dict:
    """{view: {stratum: test}}: the paired cluster jackknife of RMSE(child) - RMSE(parent) against each truth, on the
    same rows. A stratum with too few rows or clusters gives {"error": ...}; fewer than 5 clusters give no CI."""
    from evaluation.compare_v2 import compare

    out = {}
    for view, (pred, truth) in VIEWS.items():
        out[view] = {}
        for stratum in TEST_STRATA:
            a, b = (x[strata_masks(x)[stratum]].reset_index(drop=True) for x in (parent, child))
            a, b = (x.assign(error=x[pred].to_numpy() - x[truth].to_numpy()) for x in (a, b))
            try:
                out[view][stratum] = compare(a, b, "all")
            except (ValueError, ZeroDivisionError) as err:
                out[view][stratum] = {"error": str(err)}
    return out


def clear_change(under_era5: dict, under_station: dict) -> bool:
    """The pre-registered reading rule: the order of two models changes clearly only if both paired CIs exclude 0,
    with opposite signs under the two truths."""
    def sign(t: dict) -> int:
        if "error" in t or not np.isfinite(t["ci_low"]):
            return 0
        return -1 if t["ci_high"] < 0 else (1 if t["ci_low"] > 0 else 0)

    return sign(under_era5) * sign(under_station) == -1


def warnings_table(rows: pd.DataFrame, probs: pd.DataFrame | None = None, p_star: float | None = None) -> dict:
    """Yes/no warnings of one single forecast on its in-season rows. Events: station heatwave days by each rule, and
    ERA5 heatwave days for reference. Raw forecast: R-point with the ERA5 label threshold, R-prob if probabilities
    are given. Station-mapped forecast: R-point with the station's own threshold of that rule."""
    from post_test.warning_rules import scores

    if rows["seed"].nunique() > 1:
        raise ValueError("warnings are scored for a single forecast (the ensemble mean)")
    r = rows[rows["season"].to_numpy(bool)].reset_index(drop=True)
    if probs is not None:
        r = r.merge(probs[KEY + ["p"]], on=KEY, how="left", validate="one_to_one")
        if r["p"].isna().any():
            raise ValueError("an in-season row has no probability")
    point = r["pred"].to_numpy() >= r["thr_era5"].to_numpy()
    out = {"rows": int(len(r)), "events": {}}
    for rule in RULES:
        o = r[f"st_{rule}"].to_numpy(bool)
        rules = {"R-point": scores(o, point)}
        if probs is not None:
            rules[f"R-prob (p* = {p_star})"] = scores(o, r["p"].to_numpy() >= p_star)
        rules["station-mapped R-point"] = scores(o, r["mapped"].to_numpy() >= r[f"thr_{rule}"].to_numpy())
        out["events"][f"station, {RULE_NAMES[rule]}"] = {"n": int(o.sum()), "rules": rules}
    o = r["actual"].to_numpy() >= r["thr_era5"].to_numpy()
    rules = {"R-point": scores(o, point)}
    if probs is not None:
        rules[f"R-prob (p* = {p_star})"] = scores(o, r["p"].to_numpy() >= p_star)
    out["events"]["ERA5, label rule (reference)"] = {"n": int(o.sum()), "rules": rules}
    return out


def cold_bias(rows: pd.DataFrame) -> dict:
    """On station heatwave days (each rule), over all leads and per lead: forecast - station = (forecast - ERA5) +
    (ERA5 - station), as means, and ERA5's share (ERA5 - station) / (forecast - station). For a single forecast."""
    def parts(v: pd.DataFrame) -> dict:
        if v.empty:
            return {"n": 0}
        fs = float((v["pred"] - v["station"]).mean())
        fe = float((v["pred"] - v["actual"]).mean())
        es = float((v["actual"] - v["station"]).mean())
        return {"n": int(len(v)), "forecast_minus_station": fs, "forecast_minus_era5": fe, "era5_minus_station": es,
                "era5_share": es / fs if fs else float("nan")}

    out = {}
    for rule in RULES:
        v = rows[rows[f"st_{rule}"].to_numpy(bool)]
        out[rule] = {"all_leads": parts(v), "by_lead": [parts(v[v["lead"] == L]) for L in range(1, FORECAST_DAYS + 1)]}
    return out


# ------------------------------------------------------------------ (i) ERA5 vs the station


def _rmse(e: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(e.to_numpy())))) if len(e) else float("nan")


def era5_vs_station(era5: pd.DataFrame, st: pd.DataFrame, coef: dict[int, dict]) -> dict:
    """Per period: ERA5 - station (bias, RMSE), raw and after the station mapping, on all days, the heat season,
    station heatwave days (each rule) and, as the mirror view, ERA5 heatwave days. And the agreement of the labels
    on in-season days: events = station heatwave days, 'warnings' = ERA5 heatwave days (the ERA5 label, and the
    mapped ERA5 value against the station's threshold)."""
    from post_test.warning_rules import scores

    d = st.join(era5, how="inner")
    d["mapped"] = apply_mapping(d["era5"], d.index, coef)
    out = {}
    for name, (first, last) in PERIODS.items():
        p = d[(d.index.year >= first) & (d.index.year <= last)]
        masks = {"all": np.ones(len(p), dtype=bool), "season": p["season"].to_numpy(bool),
                 "st_label": p["st_label"].to_numpy(bool), "st_imd": p["st_imd"].to_numpy(bool),
                 "era5_hot": p["era5_hot"].to_numpy(bool)}
        strata = {}
        for k, m in masks.items():
            v = p[m]
            raw, mapped = v["era5"] - v["station"], v["mapped"] - v["station"]
            strata[k] = {"n": int(len(v)), "bias": float(raw.mean()) if len(v) else float("nan"), "rmse": _rmse(raw),
                         "mapped_bias": float(mapped.mean()) if len(v) else float("nan"), "mapped_rmse": _rmse(mapped)}
        s = p[masks["season"]]
        labels = {}
        for rule in RULES:
            o = s[f"st_{rule}"].to_numpy(bool)
            labels[rule] = {"era5_label": scores(o, s["era5_hot"].to_numpy(bool)),
                            "mapped_era5": scores(o, s["mapped"].to_numpy() >= s[f"thr_{rule}"].to_numpy())}
        out[name] = {"days": int(len(p)), "in_season_days": int(len(s)),
                     "first": str(p.index.min().date()) if len(p) else None,
                     "last": str(p.index.max().date()) if len(p) else None, "strata": strata, "labels": labels}
    return out


# ------------------------------------------------------------------ exploratory: are the station's dates right?


def dating_check(era5: pd.Series, station: pd.Series) -> dict:
    """Exploratory, added 2026-11-10 AFTER the registered results were seen; it uses no forecast. Per block of years
    (DATING_BLOCKS, chosen from what this check itself showed) and per year:
    - coverage: the share of calendar days with a usable station value;
    - how the station value dated D matches ERA5 of D ('same'), ERA5 of D - 1 ('prev') and the larger of the two
      ('both'): mean and SD of ERA5 - station over the reporting days;
    - the share of reporting days on which ERA5 was cooler than the day before, next to that share over all days
      (the two agree if the missing days are random);
    - from the station file alone: the share of consecutive reporting days with exactly the same value."""
    e = era5[(era5.index.year >= FIT_YEARS[0]) & (era5.index <= station.index.max())]
    f = pd.DataFrame({"same": e, "prev": e.shift(1), "st": station.reindex(e.index)})
    f["both"] = np.maximum(f["same"], f["prev"])
    f["st_prev"] = f["st"].shift(1)

    def one(g: pd.DataFrame) -> dict:
        o, pairs = g.dropna(subset=["st", "prev"]), g.dropna(subset=["st", "st_prev"])
        out = {"days": int(len(g)), "coverage": float(g["st"].notna().mean()), "reporting_days": int(len(o)),
               "cooler_share_reporting": float((o["same"] < o["prev"]).mean()) if len(o) else float("nan"),
               "cooler_share_all": float((g["same"] < g["prev"]).mean()),
               "consecutive_pairs": int(len(pairs)),
               "identical_share": float((pairs["st"] == pairs["st_prev"]).mean()) if len(pairs) else float("nan")}
        for k in ("same", "prev", "both"):
            diff = o[k] - o["st"]
            out[k] = {"offset": float(diff.mean()) if len(o) else float("nan"),
                      "sd": float(diff.std(ddof=1)) if len(o) > 1 else float("nan")}
        return out

    return {"blocks": {name: one(f[(f.index.year >= a) & (f.index.year <= b)]) for name, (a, b) in DATING_BLOCKS.items()},
            "by_year": {int(y): one(g) for y, g in f.groupby(f.index.year)}}


# ------------------------------------------------------------------ (ii) the frozen test forecasts


def _period(rows: pd.DataFrame) -> dict:
    return {"rows": int(len(rows) / max(1, rows["seed"].nunique())), "clusters": int(rows["cluster_id"].nunique()),
            "first_target": str(rows["target_date"].min().date()), "last_target": str(rows["target_date"].max().date())}


def score_forecasts(seeds: dict[str, pd.DataFrame], probs: dict[str, tuple[pd.DataFrame, float]], st: pd.DataFrame,
                    thr_era5: pd.Series, coef: dict[int, dict]) -> dict:
    """(ii) for the forecasts of TEST_RUNS given per seed. probs: {name: (probability rows, p*)} for the models that
    have a hot-day probability. The networks are compared seed by seed (the test's way); ridge_hw5, a single
    forecast, is compared with C3-hop's ensemble mean."""
    from evaluation.compare_v2 import ensemble

    rows = {name: station_rows(df.sort_values(KEY + ["seed"]), st, thr_era5, coef) for name, df in seeds.items()}
    ens = {name: station_rows(ensemble(df), st, thr_era5, coef) for name, df in seeds.items()}
    models = {}
    for name in seeds:
        p, p_star = probs.get(name, (None, None))
        n_seeds = int(rows[name]["seed"].nunique())
        models[name] = {**_period(ens[name]), "seeds": n_seeds, "ensemble": three_views(ens[name]),
                        "per_seed": three_views(rows[name]) if n_seeds > 1 else None,
                        "warnings": warnings_table(ens[name], p, p_star), "cold_bias": cold_bias(ens[name])}
    n = models["C3-hop"]["seeds"]
    tests = {f"C3-hop minus Delhi-only LSTM ({n} seeds each)": paired(rows["Delhi-only LSTM"], rows["C3-hop"]),
             "C3-hop ensemble minus ridge_hw5": paired(ens["ridge_hw5"], ens["C3-hop"])}
    return {"models": models, "tests": tests}


def frozen_forecasts(st: pd.DataFrame, era5: pd.DataFrame, coef: dict[int, dict]) -> dict:
    """(ii) on the committed test-fold forecasts, with the hot-day probabilities and p* of post-test 3."""
    from post_test.warning_rules import prob_rows

    choices = json.loads(WARNING_JSON.read_text(encoding="utf-8"))["Tmax"]["choices"]
    seeds = {name: pd.read_parquet(PRED_TEST / rid / "test.parquet") for name, rid in TEST_RUNS.items()}
    probs = {name: (prob_rows(TEST_RUNS[name], "Tmax", "test"), choices[key]["p_star"]) for name, key in PROB_RULES.items()}
    return score_forecasts(seeds, probs, st, era5["thr_era5"], coef)


# ------------------------------------------------------------------ (iii) global models and GraphCast 2018


def versus_c3hop(name: str, by_lead: dict[int, pd.Series], c3e: pd.DataFrame, st: pd.DataFrame, thr_era5: pd.Series,
                 coef: dict[int, dict]) -> dict | None:
    """One outside forecast against C3-hop's ensemble mean on the rows both have and the station observed."""
    g = station_rows(align(c3e, by_lead, name), st, thr_era5, coef)
    if g.empty:
        return None
    c3 = station_rows(restrict(c3e, g), st, thr_era5, coef)
    return {**_period(g), "model": three_views(g), "c3hop": three_views(c3), "test": paired(c3, g),
            "warnings": {"model": warnings_table(g), "c3hop": warnings_table(c3)},
            "cold_bias": {"model": cold_bias(g), "c3hop": cold_bias(c3)}}


def global_models(st: pd.DataFrame, era5: pd.DataFrame, coef: dict[int, dict]) -> dict:
    """Every model of the P3-1 daily extract (Tmax) against C3-hop, test period."""
    from evaluation.compare_v2 import ensemble

    ex = pd.read_parquet(EXTRACT)
    ex = ex[ex["target"] == "tmax"]
    c3e = ensemble(pd.read_parquet(PRED_TEST / TEST_RUNS["C3-hop"] / "test.parquet"))
    out = {}
    for name, g in ex.groupby("model", sort=False):
        by_lead = {int(L): s.set_index("date")["value"] for L, s in g.groupby("lead")}
        r = versus_c3hop(name, by_lead, c3e, st, era5["thr_era5"], coef)
        if r is not None:
            out[name] = r
    return out


def graphcast_2018(st: pd.DataFrame, coef: dict[int, dict]) -> dict | None:
    """GraphCast 2018 (the yardstick's extract and max-only correction) against C3-hop's development forecasts on
    fold f4's 2018 windows. None if the extract or the raw hourly ERA5 it needs is not on disk."""
    from evaluation import graphcast_yardstick as gy
    from evaluation.compare_v2 import ensemble, load_run
    from training.folds import fold_daily

    if not gy.EXTRACT.exists():
        return None
    val = gy.windows_2018()
    try:
        correction = gy.fit_correction(gy.era5_daily_true_and_sampled())
    except FileNotFoundError:
        return None
    gc = gy.graphcast_frame(val, gy.fetch_graphcast(pd.DatetimeIndex(val.query_dates)), correction)
    by_lead = {int(L): g.set_index("target_date")["pred"] for L, g in gc.groupby("lead")}
    d, _ = fold_daily("f4", "t_max", "v2")
    thr = pd.Series(threshold(d["clim_mean_t_max"], d.index, ANOMALY_HOT_C), index=d.index)
    c3 = load_run(TEST_RUNS["C3-hop"], folds=("f4",))
    if c3 is None:
        return None
    c3 = c3[c3["query_date"].isin(gc["query_date"].unique())].drop(columns="fold")
    return versus_c3hop("GraphCast 2018", by_lead, ensemble(c3), st, thr, coef)


# ------------------------------------------------------------------ report


def _pct(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def _score_row(s: dict) -> str:
    return (f"{s['hits']} | {s['misses']} | {s['false_alarms']} | {_pct(s['hit_rate'])} | "
            f"{_pct(s['false_alarm_ratio'])} | {_num(s['ets'], '.2f')} | {_num(s['frequency_bias'], '.2f')}")


def _coverage(st: pd.DataFrame) -> dict:
    out = {}
    for name, (first, last) in PERIODS.items():
        p = st[(st.index.year >= first) & (st.index.year <= last)]
        out[name] = {"days": int(len(p)), "in_season_days": int(p["season"].sum()),
                     "heatwave_days_label": int(p["st_label"].sum()), "heatwave_days_imd": int(p["st_imd"].sum()),
                     "last_day": str(p.index.max().date()) if len(p) else None}
    return out


def _views_table(L: list[str], entries: list[tuple[str, dict]]) -> None:
    """One table per stratum: each entry's rows, RMSE and bias under the three views."""
    for stratum, title in STRATA.items():
        L += [f"**{title}**", "", "| Forecast | Rows | RMSE vs ERA5 | RMSE vs station | RMSE station-mapped | "
              "Bias vs ERA5 | Bias vs station | Bias station-mapped |", "|---|---|---|---|---|---|---|---|"]
        for name, v in entries:
            e, s, m = (v[k][stratum] for k in VIEWS)
            L.append(f"| {name} | {s['n']} | {_num(e['rmse'])} | {_num(s['rmse'])} | {_num(m['rmse'])} | "
                     f"{_num(e['bias'], '+.2f')} | {_num(s['bias'], '+.2f')} | {_num(m['bias'], '+.2f')} |")
        L.append("")


def _lead_table(L: list[str], entries: list[tuple[str, dict]], key: str, title: str) -> None:
    L += [f"**{title}**", "", "| Forecast | Score | Day 1 | 2 | 3 | 4 | 5 |", "|---|---|---|---|---|---|---|"]
    for name, v in entries:
        for view in ("station", "station_mapped"):
            leads = v[view][key]
            L.append(f"| {name} | RMSE {VIEW_NAMES[view]} | " + " | ".join(_num(x["rmse"]) for x in leads) + " |")
            L.append(f"| {name} | Bias {VIEW_NAMES[view]} | " + " | ".join(_num(x["bias"], "+.2f") for x in leads) + " |")
    L.append("")


def _tests_table(L: list[str], tests: dict[str, dict]) -> None:
    L += ["| Comparison | Stratum | Delta vs ERA5 [95% CI] | Delta vs station [95% CI] | "
          "Delta station-mapped [95% CI] | Clear change of order |", "|---|---|---|---|---|---|"]
    for name, t in tests.items():
        for stratum in TEST_STRATA:
            e, s, m = (t[view][stratum] for view in VIEWS)
            L.append(f"| {name} | {STRATA[stratum]} | {_delta(e)} | {_delta(s)} | {_delta(m)} | "
                     f"{'yes' if clear_change(e, s) else 'no'} |")
    L.append("")


def _warnings_table(L: list[str], entries: list[tuple[str, dict]]) -> None:
    L += ["| Forecast | Events | Rule | Hits | Misses | False alarms | Hit rate | False-alarm ratio | ETS | "
          "Frequency bias |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, w in entries:
        for events, e in w["events"].items():
            for rule, s in e["rules"].items():
                L.append(f"| {name} ({w['rows']} rows) | {events} ({e['n']}) | {rule} | {_score_row(s)} |")
    L.append("")


def _cold_table(L: list[str], entries: list[tuple[str, dict]]) -> None:
    L += ["| Forecast | Station rule | Lead | Rows | Forecast - station | = forecast - ERA5 | + ERA5 - station | "
          "ERA5's share |", "|---|---|---|---|---|---|---|---|"]
    for name, c in entries:
        for rule in RULES:
            parts = [("all", c[rule]["all_leads"])] + [(str(i), p) for i, p in enumerate(c[rule]["by_lead"], 1)]
            for lead, p in parts:
                if not p["n"]:
                    L.append(f"| {name} | {RULE_NAMES[rule]} | {lead} | 0 | | | | |")
                    continue
                L.append(f"| {name} | {RULE_NAMES[rule]} | {lead} | {p['n']} | {_num(p['forecast_minus_station'], '+.2f')} | "
                         f"{_num(p['forecast_minus_era5'], '+.2f')} | {_num(p['era5_minus_station'], '+.2f')} | "
                         f"{_pct(p['era5_share'])} |")
    L.append("")


def write_report(res: dict) -> None:
    cov = res["station"]["coverage"]
    blocks = res["dating"]["blocks"]
    early, recent = blocks["1980-2008"], blocks["2019-2025"]
    L = ["# The Safdarjung station as a second truth for Tmax (P3-3, post-test)", "",
         "Pre-registered 2026-11-05 in `context/decisions_phase3.md` (P3-3), details fixed 2026-11-10 before any "
         "comparison was run. Descriptive: the test years were already used, so nothing here is confirmatory. "
         "Station: GHCN-Daily IN022021900 (Safdarjung), daily maximum, °C. ERA5: the 9-cell mean the models were "
         "trained and tested on. Bias = forecast - truth; 'ERA5 - station' is ERA5's own offset. Station-mapped = "
         "the forecast put through a per-month straight line fitted on 1980-2018 (station = a + b x ERA5).", "",
         "## Read this first: the station's recent values are not dated to the day they occurred", "",
         "Found after the registered results were seen (the check is the section after 'Station data'). Up to about "
         "2008 a station value dated D is the maximum of day D. From 2011 it behaves like the larger of the maxima "
         f"of D - 1 and D. From 2016 values are also missing on many days ({_pct(1 - recent['coverage'])} of days in "
         "2019-2025), mostly days warmer than the day before, so most remaining values are the previous day's "
         "maximum. Consequences:", "",
         f"- In 2019-2025, {_pct(recent['cooler_share_reporting'])} of the station's values fall on a day cooler than "
         "the day before. On those days a forecast for day D is compared with the maximum of day D - 1.",
         "- So the 2019-2025 station numbers below (the second period of (i), all of (ii), and (iii)) do **not** "
         "measure error against the station's true daily maximum. They are kept exactly as registered, for the "
         "record, and no conclusion is drawn from them.",
         f"- The {abs(recent['same']['offset']):.1f} °C gap between ERA5 and the station in 2019-2025 is partly this "
         f"dating effect. Matched to the larger of the two days, the gap is {abs(recent['both']['offset']):.1f} °C. It "
         f"was {abs(early['same']['offset']):.1f} °C in 1980-2008, matched to the same day.",
         "- The station normal and the mapping are fitted on 1980-2018, which includes the affected years 2011-2018.", "",
         "## Station data", "",
         "| Period | Usable days | In the heat season | Heatwave days, label rule | Heatwave days, IMD rule | Last day |",
         "|---|---|---|---|---|---|"]
    for name, c in cov.items():
        L.append(f"| {name} | {c['days']} | {c['in_season_days']} | {c['heatwave_days_label']} | "
                 f"{c['heatwave_days_imd']} | {c['last_day']} |")
    L += ["", "A usable day has a value and no quality flag. Gaps are not filled; a day without a station value is left "
          "out of every comparison. A station heatwave day is a single in-season day at or above min(45, max(40, "
          "normal + 3)) (label rule) or min(45, max(40, normal + 4.5)) (IMD rule), with the station's own 1980-2018 "
          "normal.", "",
          "## Exploratory: are the station's dates right? (added after the results were seen)", "",
          "Not registered. It uses the station and ERA5 only, no forecast; the blocks of years were read off its own "
          "per-year table (in the JSON). For the station value dated D: the mean and SD of ERA5 - station when ERA5 "
          "is taken on the same day D, on the day before, or as the larger of the two. The smallest SD shows which "
          "day the station value belongs to.", "",
          "| Years | Days with a value | Same day: mean / SD | Day before: mean / SD | Larger of the two: mean / SD | "
          "Reporting days cooler than the day before (all days) | Consecutive values that are identical |",
          "|---|---|---|---|---|---|---|"]
    for name, b in res["dating"]["blocks"].items():
        L.append(f"| {name} | {_pct(b['coverage'])} | "
                 + " | ".join(f"{_num(b[k]['offset'], '+.2f')} / {_num(b[k]['sd'], '.2f')}" for k in ("same", "prev", "both"))
                 + f" | {_pct(b['cooler_share_reporting'])} ({_pct(b['cooler_share_all'])}) | {_pct(b['identical_share'])} |")
    L += ["", "How to read it:", "",
          "- If the dates are right, the same day has the smallest SD, about as many reporting days are cooler than "
          "the day before as in the calendar as a whole, and identical consecutive values are rare. That is the "
          "picture in 1980-2008.",
          "- In 2011-2015 the larger of the two days fits best and identical consecutive values are several times "
          "more common: a hot day's maximum is repeated on the next, cooler day.",
          "- From 2016 the value is present mainly on days cooler than the day before, and the day before fits far "
          "better than the same day.",
          "- GHCN-Daily marks all of this station's values with source flag S (Global Summary of the Day). Its "
          "documentation warns that S values are derived from synoptic reports and may differ from true daily data.", ""]
    L += ["## (i) ERA5 against the station", "",
          "| Period | Days | ERA5 - station | RMSE | After the mapping: bias | RMSE |", "|---|---|---|---|---|---|"]
    names = {"all": "all days", "season": "heat season", "st_label": "station heatwave days, label rule",
             "st_imd": "station heatwave days, IMD rule", "era5_hot": "ERA5 heatwave days (mirror view)"}
    for period, r in res["era5_vs_station"].items():
        for k, title in names.items():
            s = r["strata"][k]
            L.append(f"| {period}, {title} | {s['n']} | {_num(s['bias'], '+.2f')} | {_num(s['rmse'])} | "
                     f"{_num(s['mapped_bias'], '+.2f')} | {_num(s['mapped_rmse'])} |")
    L += ["", "The mapping is fitted on 1980-2018, so its 1980-2018 rows are in-sample. Picking days by one record's "
          "heat makes the other record look colder on those days (selection), so the station-heatwave rows and the "
          "mirror row are read together.", "",
          "**Do the heatwave labels agree?** In-season days with a station value. Events = station heatwave days; "
          "'warnings' = ERA5 heatwave days.", "",
          "| Period | Station rule | ERA5 side | Both | Station only | ERA5 only | Station days ERA5 also has | "
          "ERA5 days the station lacks | ETS | ERA5 days / station days |", "|---|---|---|---|---|---|---|---|---|---|"]
    for period, r in res["era5_vs_station"].items():
        for rule in RULES:
            for side, title in (("era5_label", "ERA5 label"), ("mapped_era5", "mapped ERA5 vs the station threshold")):
                L.append(f"| {period} ({r['in_season_days']} days) | {RULE_NAMES[rule]} | {title} | "
                         f"{_score_row(r['labels'][rule][side])} |")
    L += ["", "## (iv) The station mapping", "", "station = a + b x ERA5, least squares per calendar month on the "
          "usable station days of 1980-2018.", "", "| Month | a | b | Days |", "|---|---|---|---|"]
    for m, c in res["mapping"].items():
        L.append(f"| {m} | {c['a']:+.2f} | {c['b']:.3f} | {c['n']} |")

    tf = res["test_forecasts"]
    first = next(iter(tf["models"].values()))
    L += ["", "## (ii) Our test forecasts against the station", "",
          f"Target days {first['first_target']} to {first['last_target']} with a station value: {first['rows']} rows "
          f"(query date x lead), {first['clusters']} year x season clusters. The networks are shown as the 10-seed "
          "ensemble mean and as the mean of the 10 seeds' RMSE (the test's primary score).", ""]
    entries = []
    for name, m in tf["models"].items():
        entries.append((f"{name}{' (ensemble mean)' if m['per_seed'] else ''}", m["ensemble"]))
        if m["per_seed"]:
            entries.append((f"{name} (mean of {m['seeds']} seeds)", m["per_seed"]))
    _views_table(L, entries)
    ens_entries = [(name, m["ensemble"]) for name, m in tf["models"].items()]
    _lead_table(L, ens_entries, "by_lead_all", "By lead, all days (ensemble means)")
    _lead_table(L, ens_entries, "by_lead_st_label", "By lead, station heatwave days, label rule (ensemble means)")
    L += ["**Paired tests** (delta = RMSE of the first minus RMSE of the second, same rows; negative = the first is "
          "better; paired cluster jackknife).", ""]
    _tests_table(L, tf["tests"])
    L += ["**Yes/no warnings** (in-season rows). R-point and R-prob use the forecast as issued, against the ERA5 "
          "label threshold. Station-mapped R-point uses the mapped forecast against the station's own threshold of "
          "that rule.", ""]
    _warnings_table(L, [(name, m["warnings"]) for name, m in tf["models"].items()])
    L += ["**How much of the cold bias on station heatwave days is ERA5's own?** Forecast - station = (forecast - "
          "ERA5) + (ERA5 - station), means over the rows; ensemble means.", ""]
    _cold_table(L, [(name, m["cold_bias"]) for name, m in tf["models"].items()])

    L += ["## (iii) Global models and GraphCast 2018 against the station", "",
          "Each outside forecast is paired with C3-hop's ensemble mean on the rows both have and the station "
          "observed. The global models are scored on the test period (to 2025-08-23, where the station file ends); "
          "GraphCast 2018 on fold f4's 2018 windows with C3-hop's development forecasts (2018 lies inside the "
          "mapping's fitting years).", ""]
    outside = dict(res["global_models"])
    if res["graphcast_2018"]:
        outside["GraphCast 2018"] = res["graphcast_2018"]
    for stratum, title in STRATA.items():
        L += [f"**{title}**", "", "| Forecast | Target days | Rows | Clusters | RMSE vs ERA5: it / C3-hop | "
              "RMSE vs station: it / C3-hop | Station-mapped: it / C3-hop | Bias vs station: it / C3-hop |",
              "|---|---|---|---|---|---|---|---|"]
        for name, r in outside.items():
            g, c = r["model"], r["c3hop"]
            L.append(f"| {_label(name)} | {r['first_target']} to {r['last_target']} | {g['station'][stratum]['n']} | "
                     f"{g['station'][stratum]['clusters']} | "
                     + " | ".join(f"{_num(g[v][stratum]['rmse'])} / {_num(c[v][stratum]['rmse'])}" for v in VIEWS)
                     + f" | {_num(g['station'][stratum]['bias'], '+.2f')} / {_num(c['station'][stratum]['bias'], '+.2f')} |")
        L.append("")
    L += ["**Paired tests** (delta = RMSE of the outside forecast minus RMSE of C3-hop; negative = the outside "
          "forecast is better).", ""]
    _tests_table(L, {f"{_label(name)} minus C3-hop": r["test"] for name, r in outside.items()})
    L += ["**By lead, all days**", ""]
    L += ["| Forecast | Score | Day 1 | 2 | 3 | 4 | 5 |", "|---|---|---|---|---|---|---|"]
    for name, r in outside.items():
        for who, k in ((_label(name), "model"), (f"C3-hop on {name}'s rows", "c3hop")):
            L.append(f"| {who} | RMSE vs station | " + " | ".join(_num(x["rmse"]) for x in r[k]["station"]["by_lead_all"]) + " |")
    L += ["", "**Yes/no warnings** (in-season rows; R-point only, no probabilities for these comparisons).", ""]
    wl = []
    for name, r in outside.items():
        wl += [(_label(name), r["warnings"]["model"]), (f"C3-hop on {name}'s rows", r["warnings"]["c3hop"])]
    _warnings_table(L, wl)
    L += ["**Cold bias on station heatwave days**", ""]
    cl = []
    for name, r in outside.items():
        cl += [(_label(name), r["cold_bias"]["model"]), (f"C3-hop on {name}'s rows", r["cold_bias"]["c3hop"])]
    _cold_table(L, cl)
    L += ["## Notes", "",
          f"- Station file: `{STATION_CSV.relative_to(REPO_ROOT).as_posix()}` (not in the repository), SHA-256 "
          f"{res['station']['sha256']}.",
          f"- Global-model values: `{EXTRACT.relative_to(REPO_ROOT).as_posix()}`, SHA-256 {res['extract_sha256']}.",
          "- GraphCast 2018: " + ("included." if res["graphcast_2018"] else "not included (its extract or the raw "
                                  "hourly ERA5 for the 6-hourly correction is not on disk)."),
          "- Still to add, after P3-2 has run: " + "; ".join(res["pending"]) + ".",
          "- WBGT is not part of this check: the station record has no humidity.",
          "- A single station is not the truth for a 75 km box either: its own siting and the city around it shape "
          "its readings. The check asks whether conclusions depend on which record is used."]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "station_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    from training import test_lock

    test_lock.require()
    st = station_frame(load_station())
    era5 = era5_daily()
    coef = fit_mapping(era5["era5"], st["station"])
    res = {"station": {"sha256": STATION_SHA256, "coverage": _coverage(st)},
           "mapping": coef,
           "dating": dating_check(era5["era5"], st["station"]),
           "era5_vs_station": era5_vs_station(era5, st, coef),
           "test_forecasts": frozen_forecasts(st, era5, coef),
           "global_models": global_models(st, era5, coef),
           "graphcast_2018": graphcast_2018(st, coef),
           "pending": PENDING,
           "extract_sha256": hashlib.sha256(EXTRACT.read_bytes()).hexdigest()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "station_check.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    write_report(res)
    print(f"wrote {OUT_DIR / 'station_check.md'}")


if __name__ == "__main__":
    main()
