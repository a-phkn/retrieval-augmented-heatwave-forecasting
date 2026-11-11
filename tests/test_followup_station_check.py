"""followup/station_check.py on small hand-made data (offline; the real station file and the test period are not read)."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from evaluation.stats import make_cluster_ids
from followup import station_check as sc
from pipeline.labels_v2 import hot_days_tmax, in_season


def seasonal(dates: pd.DatetimeIndex, peak: float = 40.0, swing: float = 12.0) -> np.ndarray:
    """A smooth yearly cycle peaking in early June."""
    return peak - swing * (1 - np.cos(2 * np.pi * (dates.dayofyear - 155) / 365.25)) / 2


def forecast_frame(dates: pd.DatetimeIndex, pred, actual, seed: int = 0, lead: int = 1, stratum="normal") -> pd.DataFrame:
    """One forecast row per target day, shaped like the prediction files."""
    df = pd.DataFrame({"query_date": dates - pd.Timedelta(days=lead - 1), "lead": lead, "target_date": dates,
                       "seed": seed, "pred": np.asarray(pred, float), "actual": np.asarray(actual, float),
                       "stratum": stratum, "cluster_id": make_cluster_ids(dates)})
    return df.assign(error=df["pred"] - df["actual"])


def flat_mapping(a: float = 0.0, b: float = 1.0) -> dict:
    return {m: {"a": a, "b": b, "n": 1} for m in range(1, 13)}


def test_only_unflagged_values_of_the_registered_file_are_used(tmp_path):
    path = tmp_path / "station.csv"
    path.write_text("date,tmax_c,tmin_c,prcp_mm,tmax_qflag,tmax_sflag\n"
                    "2020-05-01,41.2,25.0,0.0,,S\n2020-05-02,49.9,25.0,0.0,O,S\n2020-05-03,,25.0,0.0,,\n"
                    "2020-05-04,42.0,26.0,0.0,,S\n", encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    s = sc.load_station(path, sha)
    assert list(s.index) == list(pd.to_datetime(["2020-05-01", "2020-05-04"])) and list(s) == [41.2, 42.0]
    with pytest.raises(ValueError, match="pre-registered"):
        sc.load_station(path, "0" * 64)
    with pytest.raises(FileNotFoundError, match="not in the repository"):
        sc.load_station(tmp_path / "missing.csv")


def test_the_threshold_is_the_heatwave_rule_and_applies_in_season_only():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    t, normal = rng.uniform(34, 48, len(dates)), rng.uniform(34, 44, len(dates))
    label = t >= sc.threshold(normal, dates, 3.0)
    assert np.array_equal(label, hot_days_tmax(dates, t, t - normal)) and label.any()
    imd = t >= sc.threshold(normal, dates, 4.5)
    assert np.array_equal(imd, in_season(dates) & (((t >= 40) & (t - normal >= 4.5)) | (t >= 45)))
    assert not imd[~in_season(dates)].any() and imd.sum() < label.sum()


def test_the_station_normal_comes_from_1980_2018_only():
    dates = pd.date_range("1980-01-01", "2025-08-23", freq="D")
    base = pd.Series(seasonal(dates), index=dates)
    hot_later = base.copy()
    hot_later[dates.year >= 2019] += 6.0  # the years after the fitting period run much hotter
    a, b = sc.station_frame(base), sc.station_frame(hot_later)
    np.testing.assert_allclose(a["st_normal"], b["st_normal"])  # the normal ignores 2019 onwards
    assert not a["st_label"].any()  # a year that follows its normal has no heatwave day
    late = b[b.index.year >= 2019]
    assert late["st_imd"].any() and not late.loc[~late["season"], "st_label"].any()  # +6 °C: heatwaves, in season only
    assert (b.loc[b["st_imd"], "station"] >= 40).all() and b["st_label"].sum() >= b["st_imd"].sum()


def test_the_mapping_is_a_monthly_line_fitted_on_the_fit_years_only():
    dates = pd.date_range("1980-01-01", "2025-08-23", freq="D")
    rng = np.random.default_rng(2)
    era5 = pd.Series(seasonal(dates) + rng.normal(0, 2, len(dates)), index=dates)
    a_true, b_true = 1.0 + 0.1 * dates.month.to_numpy(), 1.0 + 0.01 * dates.month.to_numpy()
    station = pd.Series(a_true + b_true * era5.to_numpy(), index=dates)
    station[dates.year >= 2019] += 9.0  # must not move the fit
    station = station.drop(station.index[::7])  # gaps
    coef = sc.fit_mapping(era5, station)
    for m in range(1, 13):
        assert coef[m]["a"] == pytest.approx(1.0 + 0.1 * m, abs=1e-6) and coef[m]["b"] == pytest.approx(1.0 + 0.01 * m, abs=1e-6)
    july = pd.to_datetime(["2020-07-04"])
    assert sc.apply_mapping([30.0], july, coef)[0] == pytest.approx(1.7 + 1.07 * 30.0)
    with pytest.raises(ValueError, match="every calendar month"):
        sc.fit_mapping(era5[era5.index.month != 2], station)


def test_forecast_rows_without_a_station_value_are_left_out():
    dates = pd.to_datetime(["2020-05-01", "2020-05-02", "2020-05-03", "2020-12-01"])
    st = pd.DataFrame({"station": [44.0, 41.0, 30.0], "st_normal": [39.0, 39.0, 25.0], "thr_label": [42.0, 42.0, np.inf],
                       "st_label": [True, False, False], "thr_imd": [43.5, 43.5, np.inf], "st_imd": [True, False, False],
                       "season": [True, True, False]}, index=dates[[0, 1, 3]])
    thr = pd.Series([41.0, 41.0, 41.0, np.inf], index=dates)
    df = forecast_frame(dates, [42.0, 40.0, 41.0, 28.0], [42.5, 41.5, 40.5, 29.0])
    rows = sc.station_rows(df, st, thr, flat_mapping(a=1.0, b=1.0))
    assert list(rows["target_date"]) == list(dates[[0, 1, 3]])  # 3 May has no station value
    assert list(rows["station"]) == [44.0, 41.0, 30.0] and list(rows["mapped"]) == [43.0, 41.0, 29.0]
    assert list(rows["actual"]) == [42.5, 41.5, 29.0] and list(rows["st_label"]) == [True, False, False]
    assert list(rows["thr_era5"]) == [41.0, 41.0, np.inf] and list(rows["season"]) == [True, True, False]
    with pytest.raises(ValueError, match="no ERA5 label threshold"):
        sc.station_rows(df, st, thr.iloc[1:], flat_mapping())


def rows_for(dates, pred, actual, station, st_label=None, seed: int = 0, lead: int = 1) -> pd.DataFrame:
    """Rows as station_rows returns them, with a mapped forecast 1 °C above the raw one."""
    df = forecast_frame(dates, pred, actual, seed=seed, lead=lead)
    hot = np.zeros(len(dates), bool) if st_label is None else np.asarray(st_label, bool)
    return df.assign(station=np.asarray(station, float), thr_era5=np.where(in_season(dates), 41.0, np.inf),
                     season=in_season(dates), mapped=df["pred"] + 1.0, thr_label=np.where(in_season(dates), 43.0, np.inf),
                     st_label=hot, thr_imd=np.where(in_season(dates), 44.5, np.inf), st_imd=hot & False)


def test_scores_are_given_against_both_truths_and_the_cold_bias_adds_up():
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    rng = np.random.default_rng(3)
    era5 = seasonal(dates) + rng.normal(0, 1.5, len(dates))
    station = era5 + 2.0  # the station reads 2 °C above ERA5
    hot = in_season(dates) & (station >= 43.0)
    parts = [rows_for(dates, era5 - 1.0 + rng.normal(0, 0.5, len(dates)), era5, station, hot, seed=s) for s in (0, 1)]
    rows = pd.concat(parts, ignore_index=True)
    v = sc.three_views(rows)
    assert v["era5"]["all"]["n"] == len(dates) and v["era5"]["all"]["clusters"] == 9
    assert v["era5"]["all"]["bias"] == pytest.approx(-1.0, abs=0.05) and v["station"]["all"]["bias"] == pytest.approx(-3.0, abs=0.05)
    assert v["station_mapped"]["all"]["bias"] == pytest.approx(-2.0, abs=0.05)  # mapped = raw + 1 in this toy
    assert v["station"]["all"]["rmse"] > v["era5"]["all"]["rmse"] and v["station"]["st_label"]["n"] == int(hot.sum())
    assert len(v["station"]["by_lead_all"]) == 5 and v["station"]["by_lead_all"][0]["n"] == len(dates)
    assert v["station"]["by_lead_all"][1]["n"] == 0 and np.isnan(v["station"]["by_lead_all"][1]["rmse"])
    c = sc.cold_bias(parts[0])["label"]["all_leads"]
    assert c["n"] == int(hot.sum())
    assert c["forecast_minus_station"] == pytest.approx(c["forecast_minus_era5"] + c["era5_minus_station"])
    assert c["era5_minus_station"] == pytest.approx(-2.0) and c["era5_share"] == pytest.approx(-2.0 / c["forecast_minus_station"])
    assert sc.cold_bias(parts[0])["imd"]["all_leads"] == {"n": 0}


def test_paired_tests_use_the_same_rows_under_each_truth():
    dates = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    rng = np.random.default_rng(4)
    era5 = seasonal(dates) + rng.normal(0, 1.5, len(dates))
    station = era5 + 3.0
    hot = in_season(dates) & (station >= 43.0)
    cold = rows_for(dates, era5 + rng.normal(0, 0.3, len(dates)), era5, station, hot)  # exact for ERA5, 3 °C below the station
    warm = rows_for(dates, era5 + 3.0 + rng.normal(0, 0.3, len(dates)), era5, station, hot)  # exact for the station
    t = sc.paired(cold, warm)  # RMSE(warm) - RMSE(cold)
    assert set(t) == set(sc.VIEWS) and set(t["era5"]) == set(sc.TEST_STRATA)
    assert t["era5"]["all"]["delta"] > 2 and t["era5"]["all"]["ci_low"] > 0  # warm is worse against ERA5
    assert t["station"]["all"]["delta"] < -2 and t["station"]["all"]["ci_high"] < 0  # and better against the station
    assert t["station"]["all"]["n_clusters"] == 18 and t["station"]["season"]["n_clusters"] == 6
    assert sc.clear_change(t["era5"]["all"], t["station"]["all"])
    assert not sc.clear_change(t["era5"]["all"], t["era5"]["all"])
    assert not sc.clear_change({"error": "too few"}, t["station"]["all"])
    assert not sc.clear_change({"ci_low": float("nan"), "ci_high": float("nan")}, t["station"]["all"])
    one_day = rows_for(dates[:1], [30.0], [30.0], [31.0])
    assert "error" in sc.paired(one_day, one_day)["station"]["st_label"]  # no station heatwave rows: not testable


def test_warnings_use_the_threshold_that_belongs_to_each_forecast():
    dates = pd.to_datetime(["2020-05-01", "2020-05-02", "2020-05-03", "2020-05-04", "2020-12-01"])
    #                 pred  ERA5  station   ERA5 thr 41, station thr 43 (label), mapped = pred + 1
    rows = rows_for(dates, [41.5, 42.5, 40.0, 40.5, 45.0], [41.2, 40.0, 41.0, 39.0, 20.0], [44.0, 43.5, 44.0, 40.0, 21.0],
                    st_label=[True, True, True, False, False])
    probs = rows[sc.KEY].assign(p=[0.2, 0.5, 0.4, 0.1, 0.9]).iloc[:4]  # in-season rows only, as prob_rows gives
    w = sc.warnings_table(rows, probs, 0.37)
    assert w["rows"] == 4  # the December row is out of season
    st = w["events"]["station, label rule (normal + 3)"]
    assert st["n"] == 3
    point, prob, mapped = st["rules"]["R-point"], st["rules"]["R-prob (p* = 0.37)"], st["rules"]["station-mapped R-point"]
    assert (point["hits"], point["misses"], point["false_alarms"]) == (2, 1, 0)  # pred >= 41 on days 1 and 2
    assert (prob["hits"], prob["misses"], prob["false_alarms"]) == (2, 1, 0)  # p >= 0.37 on days 2 and 3
    assert (mapped["hits"], mapped["misses"], mapped["false_alarms"]) == (1, 2, 0)  # pred + 1 >= 43 on day 2 only
    ref = w["events"]["ERA5, label rule (reference)"]
    assert ref["n"] == 2 and ref["rules"]["R-point"]["hits"] == 1 and ref["rules"]["R-point"]["false_alarms"] == 1
    assert w["events"]["station, IMD rule (normal + 4.5)"]["n"] == 0
    assert "R-prob (p* = None)" not in sc.warnings_table(rows)["events"]["station, label rule (normal + 3)"]["rules"]
    with pytest.raises(ValueError, match="single forecast"):
        sc.warnings_table(pd.concat([rows, rows.assign(seed=1)]))
    with pytest.raises(ValueError, match="no probability"):
        sc.warnings_table(rows, probs.iloc[:3], 0.37)


def test_era5_and_station_labels_are_cross_tabulated_per_period():
    dates = pd.to_datetime(["2018-05-01", "2018-05-02", "2018-05-03", "2018-12-01", "2020-05-01", "2020-05-02"])
    st = pd.DataFrame({"station": [44.0, 44.0, 41.0, 22.0, 45.5, 40.0], "st_normal": 39.0,
                       "thr_label": [42.0, 42.0, 42.0, np.inf, 42.0, 42.0], "st_label": [True, True, False, False, True, False],
                       "thr_imd": [43.5, 43.5, 43.5, np.inf, 43.5, 43.5], "st_imd": [True, True, False, False, True, False],
                       "season": [True, True, True, False, True, True]}, index=dates)
    era5 = pd.DataFrame({"era5": [42.0, 40.0, 41.5, 21.0, 43.0, 41.0], "era5_normal": 38.0,
                         "era5_hot": [True, False, True, False, True, False], "thr_era5": 41.0}, index=dates)
    out = sc.era5_vs_station(era5, st, flat_mapping(a=2.0, b=1.0))
    early, late = out["1980-2018"], out["2019-2025"]
    assert (early["days"], early["in_season_days"], late["days"]) == (4, 3, 2)
    assert early["strata"]["all"]["bias"] == pytest.approx(np.mean([-2.0, -4.0, 0.5, -1.0]))
    assert early["strata"]["st_label"]["n"] == 2 and early["strata"]["st_label"]["bias"] == pytest.approx(-3.0)
    assert early["strata"]["st_label"]["mapped_bias"] == pytest.approx(-1.0)  # the mapping adds 2 °C
    assert early["strata"]["era5_hot"]["n"] == 2 and early["strata"]["era5_hot"]["bias"] == pytest.approx(-0.75)
    agree = early["labels"]["label"]["era5_label"]
    assert (agree["hits"], agree["misses"], agree["false_alarms"]) == (1, 1, 1)
    mapped = early["labels"]["label"]["mapped_era5"]  # ERA5 + 2 against 42: 44, 42, 43.5 -> all three days
    assert (mapped["hits"], mapped["misses"], mapped["false_alarms"]) == (2, 0, 1)
    assert late["labels"]["imd"]["era5_label"]["hits"] == 1 and late["strata"]["season"]["n"] == 2


def test_the_dating_check_tells_a_same_day_value_from_a_two_day_maximum():
    dates = pd.date_range("1980-01-01", "2025-08-23", freq="D")
    rng = np.random.default_rng(7)
    era5 = pd.Series(seasonal(dates) + rng.normal(0, 2.0, len(dates)), index=dates)
    true_max = era5 + 0.5  # the station really is 0.5 °C above ERA5 every day
    two_day = np.maximum(true_max, true_max.shift(1)).fillna(true_max)
    station = true_max.where(dates.year <= 2010, two_day)  # from 2011 the value dated D is the larger of D - 1 and D
    warmer = (era5 > era5.shift(1)).to_numpy()
    station = station[~((dates.year >= 2016) & warmer)]  # from 2016 the warmer days are missing
    out = sc.dating_check(era5, station)
    early, mid, late = out["blocks"]["1980-2008"], out["blocks"]["2011-2015"], out["blocks"]["2019-2025"]
    assert early["coverage"] == 1.0 and early["same"]["sd"] == pytest.approx(0.0, abs=1e-9)
    assert early["same"]["offset"] == pytest.approx(-0.5) and early["prev"]["sd"] > 1.0 and early["identical_share"] == 0.0
    assert early["cooler_share_reporting"] == pytest.approx(early["cooler_share_all"], abs=1e-3)  # nothing is missing
    assert mid["coverage"] == 1.0 and mid["both"]["sd"] == pytest.approx(0.0, abs=1e-9)
    assert mid["both"]["offset"] == pytest.approx(-0.5) and mid["same"]["offset"] < -1.0  # the same day looks too cold
    assert mid["identical_share"] > 0.2  # a peak's value is repeated on the next day
    assert 0.4 < late["coverage"] < 0.6 and late["cooler_share_reporting"] == 1.0 and late["cooler_share_all"] < 0.6
    assert late["prev"]["sd"] == pytest.approx(0.0, abs=1e-9) and late["prev"]["offset"] == pytest.approx(-0.5)
    assert late["same"]["sd"] > 1.0 and sorted(out["by_year"]) == list(range(1980, 2026))


def test_the_whole_result_is_scored_and_written_as_a_report(tmp_path, monkeypatch):
    fit = pd.date_range("1980-01-01", "2018-12-31", freq="D")
    dates = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    rng = np.random.default_rng(6)
    every = fit.append(dates)
    era5_all = seasonal(every) + rng.normal(0, 2.5, len(every))
    station = pd.Series(1.5 + 1.02 * era5_all + rng.normal(0, 0.8, len(every)), index=every).drop(every[::6])
    st = sc.station_frame(station)
    era5 = pd.DataFrame({"era5": era5_all, "era5_normal": seasonal(every)}, index=every)
    era5["thr_era5"] = sc.threshold(era5["era5_normal"], every, 3.0)
    era5["era5_hot"] = era5["era5"].to_numpy() >= era5["thr_era5"].to_numpy()
    coef = sc.fit_mapping(era5["era5"], st["station"])
    truth = era5.loc[dates, "era5"].to_numpy()

    def model(noise: float, n_seeds: int) -> pd.DataFrame:
        return pd.concat([forecast_frame(dates, truth - 0.5 + rng.normal(0, noise, len(dates)), truth, seed=s, lead=L)
                          for s in range(n_seeds) for L in (1, 2)], ignore_index=True)

    seeds = {"C3-hop": model(1.0, 3), "Delhi-only LSTM": model(1.6, 3), "ridge_hw5": model(2.0, 1)}
    in_seas = seeds["C3-hop"][(seeds["C3-hop"]["seed"] == 0) & in_season(seeds["C3-hop"]["target_date"])]
    probs = {name: (in_seas[sc.KEY].assign(p=rng.uniform(0, 1, len(in_seas))), 0.37) for name in sc.PROB_RULES}
    tf = sc.score_forecasts(seeds, probs, st, era5["thr_era5"], coef)
    assert tf["models"]["C3-hop"]["seeds"] == 3 and tf["models"]["ridge_hw5"]["per_seed"] is None
    assert tf["models"]["C3-hop"]["rows"] == 2 * station.index.isin(dates).sum()  # two leads per observed day
    lstm_test = tf["tests"]["C3-hop minus Delhi-only LSTM (3 seeds each)"]
    assert lstm_test["station"]["all"]["delta"] < 0 and lstm_test["station"]["all"]["n_seeds_child"] == 3
    assert tf["tests"]["C3-hop ensemble minus ridge_hw5"]["station"]["all"]["n_seeds_parent"] == 1
    assert "R-prob (p* = 0.37)" in tf["models"]["C3-hop"]["warnings"]["events"]["ERA5, label rule (reference)"]["rules"]

    from evaluation.compare_v2 import ensemble

    outside = {L: pd.Series(truth + rng.normal(0, 0.7, len(dates)), index=dates) for L in (1, 2)}
    res = {"station": {"sha256": "abc", "coverage": sc._coverage(st)}, "mapping": coef,
           "dating": sc.dating_check(era5["era5"], st["station"]),
           "era5_vs_station": sc.era5_vs_station(era5, st, coef), "test_forecasts": tf,
           "global_models": {"AIFS Single": sc.versus_c3hop("AIFS Single", outside, ensemble(seeds["C3-hop"]), st,
                                                           era5["thr_era5"], coef)},
           "graphcast_2018": None, "pending": sc.PENDING, "extract_sha256": "def"}
    monkeypatch.setattr(sc, "OUT_DIR", tmp_path)
    sc.write_report(res)
    text = (tmp_path / "station_check.md").read_text(encoding="utf-8")
    for part in ("## Read this first", "## Exploratory: are the station's dates right?",
                 "## Station data", "## (i) ERA5 against the station", "## (iv) The station mapping",
                 "## (ii) Our test forecasts against the station", "## (iii) Global models and GraphCast 2018",
                 "C3-hop (mean of 3 seeds)", "AIFS Single (added after the first results) minus C3-hop",
                 "GraphCast 2018: not included", "station-mapped R-point", "ERA5's share"):
        assert part in text, part
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert len(headings) == len(set(headings)) == 8  # every section once
    assert "nan" not in text.lower().replace("n/a", "")


def test_an_outside_forecast_is_paired_with_c3hop_on_shared_station_days():
    dates = pd.date_range("2024-03-01", "2025-08-23", freq="D")
    rng = np.random.default_rng(5)
    era5 = seasonal(dates) + rng.normal(0, 1.5, len(dates))
    station = pd.Series(era5 + 1.0, index=dates).drop(dates[::5])  # the station misses every fifth day
    st = sc.station_frame(pd.concat([pd.Series(seasonal(pd.date_range("1980-01-01", "2018-12-31")) + 1.0,
                                               index=pd.date_range("1980-01-01", "2018-12-31")), station]))
    thr = pd.Series(sc.threshold(seasonal(dates), dates, 3.0), index=dates)
    c3e = pd.concat([forecast_frame(dates, era5 + rng.normal(0, 2.0, len(dates)), era5, lead=L) for L in (1, 2)],
                    ignore_index=True).sort_values(sc.KEY).reset_index(drop=True)
    outside = {1: pd.Series(era5 + rng.normal(0, 0.5, len(dates)), index=dates)[dates >= "2024-06-01"]}  # lead 1 only
    r = sc.versus_c3hop("X", outside, c3e, st, thr, flat_mapping(a=1.0, b=1.0))
    shared = station.index[station.index >= "2024-06-01"]
    assert r["rows"] == len(shared) and len(shared) < (dates >= "2024-06-01").sum()  # station gaps drop rows
    assert r["first_target"] == str(shared[0].date()) and r["last_target"] == str(shared[-1].date())
    assert r["model"]["station"]["all"]["n"] == r["c3hop"]["station"]["all"]["n"] == len(shared)
    assert r["model"]["station"]["all"]["rmse"] < r["c3hop"]["station"]["all"]["rmse"]
    assert r["test"]["station"]["all"]["delta"] == pytest.approx(
        r["model"]["station"]["all"]["rmse"] - r["c3hop"]["station"]["all"]["rmse"])
    assert r["model"]["station_mapped"]["all"]["bias"] == pytest.approx(0.0, abs=0.1)  # +1 °C closes the station gap
    assert r["warnings"]["model"]["rows"] == r["warnings"]["c3hop"]["rows"]
    assert sc.versus_c3hop("none", {1: pd.Series(dtype=float)}, c3e, st, thr, flat_mapping()) is None
