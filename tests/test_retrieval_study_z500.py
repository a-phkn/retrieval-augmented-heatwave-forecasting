"""retrieval_study/z500_direct.py: the 500 hPa scores are appended to the right input days (offline, synthetic)."""
import numpy as np
import pandas as pd
import pytest

from retrieval_study import z500_direct as zd
from training.folds import FORECAST_DAYS, INPUT_DAYS, SplitArrays


def split(query_dates: pd.DatetimeIndex, n_features: int = 3) -> SplitArrays:
    n = len(query_dates)
    zeros5 = np.zeros((n, FORECAST_DAYS), dtype=np.float32)
    return SplitArrays(X=np.zeros((n, INPUT_DAYS, n_features), dtype=np.float32), y=zeros5, y_raw=zeros5,
                       clim_target=zeros5.astype(np.float64), hot=zeros5.astype(bool),
                       stratum=np.full((n, FORECAST_DAYS), "normal"), query_dates=pd.Series(query_dates))


def test_each_window_gets_the_scores_of_its_own_14_input_days():
    days = pd.date_range("2010-01-01", periods=60, freq="D")
    pcs = pd.DataFrame(np.arange(60)[:, None] + np.arange(zd.N_PCS)[None, :] / 100, index=days)
    s = split(days[[20, 40]])
    out = zd.with_z500(s, pcs)
    assert out.X.shape == (2, INPUT_DAYS, 3 + zd.N_PCS) and out.X.dtype == np.float32
    np.testing.assert_allclose(out.X[0, :, 3], np.arange(6, 20))  # days 6..19 for the window starting on day 20
    np.testing.assert_allclose(out.X[1, -1, 3:], 39 + np.arange(zd.N_PCS) / 100, rtol=1e-6)
    np.testing.assert_array_equal(out.X[:, :, :3], 0)  # the original channels are untouched
    with pytest.raises(ValueError):
        zd.with_z500(split(days[[5]]), pcs)  # not enough input days


def test_the_control_takes_the_same_dates_from_another_training_year():
    days = pd.date_range("2004-01-01", "2009-12-31", freq="D")  # fold f1: training years end 2006
    pcs = pd.DataFrame({c: days.year * 1000 + days.dayofyear for c in range(zd.N_PCS)}, index=days, dtype=float)
    out = zd.shuffled_pcs(pcs, "f1")
    donor_year = (out[0] // 1000).astype(int)
    assert (donor_year != days.year).all() and donor_year.isin([2004, 2005, 2006]).all()  # never itself; training years
    assert (donor_year.groupby(days.year).nunique() == 1).all()  # one donor year per year: sequences stay coherent
    same = days[(days.month != 2) | (days.day != 29)]
    np.testing.assert_array_equal(out.loc[same, 0] % 1000, pd.DatetimeIndex(
        [pd.Timestamp(year=int(y), month=d.month, day=d.day) for y, d in zip(donor_year.loc[same], same)]).dayofyear)
    leap_day = pd.Timestamp("2008-02-29")
    assert out.loc[leap_day, 0] % 1000 in (59, 60)  # 28 Feb of a common year, or 29 Feb of a leap one
    np.testing.assert_array_equal(zd.shuffled_pcs(pcs, "f1").to_numpy(), out.to_numpy())  # fixed seed


def test_the_report_lists_every_comparison_and_its_lead_row(tmp_path, monkeypatch):
    monkeypatch.setattr(zd, "OUT_DIR", tmp_path)
    t = {"parent": 2.2, "child": 2.1, "delta": -0.1, "ci_low": -0.15, "ci_high": -0.05, "p_value": 0.01, "n_clusters": 36}
    null = {**t, "parent": 2.2, "child": 2.21, "delta": 0.01, "ci_low": -0.01, "ci_high": 0.03, "p_value": 0.3}
    refs = ("C3-hop", "the shuffled-year control")
    res = {"WBGT": {"tests": {f"vs {n}|{s}": t for n in refs for s in ("all", "extreme")},
                    "by_lead": [1.0] * 5, **{f"by_lead {n}": [1.1] * 5 for n in refs},
                    "bias": {"all": 0.01, "extreme": -0.5}, **{f"bias {n}": {"all": 0.02, "extreme": -0.9} for n in refs},
                    "control vs C3-hop": {"all": null, "extreme": null}},
           "Tmax": {"status": "not complete"}}
    zd.write_report(res)
    text = (tmp_path / "step4.md").read_text(encoding="utf-8")
    assert "| Its RMSE |" in text and text.count("**(CI < 0)**") == 4
    for row in ("| C3-hop | 1.100", "| the shuffled-year control | 1.100", "| C3-hop + 500 hPa input | 1.000",
                "| C3-hop | +0.02 | -0.90 |", "| C3-hop + 500 hPa input | +0.01 | -0.50 |",
                "| all days | 2.200 | 2.210 | +0.010 [-0.010, +0.030] |", "| heatwave days | 2.200 | 2.210 |"):
        assert row in text
    assert "Not complete yet" in text


def test_bias_is_the_mean_error_on_all_days_and_on_heatwave_days():
    df = pd.DataFrame({"seed": [0, 0, 1, 1], "stratum": ["normal", "extreme"] * 2, "error": [1.0, -2.0, 3.0, -4.0]})
    assert zd.bias(df) == {"all": -0.5, "extreme": -3.0}


def long_frame(query_dates: pd.DatetimeIndex, seeds=(0, 1), offset: float = 0.0) -> pd.DataFrame:
    """A prediction frame in the shape evaluation.predict_v1.long_frame writes (error = offset everywhere)."""
    rows = [{"model": "m", "seed": seed, "query_date": q, "lead": lead, "target_date": q + pd.Timedelta(days=lead - 1),
             "pred": 30.0 + offset, "actual": 30.0, "stratum": "normal", "cluster_id": 20191}
            for seed in seeds for q in query_dates for lead in range(1, FORECAST_DAYS + 1)]
    df = pd.DataFrame(rows)
    return df.assign(error=df["pred"] - df["actual"])


def test_same_windows_cuts_the_test_forecasts_to_the_test2021_windows():
    child = long_frame(pd.date_range("2019-03-01", periods=3))
    parent = long_frame(pd.date_range("2019-02-25", periods=12), offset=1.0)  # the longer 2019-2026 run
    out = zd.same_windows(parent, child)
    keys = ["seed", "query_date", "lead"]
    assert list(out.columns) == list(parent.columns)
    pd.testing.assert_frame_equal(out[keys].reset_index(drop=True), child[keys].reset_index(drop=True))
    assert (out["error"] == 1.0).all()
    with pytest.raises(ValueError, match="lacks"):
        zd.same_windows(parent[parent["query_date"] != pd.Timestamp("2019-03-02")], child)  # a window is missing
    with pytest.raises(ValueError, match="truth"):
        zd.same_windows(parent.assign(actual=parent["actual"] + 0.5), child)  # another truth


def test_the_control_and_unknown_folds_are_refused_before_any_training():
    with pytest.raises(ValueError, match="development folds only"):
        zd.run_family("WBGT", [zd.STEP6_FOLD], control=True)
    with pytest.raises(ValueError, match="unknown fold"):
        zd.run_family("WBGT", ["f9"])


def test_the_step6_report_applies_the_fixed_reading_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(zd, "OUT_DIR", tmp_path)
    nan = float("nan")
    gain = {"parent": 2.30, "child": 2.25, "delta": -0.05, "ci_low": -0.09, "ci_high": -0.01, "p_value": 0.03,
            "n_clusters": 9}
    loss = {**gain, "child": 2.32, "delta": 0.02, "ci_low": -0.01, "ci_high": 0.05, "p_value": 0.2}
    heat = {"parent": 3.1, "child": 3.0, "delta": -0.1, "ci_low": nan, "ci_high": nan, "p_value": nan, "n_clusters": 3}

    def family(short: str, all_days: dict) -> dict:
        return {"n_windows": 1092, "n_seeds": 10, "tests": {"vs C3-hop|all": all_days, "vs C3-hop|extreme": heat},
                "by_lead": [1.0] * 5, "by_lead C3-hop": [1.1] * 5, "bias": {"all": 0.1, "extreme": -2.0},
                "bias C3-hop": {"all": 0.2, "extreme": -2.5}, "reading": zd.step6_reading(short, all_days, heat)}

    res = {"WBGT": family("WBGT", gain), "Tmax": family("Tmax", loss)}
    assert res["WBGT"]["reading"]["P13 (all days below 0)"] and res["WBGT"]["reading"]["P14 (smaller than in development)"]
    assert res["Tmax"]["reading"]["all days"] == "failure to replicate (fragile: 9 clusters)"
    assert not res["Tmax"]["reading"]["P13 (all days below 0)"]
    assert zd.step6_reading("Tmax", {**gain, "delta": -0.09}, heat)["P14 (smaller than in development)"] is False
    zd.write_report_step6(res)
    text = (tmp_path / "step6.md").read_text(encoding="utf-8")
    for row in ("| C3-hop | all days | 2.300 | 2.250 | -0.050 [-0.090, -0.010], fragile: 9 clusters |",
                "| C3-hop | heatwave days | 3.100 | 3.000 | -0.100 (no CI: 3 clusters) |",
                "no conclusion (no prediction was made); no CI with 3 clusters",
                "- P13 (all days below 0 in both families): failed.", "| C3-hop + 500 hPa input | +0.10 | -2.00 |"):
        assert row in text
    assert zd._json_safe({"a": nan, "b": [np.float64(1.5), np.int64(2), np.bool_(True)]}) == {"a": None,
                                                                                            "b": [1.5, 2, True]}
