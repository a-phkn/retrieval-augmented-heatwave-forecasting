"""
pipeline/hourly_features.py: formula correctness, daily aggregation, and an
integration check that the hourly pipeline reproduces the frozen v1 Tmax series.
"""
import json

import numpy as np
import pandas as pd
import pytest

from pipeline.hourly_features import (
    RAW_DIR,
    REPO_ROOT,
    daily_cell_features,
    domain_daily,
    heat_index_c,
    load_cell_hourly,
    vapour_pressure_hpa,
    wbgt_bom,
    wet_bulb_stull,
)

f2c = lambda f: (f - 32) * 5 / 9  # noqa: E731
c2f = lambda c: c * 9 / 5 + 32  # noqa: E731


# ---------------------------------------------------------------- formulas


def test_vapour_pressure_known_values():
    # Saturation at 0 C is 6.105 hPa by construction; at 20 C ~23.3 hPa (Tetens/BoM).
    assert vapour_pressure_hpa(0.0, 100.0) == pytest.approx(6.105)
    assert vapour_pressure_hpa(20.0, 100.0) == pytest.approx(23.32, abs=0.02)
    assert vapour_pressure_hpa(20.0, 50.0) == pytest.approx(vapour_pressure_hpa(20.0, 100.0) / 2)


def test_wbgt_bom_worked_example():
    # Plan v5 / Appendix A: air 45 C at 15% RH -> e = 14.3 hPa, WBGT ~ 35.1 C (not 45).
    assert vapour_pressure_hpa(45.0, 15.0) == pytest.approx(14.31, abs=0.02)
    assert wbgt_bom(45.0, 15.0) == pytest.approx(0.567 * 45 + 0.393 * 14.31 + 3.94, abs=0.01)
    assert wbgt_bom(45.0, 15.0) == pytest.approx(35.07, abs=0.02)


def test_wet_bulb_stull_worked_example():
    # Stull (2011) worked example: T = 20 C, RH = 50% -> Tw = 13.7 C.
    assert wet_bulb_stull(20.0, 50.0) == pytest.approx(13.7, abs=0.05)
    # Saturated air: wet-bulb ~ air temperature (within the formula's ~0.3 C accuracy).
    assert wet_bulb_stull(30.0, 99.0) == pytest.approx(30.0, abs=0.4)


def test_heat_index_rothfusz_hand_computed():
    # Rothfusz regression evaluated by hand from its coefficients at 90 F / 50%: 94.598 F.
    assert c2f(heat_index_c(f2c(90.0), 50.0)) == pytest.approx(94.598, abs=0.01)


@pytest.mark.parametrize("t_f, rh, chart_f", [(90, 50, 95), (100, 40, 109), (80, 40, 80), (86, 90, 105)])
def test_heat_index_matches_nws_chart(t_f, rh, chart_f):
    # NWS heat index chart values (rounded); the regression is within ~1.3 F of the chart.
    assert c2f(heat_index_c(f2c(t_f), rh)) == pytest.approx(chart_f, abs=2.0)


def test_heat_index_branches_and_adjustments():
    # Below ~80 F the simple formula applies and HI stays close to T.
    assert c2f(heat_index_c(f2c(70.0), 50.0)) == pytest.approx(0.5 * (70 + 61 + 2 * 1.2 + 50 * 0.094))
    # Low-RH adjustment lowers HI (100 F, 10%): compare with the unadjusted regression.
    t, rh = 100.0, 10.0
    raw = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh - 0.00683783 * t**2
           - 0.05481717 * rh**2 + 0.00122874 * t**2 * rh + 0.00085282 * t * rh**2 - 0.00000199 * t**2 * rh**2)
    adj = ((13 - rh) / 4) * np.sqrt((17 - abs(t - 95)) / 17)
    assert c2f(heat_index_c(f2c(t), rh)) == pytest.approx(raw - adj, abs=1e-6)
    # High-RH adjustment raises HI (85 F, 90%).
    t, rh = 85.0, 90.0
    raw = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh - 0.00683783 * t**2
           - 0.05481717 * rh**2 + 0.00122874 * t**2 * rh + 0.00085282 * t * rh**2 - 0.00000199 * t**2 * rh**2)
    assert c2f(heat_index_c(f2c(t), rh)) == pytest.approx(raw + ((rh - 85) / 10) * ((87 - t) / 5), abs=1e-6)


def test_heat_index_increases_with_humidity_in_heat():
    rh = np.arange(20, 80, 5.0)
    hi = heat_index_c(np.full_like(rh, 38.0), rh)
    assert np.all(np.diff(hi) > 0)


def test_formulas_vectorise_and_propagate_nan():
    out = wbgt_bom(np.array([30.0, np.nan]), np.array([50.0, 50.0]))
    assert out.shape == (2,) and np.isfinite(out[0]) and np.isnan(out[1])
    assert np.isnan(heat_index_c(np.nan, 50.0))


# ---------------------------------------------------------------- aggregation


def _write_month(tmp_path, cell, start, hours, t, rh):
    times = pd.date_range(start, periods=hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    d = tmp_path / f"cell_{cell}"
    d.mkdir(parents=True, exist_ok=True)
    payload = {"hourly": {"time": times, "temperature_2m": list(t), "relative_humidity_2m": list(rh),
                          "wind_speed_10m": [1.0] * hours, "surface_pressure": [980.0] * hours}}
    (d / f"{start[:7]}.json").write_text(json.dumps(payload))


def test_daily_features_on_synthetic_day(tmp_path):
    t = [25.0] * 24
    t[14] = 40.0  # peak at 14:00
    rh = [60.0] * 24
    rh[14] = 20.0
    _write_month(tmp_path, 1, "2020-05-01T00:00", 24, t, rh)
    d = daily_cell_features(load_cell_hourly(1, tmp_path))
    row = d.iloc[0]
    assert row["t_max"] == 40.0 and row["rh_at_tmax"] == 20.0
    assert row["wbgt_bom_max"] == pytest.approx(max(wbgt_bom(40.0, 20.0), wbgt_bom(25.0, 60.0)))
    assert row["wbgt_bom_max_hour"] in (14, 0)


def test_incomplete_days_are_dropped(tmp_path):
    _write_month(tmp_path, 1, "2020-05-01T00:00", 30, [30.0] * 30, [50.0] * 30)  # day 2 has 6 h
    d = daily_cell_features(load_cell_hourly(1, tmp_path))
    assert len(d) == 1


def test_days_with_nan_hours_are_dropped(tmp_path):
    t = [30.0] * 48
    t[30] = None  # one unpublished hour on day 2 (json null)
    _write_month(tmp_path, 1, "2020-05-01T00:00", 48, t, [50.0] * 48)
    d = daily_cell_features(load_cell_hourly(1, tmp_path))
    assert list(d.index.strftime("%Y-%m-%d")) == ["2020-05-01"]


def test_missing_hours_raise(tmp_path):
    times = pd.date_range("2020-05-01", periods=48, freq="h").delete(10).strftime("%Y-%m-%dT%H:%M").tolist()
    d = tmp_path / "cell_1"
    d.mkdir()
    n = len(times)
    (d / "2020-05.json").write_text(json.dumps({"hourly": {
        "time": times, "temperature_2m": [30.0] * n, "relative_humidity_2m": [50] * n,
        "wind_speed_10m": [1.0] * n, "surface_pressure": [980.0] * n}}))
    with pytest.raises(ValueError, match="missing hours"):
        load_cell_hourly(1, tmp_path)


# ---------------------------------------------------------------- integration with frozen v1


@pytest.mark.skipif(not RAW_DIR.exists(), reason="raw ERA5 data not present (gitignored)")
def test_domain_tmax_reproduces_frozen_v1_series():
    _, domain = domain_daily()
    v1 = pd.read_parquet(REPO_ROOT / "data/processed/weather_daily.parquet").set_index("date")
    # Same day coverage as v1 (counts only); values are compared on pre-test years only,
    # because the test period (2019+) stays locked even for input sanity checks.
    assert domain.index.min() == v1.index.min() and domain.index.max() == v1.index.max()
    assert len(domain) == len(v1) == 17051
    dev = domain[domain.index < "2019-01-01"]
    assert np.allclose(dev["t_max"].to_numpy(), v1["t_max"].reindex(dev.index).to_numpy(), atol=1e-9)

    # Physical sanity (pre-test years). The BoM index's +3.94 term puts it ABOVE air
    # temperature in cool humid weather, and it runs high in humid heat.
    assert dev["wbgt_bom_max"].between(5, 45).all()
    month = dev.index.month
    assert dev["wbgt_bom_max"][month.isin([6, 7])].mean() > dev["wbgt_bom_max"][month.isin([12, 1])].mean() + 10
    # Wet-bulb never exceeds air temperature.
    assert (dev["tw_max"] <= dev["t_max"] + 1e-9).all()
    # NWS Heat Index: below air temperature in very dry heat, above it in humid heat.
    dry = (dev["rh_at_tmax"] < 13) & (dev["t_max"] >= 35)
    humid = (dev["rh_at_tmax"] >= 50) & (dev["t_max"] >= 32)
    assert (dev.loc[dry, "hi_max"] - dev.loc[dry, "t_max"]).mean() < 0
    assert (dev.loc[humid, "hi_max"] > dev.loc[humid, "t_max"]).all()
