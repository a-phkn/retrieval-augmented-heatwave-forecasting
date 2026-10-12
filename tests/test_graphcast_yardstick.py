"""evaluation/graphcast_yardstick.py: timing, area weights and the sampling correction (no network)."""
import numpy as np
import pandas as pd
import pytest

import evaluation.graphcast_yardstick as gy


def test_init_is_12utc_the_day_before_and_leads_hit_the_target_days_utc_times():
    q = pd.Timestamp("2018-05-10")
    init, leads = gy.init_and_leads(q, 1)
    assert init == pd.Timestamp("2018-05-09 12:00")
    assert [lt / pd.Timedelta(hours=1) for lt in leads] == [12, 18, 24, 30]  # 00/06/12/18 UTC on 05-10
    _, leads5 = gy.init_and_leads(q, 5)
    assert [init + lt for lt in leads5][0] == pd.Timestamp("2018-05-14 00:00")


def test_chunk_plan_maps_inits_and_leads_to_zarr_chunks():
    times = pd.date_range("2017-11-16", periods=10, freq="12h")
    file_leads = np.arange(6, 246, 6)  # 6..240 h, as in the WeatherBench 2 file
    plan = gy.chunk_plan([times[3]], [12, 48, 54], times, file_leads, lead_chunk=8)
    # lead 12 h is index 1 (chunk 0); 48 h is index 7 (chunk 0); 54 h is index 8 (chunk 1, position 0)
    assert plan == {(3, 0): [(times[3], 12, 1), (times[3], 48, 7)], (3, 1): [(times[3], 54, 0)]}


def test_overlap_weights_cover_the_box_proportionally():
    w = gy.overlap_weights(np.array([28.0, 28.5, 29.0]), np.array([76.5, 77.2, 77.9]))
    assert w.sum() == pytest.approx(1.0)
    # box 28.125-28.875 splits evenly in latitude; longitude 76.875-77.2 vs 77.2-77.625
    np.testing.assert_allclose(w.sum(axis=1), [0.5, 0.5])
    np.testing.assert_allclose(w.sum(axis=0), [0.325 / 0.75, 0.425 / 0.75])


def test_sampled_max_uses_the_four_half_hours_and_misses_the_afternoon_peak():
    idx = pd.date_range("2018-05-10", periods=48, freq="h")
    t = pd.Series(30.0, index=idx)
    t[idx.hour == 14] = 45.0  # afternoon peak between the 11:30 and 17:30 samples
    t[(idx.hour == 11) | (idx.hour == 12)] = 36.0
    s = gy.sampled_daily_max(t)
    assert s.loc["2018-05-10"] == pytest.approx(36.0)  # the 11:30 sample, not the 45 °C peak


def test_correction_recovers_a_monthly_linear_relation():
    rng = np.random.default_rng(0)
    idx = pd.date_range("1990-01-01", "2000-12-31", freq="D")
    sampled = rng.normal(30, 4, len(idx))
    true = 1.5 + 1.02 * sampled + np.where(idx.month == 5, 1.0, 0.0)
    coef = gy.fit_correction(pd.DataFrame({"true": true, "sampled": sampled}, index=idx), years=(1990, 2000))
    assert coef[1] == pytest.approx((1.5, 1.02), abs=1e-6) and coef[5][0] == pytest.approx(2.5, abs=1e-6)
    np.testing.assert_allclose(gy.apply_correction(np.array([30.0]), np.array([5]), coef), 2.5 + 30.6)


def test_hourly_reconstruction_matches_the_tmax_target():
    """'true' from our hourly files must equal the project's Tmax target (mean of cell maxima)."""
    daily = gy.era5_daily_true_and_sampled()
    target = pd.read_parquet(gy.REPO_ROOT / "datasets_v2" / "all_daily_v2.parquet", columns=["date", "t_max"])
    target = target.set_index("date")["t_max"]
    both = daily.join(target, how="inner").loc["2010":"2015"]
    np.testing.assert_allclose(both["true"], both["t_max"], atol=1e-4)
    assert (daily.index < pd.Timestamp("2019-01-01")).all()
    assert (both["sampled"] <= both["true"] + 1e-6).all()  # samples never exceed the daily max
