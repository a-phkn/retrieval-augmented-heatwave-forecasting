"""retrieval_study/hres_anen.py: box weights, the forecast keys and the analogue ensemble (offline, synthetic)."""
import numpy as np
import pandas as pd
import pytest

from retrieval_study import hres_anen as ha


def test_the_delhi_box_falls_in_two_15_degree_cells_with_equal_weight():
    lat = np.linspace(-90, 90, 121)  # with poles, 1.5 degrees
    lon = np.arange(0, 360, 1.5)
    ilat, ilon, w = ha.box_weights(lat, lon, (28.125, 28.875, 76.875, 77.625))
    assert list(lat[ilat]) == [28.5] and list(lon[ilon]) == [76.5, 78.0]
    np.testing.assert_allclose(w, [[0.5, 0.5]])
    ilat2, _, w2 = ha.box_weights(lat[::-1], lon, (28.125, 28.875, 76.875, 77.625))  # descending latitudes too
    assert lat[::-1][ilat2][0] == 28.5 and w2.sum() == pytest.approx(1.0)


def test_keys_use_the_run_started_12_utc_the_day_before():
    inits = pd.date_range("2016-01-01 12:00", periods=3, freq="D")
    rows = [(i, h, 100 * k + h) for k, i in enumerate(inits) for h in ha.LEADS_H]
    hres = pd.DataFrame(rows, columns=["init", "lead_hours", "t2m"])
    q = pd.DatetimeIndex(["2016-01-02", "2016-01-03"])
    k = ha.keys_table(hres, q)
    np.testing.assert_allclose(k[0, 0], [12, 18, 24, 30])  # lead 1: 00-18 UTC of 2016-01-02 from the 2016-01-01 run
    np.testing.assert_allclose(k[1, 4], [100 + 108, 100 + 114, 100 + 120, 100 + 126])  # lead 5 from the next run


def test_the_analogue_ensemble_uses_other_years_in_season_only():
    rng = np.random.default_rng(0)
    n = 600
    doy = np.tile(np.arange(1, 201), 3)
    year = np.repeat([2016, 2017, 2018], 200)
    keys = rng.normal(size=(n, 4))
    truth = keys.sum(axis=1) * 2.0 + 30  # the outcome follows the forecast
    out = ha.analogue_ensemble(keys, truth, doy, year, k=5, season=10)
    rand = ha.analogue_ensemble(keys, truth, doy, year, k=5, season=10, seed=1)
    assert np.sqrt(np.mean((out - truth) ** 2)) < 0.6 * np.sqrt(np.mean((rand - truth) ** 2))
    same = truth.copy()
    same[year == 2016] = 1000.0  # if a query could see its own year, 2016 forecasts would be near 1000
    assert ha.analogue_ensemble(keys, same, doy, year, k=5, season=10)[year == 2016].max() < 100
    with pytest.raises(ValueError):
        ha.analogue_ensemble(keys, truth, doy, year, k=5, season=0)  # too few members within 0 days... per year
    rho = ha.key_value_spearman(keys, truth - 30, doy, year)
    assert rho > 0.3
