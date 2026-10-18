"""evaluation/analog_correction.py (option 2): honest pool, analog correction and its controls (synthetic)."""
import numpy as np
import pandas as pd
import pytest

import evaluation.analog_correction as ac


def test_doy_distance_is_circular():
    assert ac.doy_distance(np.array([1, 360, 100]), np.array([365, 5, 150])).tolist() == [1, 10, 50]


def test_pool_uses_only_honest_past_rows():
    def frame(q0, q1):
        q = pd.date_range(q0, q1, freq="D")
        return pd.DataFrame({"query_date": q, "target_date": q + pd.Timedelta(days=4), "seed": 0, "lead": 5})
    inner = frame("2011-01-01", "2012-12-31")  # its last targets spill past the validation start
    earlier = [frame("2007-01-01", "2009-12-31"), frame("2010-01-01", "2012-12-31")]
    pool = ac.build_pool(inner, earlier, pd.Timestamp("2013-01-01"))
    assert (pool["target_date"] < pd.Timestamp("2013-01-01")).all()
    assert not pool["query_date"].duplicated().any()  # the inner block wins on overlapping dates
    assert pool["query_date"].min() == pd.Timestamp("2007-01-01")


def _block(n, rng, doy=None):
    pred = rng.normal(30, 3, n)
    fc_mean = pred + rng.normal(0, 0.5, n)
    last = pred + rng.normal(0, 1, n)
    err = np.where(pred > 33, -2.0, 0.0) + rng.normal(0, 0.1, n)  # the model is too cold only on hot forecasts
    return pd.DataFrame({"pred": pred, "fc_mean": fc_mean, "last_obs": last, "error": err, "actual": pred - err,
                         "doy": doy if doy is not None else rng.integers(150, 200, n)})


def test_analogs_recover_a_state_dependent_bias_that_random_and_linear_miss():
    rng = np.random.default_rng(0)
    pool, val = _block(3000, rng), _block(400, rng)
    out = ac.correct_block(val, pool, np.random.default_rng(1))
    truth = val["actual"].to_numpy()
    rmse = {k: np.sqrt(np.mean((v - truth) ** 2)) for k, v in out.items()}
    assert rmse["anen"] < rmse["random"] and rmse["anen"] < rmse["linear"] and rmse["anen"] < rmse["raw"]
    hot = val["pred"].to_numpy() > 34
    assert np.mean(out["anen"][hot] - truth[hot]) == pytest.approx(0.0, abs=0.3)


def test_season_window_is_enforced():
    rng = np.random.default_rng(0)
    pool = _block(500, rng, doy=np.full(500, 10))  # all candidates far from the query's season
    with pytest.raises(ValueError, match="in-season"):
        ac.correct_block(_block(5, rng, doy=np.full(5, 200)), pool, rng)
