"""Rz ("synoptic" mode, pre-registered 2026-10-17): matching on 500 hPa height patterns, on a synthetic
extract with a planted repeat; the screen's pass rule; the run's data fingerprint."""
import numpy as np
import pandas as pd
import pytest

import pipeline.download_era5_z500 as dz
import retrieval.fold_retrieval as fr_mod

LAT = -88.59375 + 2.8125 * np.arange(64)
LON = 2.8125 * np.arange(128)


@pytest.fixture
def synthetic_z500(tmp_path, monkeypatch):
    ilat, ilon = dz.box_indices(LAT, LON)
    dates = pd.date_range("1980-01-01", "2018-12-31", freq="D", name="date")
    rng = np.random.default_rng(0)
    doy = dates.dayofyear.to_numpy()[:, None]
    vals = 5800 + 40 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0, 30, (len(dates), 252))
    df = pd.DataFrame(vals.astype(np.float32), index=dates, columns=dz.columns(LAT[ilat], LON[ilon]))
    q, c = pd.Timestamp("1995-01-20"), pd.Timestamp("1990-01-20")  # same day of year, so same climatology
    for k in range(1, 4):
        df.loc[q - pd.Timedelta(days=k)] = df.loc[c - pd.Timedelta(days=k)].to_numpy()
    path = tmp_path / "z500.parquet"
    df.to_parquet(path)
    monkeypatch.setattr(fr_mod, "Z500_PATH", path)
    return q, c


def test_planted_repeat_is_the_top_analogue_and_vectors_are_unit_pca_scores(synthetic_z500):
    q, c = synthetic_z500
    r = fr_mod.FoldRetriever("f1", "v2", "t_max")
    out = r.retrieve(pd.DatetimeIndex([q]), "synoptic", 5)
    assert r.cand_dates[out.idx[0, 0]] == c and out.sim[0, 0] == pytest.approx(1.0, abs=1e-9)
    vec, mean, comps, za, w = r._synoptic_state()
    assert comps.shape == (fr_mod.SYNOPTIC_PCS, 3 * 252) and vec.shape == (len(r.cand_dates), fr_mod.SYNOPTIC_PCS)
    np.testing.assert_allclose(np.linalg.norm(vec, axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(comps @ comps.T, np.eye(fr_mod.SYNOPTIC_PCS), atol=1e-8)  # orthonormal patterns
    assert (w > 0).all() and w.max() <= 1 and za.index.max() < pd.Timestamp("2019-01-01")
    # the same eligibility as every mode: chronological buffer and dedup spacing
    days = r.cand_dates[out.idx[0]]
    assert (days <= q - pd.Timedelta(days=fr_mod.BUFFER_DAYS)).all()
    gaps = np.abs(np.subtract.outer(days.values, days.values)).astype("timedelta64[D]").astype(int)
    assert (gaps[np.triu_indices(5, 1)] >= fr_mod.MIN_DAYS_APART).all()


def test_screen_pass_rule_needs_both_contrasts_below_zero():
    from evaluation.synoptic_screen import passes

    ok = {"all": {"Rz vs base": (-0.02, -0.03, -0.01), "Rz vs R0-rand": (-0.02, -0.03, -0.005)}}
    assert passes(ok)
    assert not passes({"all": {**ok["all"], "Rz vs R0-rand": (-0.01, -0.03, 0.001)}})
    assert not passes({"all": {**ok["all"], "Rz vs base": (-0.01, -0.02, 0.0)}})


def test_synoptic_runs_fingerprint_the_z500_extract():
    import training.train_unified as tu

    assert fr_mod.Z500_PATH in tu._data_files({"labels": "v2", "retrieval": {"mode": "synoptic", "k": 5}})
    assert fr_mod.Z500_PATH not in tu._data_files({"labels": "v2", "retrieval": {"mode": "region", "k": 5}})
    assert "pipeline/download_era5_z500.py" in tu.CODE_FILES and "synoptic" in fr_mod.MODES
