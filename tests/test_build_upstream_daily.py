"""pipeline/build_upstream_daily.py: cleaning rules, wind convention, fingerprint, and the
built dataset (when present). Raw data is gitignored, so raw-file tests use small fakes."""
import json

import numpy as np
import pandas as pd
import pytest

import pipeline.build_upstream_daily as bu
from pipeline.download_era5_upstream import DAILY_VARIABLES, NODES
from scripts.make_manifest import REPO_ROOT


def _raw(direction, speed=5.0, soil=0.1):
    idx = pd.date_range("2000-01-01", periods=len(direction))
    data = {v: np.full(len(idx), 1.0) for v in DAILY_VARIABLES}
    data["wind_direction_10m_dominant"] = np.asarray(direction, dtype=float)
    data["wind_speed_10m_mean"] = np.full(len(idx), speed)
    data["soil_moisture_0_to_7cm_mean"] = np.asarray(soil, dtype=float) * np.ones(len(idx))
    return pd.DataFrame(data, index=idx)


def test_wind_components_follow_the_meteorological_convention():
    out = bu.clean_node(_raw([0.0, 90.0, 180.0, 270.0]))
    # from N -> blows south (v<0); from E -> blows west (u<0); from S -> north; from W -> east
    assert np.allclose(out["wind_u_10m"], [0, -5, 0, 5], atol=1e-9)
    assert np.allclose(out["wind_v_10m"], [-5, 0, 5, 0], atol=1e-9)
    near = bu.clean_node(_raw([359.0, 1.0]))  # almost the same direction -> almost the same components
    assert np.allclose(near["wind_u_10m"].iloc[0], near["wind_u_10m"].iloc[1], atol=0.2)
    assert np.allclose(near["wind_v_10m"].iloc[0], near["wind_v_10m"].iloc[1], atol=1e-3)


def test_negative_soil_moisture_is_clipped_and_nothing_else_changes():
    raw = _raw([10.0, 20.0, 30.0], soil=[-0.002, 0.0, 0.25])
    out = bu.clean_node(raw)
    assert out["soil_moisture_0_to_7cm_mean"].tolist() == [0.0, 0.0, 0.25]
    assert out[DAILY_VARIABLES].drop(columns="soil_moisture_0_to_7cm_mean").equals(
        raw.drop(columns="soil_moisture_0_to_7cm_mean"))


def test_raw_fingerprint_changes_when_any_file_changes(tmp_path):
    (tmp_path / "n24e068").mkdir()
    f = tmp_path / "n24e068" / "1980.json"
    f.write_text(json.dumps({"a": 1}))
    a = bu.raw_fingerprint(tmp_path)
    f.write_text(json.dumps({"a": 2}))
    assert bu.raw_fingerprint(tmp_path) != a


@pytest.mark.skipif(not bu.OUT.exists(), reason="dataset not built")
def test_built_dataset_is_complete_and_matches_its_metadata():
    t = pd.read_parquet(bu.OUT)
    meta = json.loads(bu.META.read_text(encoding="utf-8"))
    assert t.index[0] == pd.Timestamp("1980-01-01") and t.index[-1] == pd.Timestamp(meta["last_date"])
    assert len(t) == meta["n_days"] and t.index.is_monotonic_increasing and not t.index.duplicated().any()
    assert t.shape[1] == len(NODES) * len(meta["variables"]) and not t.isna().any().any()
    assert (t.filter(like="soil_moisture").to_numpy() >= 0).all()
    assert t.filter(like="relative_humidity").to_numpy().max() <= 100
    if bu.UPSTREAM_DIR.exists():  # raw data is local only (gitignored)
        assert bu.raw_fingerprint() == meta["raw_sha256"]
