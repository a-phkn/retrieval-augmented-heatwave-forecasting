"""pipeline/build_peak_ingredients.py: each cell's ingredients at its own WBGT peak hour
reproduce the forecast target exactly (physics head design D3, decision 2026-10-07)."""
import numpy as np
import pandas as pd
import pytest
import torch

from pipeline import build_peak_ingredients as bpi
from pipeline.wbgt_liljegren import wbgt as wbgt_np
from pipeline.wbgt_liljegren_torch import wbgt as wbgt_torch
from training.folds import TEST_START

pytestmark = pytest.mark.skipif(not bpi.OUT.exists(), reason="build with python -m pipeline.build_peak_ingredients")


@pytest.fixture(scope="module")
def pk():
    df = pd.read_parquet(bpi.OUT)
    return df[df.index < TEST_START]  # pre-2019 only


def test_cell_mean_of_peak_wbgt_is_the_forecast_target(pk):
    target = pd.read_parquet(bpi.OUT_DIR / "wbgt_liljegren_daily.parquet").set_index("date")["wbgt_lj_max"]
    mean = pk[[f"wbgt__c{c}" for c in bpi.CELLS]].mean(axis=1)
    common = mean.index.intersection(target.index)
    assert len(common) == len(mean) > 14_000
    assert np.abs(mean[common] - target[common]).max() < 1e-9


def test_exact_formula_on_stored_ingredients_returns_the_peak_wbgt(pk):
    s = pk.sample(1500, random_state=0)
    for c in (1, 5, 9):
        g = {k: s[f"{k}__c{c}"].to_numpy() for k in (*bpi.INGREDIENTS, "wbgt")}
        out = wbgt_np(g["t"], g["rh"], g["pressure"], g["wind"], g["ghi"], g["fdir"] * g["ghi"], g["cosz"])["wbgt"]
        assert np.abs(out - g["wbgt"]).max() < 1e-6
        t = {k: torch.tensor(v, dtype=torch.float64) for k, v in g.items()}
        out_t = wbgt_torch(t["t"], t["rh"], t["pressure"], t["wind"], t["ghi"], t["fdir"], t["cosz"])["wbgt"]
        assert np.abs(out_t.numpy() - g["wbgt"]).max() < 2e-4


def test_peak_rows_picks_the_peak_hour_and_drops_incomplete_days():
    times = pd.date_range("2000-06-01", periods=24 + 10, freq="h")
    wb = np.r_[np.arange(24.0), np.arange(10.0)]
    wb[13] = 99.0
    h = pd.DataFrame({"time": times, "wbgt": wb, "t": wb, "rh": 50.0, "pressure": 980.0, "wind": 2.0,
                      "ghi": 500.0, "diffuse": 100.0, "cosz": 0.5})
    out = bpi.peak_rows(h)
    assert list(out.index) == [pd.Timestamp("2000-06-01")]  # 2000-06-02 has only 10 hours
    row = out.iloc[0]
    assert row["hour"] == 13 and row["wbgt"] == 99.0 and row["fdir"] == pytest.approx(0.8)
