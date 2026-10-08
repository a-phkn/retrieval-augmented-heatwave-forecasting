"""evaluation/calibration_check.py: recalibration, the event threshold and the Brier score."""
import numpy as np
import pandas as pd
import pytest

import evaluation.calibration_check as cc
from training.folds import build_fold


def _frame(pred, actual, seeds=(0,), leads=(1,)):
    n = len(pred)
    return pd.DataFrame({"seed": np.repeat(seeds, n * len(leads)) if len(seeds) > 1 else seeds[0],
                         "lead": np.tile(np.repeat(leads, n), len(seeds)), "pred": np.tile(pred, len(seeds) * len(leads)),
                         "actual": np.tile(actual, len(seeds) * len(leads))})


def test_recalibration_removes_a_linear_distortion():
    rng = np.random.default_rng(0)
    truth = rng.normal(30, 3, 500)
    inner = _frame(0.8 * truth + 9.0, truth)  # forecasts shrunk and shifted
    coef = cc.fit_recal(inner)
    assert coef.loc[0, "b"] == pytest.approx(1.25) and coef.loc[0, "a"] == pytest.approx(-11.25)
    out = cc.apply_recal(_frame(0.8 * truth[:50] + 9.0, truth[:50]), coef)
    np.testing.assert_allclose(out["error"], 0.0, atol=1e-9)
    with pytest.raises(ValueError, match="coefficients"):
        cc.apply_recal(_frame(truth[:5], truth[:5], leads=(2,)), coef)


@pytest.mark.parametrize("family,target,labels", [("Tmax", "t_max", "v2"), ("WBGT (physical)", "wbgt_lj_max", "wbgt")])
def test_threshold_rule_reproduces_the_hot_labels(family, target, labels):
    val = build_fold("f2", target, labels).val
    dates = pd.DatetimeIndex(val.query_dates)
    tdates = pd.Series(np.repeat(dates.to_numpy(), 5) + np.tile(np.arange(5), len(dates)).astype("timedelta64[D]"))
    thr = cc.event_thresholds("f2", family, tdates)
    o = val.y_raw.reshape(-1) >= thr
    assert np.array_equal(o, val.hot.reshape(-1)) and o.sum() > 50


def test_brier_and_reliability_on_a_toy_case():
    df = pd.DataFrame({"p": [0.0, 0.0, 1.0, 0.5], "o": [0, 0, 1, 1.0], "p_clim": [0.25] * 4})
    b = cc.brier(df)
    assert b["brier"] == pytest.approx(0.0625) and b["brier_clim"] == pytest.approx((2 * 0.0625 + 2 * 0.5625) / 4)
    assert b["bss"] > 0 and b["n_events"] == 2 and sum(r["n"] for r in b["reliability"]) == 4
