"""evaluation/gd0_upstream_signal.py: input layout, ridge, inner alpha choice and leakage
guards for gate G-D0. Uses the built upstream dataset (skipped if it is absent)."""
import numpy as np
import pandas as pd
import pytest

import evaluation.gd0_upstream_signal as g
from training.folds import TEST_START, fold_bounds

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"),
              pytest.mark.skipif(not g.UPSTREAM_PATH.exists(), reason="upstream dataset not built")]


def test_design_layout_is_delhi_then_node_major_lags():
    idx = pd.date_range("2000-01-01", periods=30)
    dz = pd.Series(np.arange(30.0), index=idx)
    uz = pd.DataFrame({"a": 100 + np.arange(30.0), "b": 200 + np.arange(30.0)}, index=idx)
    X = g.design([idx[10]], dz, uz, ["a", "b"])
    # q = day 10: Delhi z(q-1)=9; a at q-1,q-2,q-3 = 109,108,107; b = 209,208,207
    assert X.tolist() == [[9, 109, 108, 107, 209, 208, 207]]
    assert g.targets([idx[10]], dz).tolist() == [[10, 11, 12, 13, 14]]
    with pytest.raises(ValueError, match="too early"):
        g.design([idx[2]], dz, uz, ["a"])


def test_ridge_recovers_a_linear_signal_and_shrinks_with_alpha():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = (X @ np.array([1.0, -2.0, 0.5]) + 3.0)[:, None]
    c, b = g.ridge(X, y, 1e-6)
    assert np.allclose(c[:, 0], [1.0, -2.0, 0.5], atol=1e-4) and np.isclose(b[0], 3.0, atol=1e-4)
    big, _ = g.ridge(X, y, 1e6)
    assert np.abs(big).max() < 0.01


def test_upstream_data_is_read_without_the_test_period():
    up = g.load_upstream_tmax()
    assert up.index.max() < TEST_START and up.shape[1] == 27 and not up.isna().any().any()


def test_no_leakage_from_after_the_training_years():
    """Changing upstream values after train_end must not change any fitted coefficient or
    chosen ridge strength (climatology, anomalies and fits use training years only)."""
    up = g.load_upstream_tmax()
    train_end, _, _ = fold_bounds("f2")
    a = g.fold_results("f2", "t_max", "v2", up)
    up2 = up.copy()
    up2.loc[up2.index > train_end] += 6.0
    b = g.fold_results("f2", "t_max", "v2", up2)
    for name in ("upstream_all", "node_n28e074"):
        assert np.array_equal(a["coef"][name], b["coef"][name]) and np.array_equal(a["alpha"][name], b["alpha"][name])
    # damped persistence does not use upstream data at all
    assert a["frames"]["damped_persistence"].equals(b["frames"]["damped_persistence"])


def test_forecasts_sit_on_the_scored_windows():
    up = g.load_upstream_tmax()
    r = g.fold_results("f1", "t_max", "v2", up)
    dp, aug = r["frames"]["damped_persistence"], r["frames"]["upstream_all"]
    assert len(dp) == len(aug) and dp["actual"].equals(aug["actual"]) and dp["stratum"].equals(aug["stratum"])
    assert aug["target_date"].between("2007-01-01", "2009-12-31").all() and np.isfinite(aug["pred"]).all()
    assert set(r["alpha"]["upstream_all"]) <= set(g.ALPHAS)
