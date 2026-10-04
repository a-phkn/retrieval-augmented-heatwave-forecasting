"""training/folds.py: fold boundaries, leakage guards, and exact v1 reproduction."""
import numpy as np
import pandas as pd
import pytest

from training.folds import FOLDS, TEST_START, build_fold, fold_bounds, fold_daily, fold_windows

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.parametrize("fold", list(FOLDS))
def test_fold_windows_respect_block_boundaries(fold):
    train_end, val_start, val_end = fold_bounds(fold)
    tr, va = fold_windows(fold)
    assert (tr["forecast_end"] <= train_end).all() and (tr["query_date"] <= train_end).all()
    assert (va["query_date"] >= val_start).all() and (va["forecast_end"] <= val_end).all()
    assert (va["forecast_end"] < TEST_START).all()
    assert len(va) in (1091, 1092)  # 3-year block minus windows crossing its end


def test_primary_fold_window_counts_match_v1():
    tr, va = fold_windows("f4")
    assert (len(tr), len(va)) == (13131, 1092)


def test_test_period_is_never_loaded():
    d, _ = fold_daily("f4", "t_max", "v2")
    assert d.index.max() < TEST_START


def test_primary_fold_reproduces_v1_arrays_bit_for_bit():
    """v1's own functions, called for train and val only (the test split is never built)."""
    from training.data import (
        _load_daily,
        _load_windows,
        build_split_arrays,
        build_split_target_climatology,
        build_split_target_stratum,
        load_normalization_stats,
        normalize_X,
        normalize_y,
    )

    f = build_fold("f4", "t_max", "v1", False)
    daily, windows, stats = _load_daily(), _load_windows(), load_normalization_stats()
    assert np.array_equal(f.feature_mean, stats.feature_mean) and np.array_equal(f.feature_std, stats.feature_std)
    for split in ("train", "val"):
        X_raw, y_raw, _ = build_split_arrays(split, daily, windows)
        a = getattr(f, split)
        assert np.array_equal(a.X, normalize_X(X_raw, stats)) and np.array_equal(a.y, normalize_y(y_raw, stats))
        assert np.array_equal(a.y_raw, y_raw)
        assert np.array_equal(a.hot, build_split_target_climatology(split, daily, windows)[2])
        assert np.array_equal(a.stratum, build_split_target_stratum(split, daily, windows))


def _perturbed_build(monkeypatch, fold, target, labels, anomaly, after, delta=7.0):
    """build_fold with every weather value after `after` shifted by delta."""
    import training.folds as tf

    original = tf._load_daily_pre_test

    def altered():
        d = original().copy()
        cols = d.select_dtypes("number").columns.drop(["doy_sin", "doy_cos", "years_since_1980"])
        d.loc[d.index > after, cols] += delta
        return d

    monkeypatch.setattr(tf, "_load_daily_pre_test", altered)
    try:
        return build_fold(fold, target, labels, anomaly)
    finally:
        monkeypatch.setattr(tf, "_load_daily_pre_test", original)


ARRAY_FIELDS = ("X", "y", "y_raw", "clim_target", "hot", "stratum", "persist")


@pytest.mark.parametrize("fold", list(FOLDS))
@pytest.mark.parametrize("target, labels, anomaly", [("t_max", "v2", False), ("wbgt_bom_max", "v2", True)])
def test_no_leakage_from_after_the_training_years(monkeypatch, fold, target, labels, anomaly):
    """Changing every value after train_end must leave normalisation, target scaling and all
    training arrays bit-identical (climatology, labels, hot mask and inputs are train-only)."""
    train_end, _, val_end = fold_bounds(fold)
    a = build_fold(fold, target, labels, anomaly)
    b = _perturbed_build(monkeypatch, fold, target, labels, anomaly, after=train_end)
    assert np.array_equal(a.feature_mean, b.feature_mean) and np.array_equal(a.feature_std, b.feature_std)
    assert (a.y_mean, a.y_std) == (b.y_mean, b.y_std)
    for k in ARRAY_FIELDS:
        assert np.array_equal(getattr(a.train, k), getattr(b.train, k)), k
    if fold != "f4":  # data after the val block must not change the val arrays either
        c = _perturbed_build(monkeypatch, fold, target, labels, anomaly, after=val_end)
        for k in ARRAY_FIELDS:
            assert np.array_equal(getattr(a.val, k), getattr(c.val, k)), k


def test_wbgt_target_inputs_and_anomaly_round_trip():
    f = build_fold("f2", "wbgt_bom_max", "v2", anomaly_target=True)
    assert f.feature_columns[-4:] == ["wbgt_bom_max", "clim_mean_wbgt_bom_max", "clim_std_wbgt_bom_max", "wbgt_bom_max_anomaly"]
    assert f.train.X.shape[2] == 14
    # to_raw(normalised model target) must give back the raw target.
    assert np.allclose(f.to_raw(f.val.y.astype(np.float64), "val"), f.val.y_raw, atol=1e-4)


def test_v2_labels_have_extreme_days_in_every_fold():
    for fold in FOLDS:
        f = build_fold(fold, "t_max", "v2")
        assert (f.val.stratum == "extreme").sum() > 0
        assert not f.val.hot[np.asarray(pd.DatetimeIndex(f.val.query_dates).month.isin([1, 2]))].any()


def test_episode_count_rule_with_each_folds_own_climatology():
    """Pre-registered minimum-sample rule (>= 25 episodes), counted as the folds see them:
    each validation block labelled with ITS fold's training-only climatology."""
    from pipeline.labels_v2 import label_frame

    total = 0
    for fold in FOLDS:
        d, _ = fold_daily(fold, "t_max", "v2")
        lab = label_frame(d.index, d["t_max"], d["t_max_anomaly"])
        _, val_start, val_end = fold_bounds(fold)
        ids = lab.loc[val_start:val_end, "episode_id_v2"]
        total += ids[ids > 0].nunique()
    assert total >= 25


def test_unknown_inputs_are_rejected():
    with pytest.raises(ValueError):
        fold_bounds("f9")
    with pytest.raises(ValueError):
        fold_daily("f1", "hi_max", "v2")
    with pytest.raises(ValueError):
        fold_daily("f1", "t_max", "v3")
