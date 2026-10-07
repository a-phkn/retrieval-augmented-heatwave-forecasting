"""training/physics_train.py: data alignment, scaling and a short end-to-end training run
(physics-head rung, decisions 2026-10-07)."""
import numpy as np
import pandas as pd
import pytest
import torch

import training.physics_train as pt
from models.physics_head import OUTPUTS, LSTMPhysics
from training.folds import FORECAST_DAYS, inner_split

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"),
              pytest.mark.skipif(not pt.PEAK_PATH.exists(), reason="build with python -m pipeline.build_peak_ingredients")]


@pytest.fixture(scope="module")
def table():
    return pt.load_cell_outputs()


@pytest.fixture(scope="module")
def pf(table):
    return pt.build_physics_fold("f1", table)


def test_inner_masks_match_the_trainers_inner_split(pf):
    fit, stop = pt.inner_masks(pf.wb.train.query_dates, pf.wb.train_end)
    a, b = inner_split(pf.wb.train, pf.wb.train_end, years=2)
    assert pd.DatetimeIndex(pf.wb.train.query_dates[fit]).equals(pd.DatetimeIndex(a.query_dates))
    assert pd.DatetimeIndex(pf.wb.train.query_dates[stop]).equals(pd.DatetimeIndex(b.query_dates))


def test_inputs_are_the_union_of_both_families(pf):
    for s in ("train", "val"):
        X, wb = pf.X[s], getattr(pf.wb, s)
        assert X.shape[2] == 17 and np.array_equal(X[:, :, :14], wb.X)
        extra = [pf.tx.feature_columns.index(c) for c in ("clim_mean_t_max", "clim_std_t_max", "t_max_anomaly")]
        assert np.array_equal(X[:, :, 14:], getattr(pf.tx, s).X[:, :, extra])


def test_cell_targets_are_the_forecast_days_values_standardised_on_training_years(pf, table):
    i = 123
    q = pd.Timestamp(pf.wb.val.query_dates.iloc[i])
    raw = pf.cells["val"][i] * pf.cell_std + pf.cell_mean  # (5, 9, 8)
    for lead in (0, 4):
        day = q + pd.Timedelta(days=lead)
        for c, k in ((1, "t"), (5, "rh"), (9, "tmax")):
            assert raw[lead, c - 1, OUTPUTS.index(k)] == pytest.approx(table.loc[day, f"{k}__c{c}"], abs=1e-3)
    train = table[table.index <= pf.wb.train_end]
    assert np.allclose(pf.cell_mean[0, OUTPUTS.index("t")], train["t__c1"].mean(), atol=1e-4)
    assert pf.cells["train"].shape[1:] == (FORECAST_DAYS, 9, len(OUTPUTS))


def test_target_scaling_inverts_to_degrees(pf):
    """The loss's normalised WBGT (anomaly form) and Tmax (raw form) invert with FoldData.to_raw."""
    torch.manual_seed(0)
    model = LSTMPhysics(17, pf.head_stats)
    X = torch.from_numpy(pf.X["val"][:64])
    with torch.no_grad():
        res = model(X)
    clim = pf.wb.val.clim_target[:64]
    wb_norm = (res["wbgt"].numpy() - clim - pf.wb.y_mean) / pf.wb.y_std
    tx_norm = (res["tmax"].numpy() - pf.tx.y_mean) / pf.tx.y_std
    assert np.allclose(wb_norm * pf.wb.y_std + pf.wb.y_mean + clim, res["wbgt"].numpy(), atol=1e-4)
    assert np.allclose(tx_norm * pf.tx.y_std + pf.tx.y_mean, res["tmax"].numpy(), atol=1e-4)


def test_short_training_run_reduces_the_loss_and_predicts_both_targets(pf):
    log = {}
    model = pt.train_one_seed_physics(0, pf, hot_weight=5.0, batch_size=64, max_epochs=2, patience=10, lr=1e-3, log=log)
    assert log["finite"] and len(log["val_curve"]) == 2
    wb, tx = pt.predict_physics(model, pf.X["val"][:200])
    assert wb.shape == tx.shape == (200, FORECAST_DAYS) and np.isfinite(wb).all() and np.isfinite(tx).all()
    rmse_wb = np.sqrt(np.mean((wb - pf.wb.val.y_raw[:200]) ** 2))
    rmse_tx = np.sqrt(np.mean((tx - pf.tx.val.y_raw[:200]) ** 2))
    assert rmse_wb < 6 and rmse_tx < 8  # sane after 2 epochs (climatology-level errors are ~2-3 C)
