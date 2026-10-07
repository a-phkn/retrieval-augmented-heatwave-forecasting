"""training/graph_data.py and the graph / upstream backbones in training/train_unified.py
(pre-registered 2026-10-07). Pre-2019 data only."""
import json

import numpy as np
import pandas as pd
import pytest

import training.train_unified as tu
from pipeline.climatology import apply_climatology, doy_climatology
from pipeline.graph import advective_weights, node_coords
from training import graph_data as gd
from training.folds import FORECAST_DAYS, INPUT_DAYS, fold_bounds

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def up():
    return gd.load_upstream()


@pytest.fixture(scope="module")
def f4(up):
    return gd.build_upstream("f4", up)


def test_wind_direction_is_meteorological_from():
    speed, d = gd.wind_from_uv(np.array([0.0, -5.0, 3.0]), np.array([-5.0, 0.0, 4.0]))
    assert np.allclose(speed, [5.0, 5.0, 5.0])
    assert np.allclose(d, [0.0, 90.0, 216.8698976], atol=1e-6)  # from N, from E, from SW


def test_inputs_are_aligned_by_date_and_scaled_on_training_years_only(up, f4):
    train_end, _, _ = fold_bounds("f4")
    train = up.index <= train_end
    node, var = 5, "dew_point_2m_mean"
    raw = up[f"{var}__{gd.NODE_IDS[node]}"].to_numpy()
    expected = (raw - raw[train].mean()) / raw[train].std()
    assert np.allclose(f4.x[:, node, gd.UPSTREAM_VARS.index(var)], expected, atol=1e-5)
    tmax = up[f"temperature_2m_max__{gd.NODE_IDS[node]}"]
    m, s = apply_climatology(up.index, doy_climatology(tmax, train))
    anom = (tmax.to_numpy() - m) / s
    assert np.allclose(f4.x[:, node, -1], (anom - anom[train].mean()) / anom[train].std(), atol=1e-5)


def test_window_positions_are_the_14_input_days_only(f4):
    q = pd.DatetimeIndex(["2010-06-01", "2015-05-20"])
    pos = f4.positions(q)
    assert pos.shape == (2, INPUT_DAYS)
    days = f4.dates.values[pos]
    assert (days[:, -1] == (q - pd.Timedelta(days=1)).values).all()
    assert (days[:, 0] == (q - pd.Timedelta(days=INPUT_DAYS)).values).all()
    assert (days < q.values[:, None]).all()  # no forecast day


def test_no_leakage_from_after_the_training_years(up):
    """Changing every upstream value after train_end must not change any training-year input
    or edge (scaling, climatology and edges use training years / the day's own wind)."""
    train_end, _, _ = fold_bounds("f2")
    before = gd.build_upstream("f2", up)
    altered = up.copy()
    altered.loc[altered.index > train_end] += 5.0
    after = gd.build_upstream("f2", altered)
    train = before.dates <= train_end
    assert np.array_equal(before.x[train], after.x[train]) and np.array_equal(before.adj[train], after.adj[train])
    assert not np.allclose(before.x[~train], after.x[~train])  # the change took effect


def test_edges_follow_each_days_wind_and_stay_local(up, f4):
    day = 5000
    u = np.array([up[f"wind_u_10m__{n}"].iloc[day] for n in gd.NODE_IDS])
    v = np.array([up[f"wind_v_10m__{n}"].iloc[day] for n in gd.NODE_IDS])
    nb = f4.mask[0, 1:]
    u, v = np.concatenate([[u[nb].mean()], u]), np.concatenate([[v[nb].mean()], v])
    speed, direction = gd.wind_from_uv(u, v)
    assert np.allclose(f4.adj[day], advective_weights(speed, direction, node_coords()), atol=1e-6)
    assert not (f4.adj[:, ~f4.mask] != 0).any() and nb.sum() == 8
    assert (f4.static_adj[~f4.mask] == 0).all()


def test_flattened_upstream_inputs_sit_next_to_delhis(f4):
    from training.folds import build_fold

    data = build_fold("f4", "t_max", "v2")
    aug = tu.with_upstream(data.val, f4)
    n_delhi = data.val.X.shape[2]
    assert aug.X.shape == (*data.val.X.shape[:2], n_delhi + 27 * gd.N_UP_FEATURES)
    assert np.array_equal(aug.X[:, :, :n_delhi], data.val.X)
    i = 7
    assert np.array_equal(aug.X[i, :, n_delhi:].reshape(INPUT_DAYS, 27, -1), f4.x[f4.positions(data.val.query_dates)[i]])


def _cfg(**extra):
    base = json.loads((tu.REPO_ROOT / "configs" / "A1prime_hw5.json").read_text(encoding="utf-8"))
    return {**base, "run_id": "test_graph", **extra}


def test_config_validation_for_backbones(tmp_path):
    def load(cfg):
        path = tmp_path / "c.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return tu.load_config(path)

    assert "backbone" not in load(_cfg())  # existing configs keep their hash
    load(_cfg(backbone="dstgnn", graph={"mode": "dynamic", "adaptive": True}))
    load(_cfg(backbone="lstm_upstream"))
    for bad in (_cfg(backbone="gnn"), _cfg(backbone="dstgnn"), _cfg(backbone="dstgnn", graph={"mode": "static", "adaptive": True}),
                _cfg(backbone="lstm_upstream", graph={"mode": "none", "adaptive": False}),
                _cfg(backbone="dstgnn", graph={"mode": "none", "adaptive": False}, retrieval={"mode": "sim", "k": 5})):
        with pytest.raises(ValueError):
            load(bad)


@pytest.mark.parametrize("extra", [{"backbone": "dstgnn", "graph": {"mode": "dynamic", "adaptive": True}},
                                   {"backbone": "dstgnn", "graph": {"mode": "static", "adaptive": False}},
                                   {"backbone": "lstm_upstream"}])
def test_backbones_train_end_to_end(tmp_path, monkeypatch, extra):
    """One epoch on one fold and seed: predictions on every validation window, a G-D2 log for
    the graph, nothing written to the real registry."""
    monkeypatch.setattr(tu, "MAX_EPOCHS", 1)
    cfg = _cfg(**extra)
    res = tu.run(cfg, folds=["f1"], seeds=[0], out_dir=tmp_path / "pred", model_dir=tmp_path / "models",
                 write_registry=False, threads=2)
    df = res["f1"]
    assert np.isfinite(df["pred"]).all() and df["lead"].nunique() == FORECAST_DAYS
    log = tmp_path / "pred" / "test_graph" / "f1_training_log.json"
    if extra["backbone"] == "dstgnn":
        entry = json.loads(log.read_text(encoding="utf-8"))[0]
        assert entry["finite"] and len(entry["val_curve"]) == 1
    else:
        assert not log.exists()
