"""models/retrieval_lstm_v2.py and the retrieval path of training/train_unified.py."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

import training.train_unified as tu
from models.retrieval_lstm_v2 import RetrievalAugmentedLSTMv2
from scripts.make_manifest import REPO_ROOT

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _inputs(b=3, k=4, f=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, 14, f, generator=g), torch.randn(b, k, 14, f, generator=g),
            torch.randn(b, k, 5, generator=g))


def test_no_analogue_means_zero_context():
    torch.manual_seed(0)
    model = RetrievalAugmentedLSTMv2(n_features=6).eval()
    xq, xa, ya = _inputs()
    mask = torch.tensor([[True, True, False, False], [False] * 4, [True] * 4])
    with torch.no_grad():
        pred, w = model(xq, xa, ya, mask)
        expected = model.head(torch.cat([model.encode(xq[1:2]), torch.zeros(1, 64)], dim=-1))
    assert torch.allclose(pred[1:2], expected)  # zero context: no learned constant sneaks in
    assert torch.all(w[1] == 0)
    assert torch.allclose(w[[0, 2]].sum(dim=1), torch.ones(2))
    assert torch.all(w[0, 2:] == 0)


def test_padded_slots_do_not_affect_the_forecast():
    torch.manual_seed(0)
    model = RetrievalAugmentedLSTMv2(n_features=6).eval()
    xq, xa, ya = _inputs()
    mask = torch.tensor([[True, True, False, False]] * 3)
    xa2, ya2 = xa.clone(), ya.clone()
    xa2[:, 2:] += 50.0
    ya2[:, 2:] -= 50.0
    with torch.no_grad():
        assert torch.allclose(model(xq, xa, ya, mask)[0], model(xq, xa2, ya2, mask)[0])


def test_retrieval_config_validation(tmp_path):
    cfg = json.loads((REPO_ROOT / "configs/A2Lr_hw5_R0.json").read_text())
    assert tu.load_config(REPO_ROOT / "configs/A2Lr_hw5_R0.json")["retrieval"] == {"mode": "sim", "k": 5}
    bad = tmp_path / "c.json"
    for r in ({"mode": "nearest", "k": 5}, {"mode": "sim", "k": 0}, {"mode": "sim"}, {"mode": "sim", "k": 5, "x": 1}):
        bad.write_text(json.dumps({**cfg, "retrieval": r}))
        with pytest.raises(ValueError, match="retrieval"):
            tu.load_config(bad)


def test_retrieval_configs_change_one_thing_from_their_parent():
    """R0 / R0-rand = control + retrieval; R1 = R0 with the calendar window."""
    for control in ("A1prime_hw5", "A2Lr_hw5"):
        base = tu.load_config(REPO_ROOT / f"configs/{control}.json")
        for rung, parent, mode in (("R0", control, "sim"), ("R0rand", control, "rand"), ("R1", f"{control}_R0", "time")):
            cfg = tu.load_config(REPO_ROOT / f"configs/{control}_{rung}.json")
            assert cfg["parent"] == parent and cfg["retrieval"] == {"mode": mode, "k": 5}
            same = {k for k in base if k not in ("run_id", "parent", "description")}
            assert all(cfg[k] == base[k] for k in same), rung


def test_smoke_retrieval_run_writes_predictions_analogues_and_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(tu, "MAX_EPOCHS", 2)
    monkeypatch.setattr(tu, "REGISTRY", tmp_path / "runs.csv")
    cfg = tu.load_config(REPO_ROOT / "configs/A1prime_hw5_R0rand.json")
    res = tu.run(cfg, folds=["f1"], seeds=[0, 1], out_dir=tmp_path / "p", model_dir=tmp_path / "m")
    df = res["f1"]
    assert df["target_date"].max() <= pd.Timestamp("2009-12-31") and np.isfinite(df["pred"]).all()
    an = pd.read_parquet(tmp_path / "p" / cfg["run_id"] / "f1_analogues.parquet")
    n_val = df.groupby("seed")["query_date"].nunique().iloc[0]
    assert len(an) == 2 * n_val * 5
    assert (an["analogue_query_date"] <= pd.Timestamp("2006-12-27")).all()  # fold f1 training windows only
    att = an.groupby(["seed", "query_date"])["attention"].sum()
    assert np.allclose(att, 1.0, atol=1e-5)
    s0 = an[an["seed"] == 0].set_index(["query_date", "rank"])["analogue_query_date"]
    s1 = an[an["seed"] == 1].set_index(["query_date", "rank"])["analogue_query_date"]
    assert (s0 != s1).mean() > 0.5  # random analogues are drawn per seed
    row = pd.read_csv(tmp_path / "runs.csv").iloc[0]
    assert row["retrieval"] == "rand k=5" and row["run_id"] == cfg["run_id"]
