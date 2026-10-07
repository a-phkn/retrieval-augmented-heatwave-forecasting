"""
Training the physics-head rung (decisions 2026-10-07: exact Liljegren formula, per cell at
each cell's own peak hour (D3); outputs WBGT and Tmax; loss = WBGT + Tmax + 0.1 x ingredients;
kept unless worse than the direct models by more than 0.05 °C).

One model forecasts both targets, so it is built from the two control families' fold data
for the SAME windows:
  WBGT  the A2Lr_hw5 recipe: physical WBGT, WBGT label, anomaly target form;
  Tmax  the A1prime_hw5 recipe: Tmax, v2 labels, raw target form.
Inputs: the union of the two families' input columns (the WBGT family's 14 plus the Tmax
family's 3 Tmax-climatology channels = 17), normalised exactly as in each family.

Loss per batch (hot-weighted MSE as training/train_unified.py, each in its own normalised
units, hot_weight from the config):
    MSE_w(WBGT) + MSE_w(Tmax) + 0.1 x mean over (lead, cell, output) of the squared error of the
    8 per-cell outputs (7 peak-hour ingredients + daily Tmax), each standardised by its
    fold-training-years mean and SD per cell.
Early stopping: the same total loss on the inner 2-year stop block (as inner_2y), same
patience, epochs, batch size and learning rate as the LSTM recipe.

Ingredient targets: datasets_v2/peak_ingredients_daily.parquet (7 per cell) and the cell's daily
Tmax from datasets_v2/per_cell_daily.parquet, for each window's 5 forecast days. 2019+ rows are
dropped on read; statistics use the fold's training years only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.physics_head import OUTPUTS, LSTMPhysics
from pipeline.build_peak_ingredients import CELLS, OUT as PEAK_PATH
from training.folds import FORECAST_DAYS, TEST_START, FoldData, build_fold

PER_CELL_PATH = PEAK_PATH.parent / "per_cell_daily.parquet"
INGREDIENT_WEIGHT = 0.1
WBGT_FAMILY = ("wbgt_lj_max", "wbgt", "anomaly")  # A2Lr_hw5
TMAX_FAMILY = ("t_max", "v2", "raw")  # A1prime_hw5
INNER_YEARS = 2


def load_cell_outputs() -> pd.DataFrame:
    """(dates, 9 x 8) per-cell targets of the head's outputs, pre-2019 rows only: columns
    `<output>__c<k>` for OUTPUTS (7 peak-hour ingredients + 'tmax')."""
    pk = pd.read_parquet(PEAK_PATH)
    pk = pk[pk.index < TEST_START]
    pc = pd.read_parquet(PER_CELL_PATH, columns=["date", "cell", "t_max"], filters=[("date", "<", TEST_START)])
    tmax = pc.pivot(index="date", columns="cell", values="t_max")
    out = pk[[f"{k}__c{c}" for c in CELLS for k in OUTPUTS if k != "tmax"]].copy()
    for c in CELLS:
        out[f"tmax__c{c}"] = tmax[c].reindex(out.index)
    if out.isna().any().any():
        raise ValueError("per-cell targets missing for some days")
    return out


def inner_masks(query_dates, train_end: pd.Timestamp, years: int = INNER_YEARS) -> tuple[np.ndarray, np.ndarray]:
    """(fit, stop) masks over training windows, exactly as training.folds.inner_split."""
    stop_start = train_end - pd.DateOffset(years=years) + pd.Timedelta(days=1)
    q = pd.DatetimeIndex(query_dates)
    return np.asarray(q + pd.Timedelta(days=FORECAST_DAYS - 1) < stop_start), np.asarray(q >= stop_start)


@dataclass
class PhysicsFold:
    fold: str
    wb: FoldData  # WBGT family (A2Lr_hw5 recipe)
    tx: FoldData  # Tmax family (A1prime_hw5 recipe)
    X: dict[str, np.ndarray]  # "train" / "val": (N, 14, 17) union inputs
    cells: dict[str, np.ndarray]  # "train" / "val": (N, 5, 9, 8) standardised per-cell targets
    cell_mean: np.ndarray  # (9, 8)
    cell_std: np.ndarray  # (9, 8)
    head_stats: dict[str, torch.Tensor]


def build_physics_fold(fold: str, cell_table: pd.DataFrame | None = None) -> PhysicsFold:
    wb = build_fold(fold, WBGT_FAMILY[0], WBGT_FAMILY[1], target_form=WBGT_FAMILY[2])
    tx = build_fold(fold, TMAX_FAMILY[0], TMAX_FAMILY[1], target_form=TMAX_FAMILY[2])
    for s in ("train", "val"):
        if not pd.DatetimeIndex(getattr(wb, s).query_dates).equals(pd.DatetimeIndex(getattr(tx, s).query_dates)):
            raise ValueError(f"{fold} {s}: WBGT and Tmax windows differ")
    extra = [tx.feature_columns.index(c) for c in tx.feature_columns if c not in wb.feature_columns]
    X = {s: np.concatenate([getattr(wb, s).X, getattr(tx, s).X[:, :, extra]], axis=2) for s in ("train", "val")}

    table = load_cell_outputs() if cell_table is None else cell_table
    arr = table[[f"{k}__c{c}" for c in CELLS for k in OUTPUTS]].to_numpy(np.float64).reshape(len(table), len(CELLS), len(OUTPUTS))
    train_days = np.asarray(table.index <= wb.train_end)
    mean, std = arr[train_days].mean(axis=0), arr[train_days].std(axis=0)
    std[std == 0] = 1.0
    cells = {}
    for s in ("train", "val"):
        pos = table.index.get_indexer(pd.DatetimeIndex(getattr(wb, s).query_dates))
        if (pos < 0).any() or (pos + FORECAST_DAYS > len(table)).any():
            raise ValueError("per-cell targets missing for some windows")
        cells[s] = ((arr[pos[:, None] + np.arange(FORECAST_DAYS)] - mean) / std).astype(np.float32)
    t_i, p_i = OUTPUTS.index("t"), OUTPUTS.index("pressure")
    stats = {"t_mean": mean[:, t_i], "t_std": std[:, t_i], "p_mean": mean[:, p_i], "p_std": std[:, p_i]}
    return PhysicsFold(fold, wb, tx, X, cells, mean.astype(np.float32), std.astype(np.float32),
                       {k: torch.as_tensor(v, dtype=torch.float32) for k, v in stats.items()})


def _weighted_mse(pred, target, hot, hot_weight):  # as train_unified.weighted_mse
    return torch.mean((1.0 + (hot_weight - 1.0) * hot) * (pred - target) ** 2)


class _Batches:
    """Tensors of one split (or a subset) needed for the loss."""

    def __init__(self, pf: PhysicsFold, split: str, mask: np.ndarray | None = None):
        m = slice(None) if mask is None else mask
        wb, tx = getattr(pf.wb, split), getattr(pf.tx, split)
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a[m]))  # noqa: E731
        self.tensors = (f(pf.X[split]), f(wb.y), f(wb.hot.astype(np.float32)), f(wb.clim_target.astype(np.float32)),
                        f(tx.y), f(tx.hot.astype(np.float32)), f(pf.cells[split]))


def physics_loss(model: LSTMPhysics, batch, pf: PhysicsFold, hot_weight: float,
                 cell_mean: torch.Tensor, cell_std: torch.Tensor) -> torch.Tensor:
    x, y_wb, hot_wb, clim_wb, y_tx, hot_tx, cells = batch
    res = model(x)
    wb_norm = (res["wbgt"] - clim_wb - pf.wb.y_mean) / pf.wb.y_std  # anomaly target form
    tx_norm = (res["tmax"] - pf.tx.y_mean) / pf.tx.y_std  # raw target form
    pred_cells = (torch.stack([res["cells"][k] for k in OUTPUTS], dim=-1) - cell_mean) / cell_std
    return (_weighted_mse(wb_norm, y_wb, hot_wb, hot_weight) + _weighted_mse(tx_norm, y_tx, hot_tx, hot_weight)
            + INGREDIENT_WEIGHT * torch.mean((pred_cells - cells) ** 2))


def predict_physics(model: LSTMPhysics, X: np.ndarray, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """(WBGT (N, 5), Tmax (N, 5)) in deg C, eval mode."""
    model.eval()
    wb, tx = [], []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            res = model(torch.from_numpy(X[s:s + batch]))
            wb.append(res["wbgt"].numpy())
            tx.append(res["tmax"].numpy())
    return np.concatenate(wb), np.concatenate(tx)


def train_one_seed_physics(seed: int, pf: PhysicsFold, hot_weight: float, batch_size: int, max_epochs: int,
                           patience: int, lr: float, log: dict | None = None) -> LSTMPhysics:
    """Same recipe as train_unified.train_one_seed (seeding, Adam, batch, epochs, patience) with
    the physics loss; early stopping on that loss over the inner 2-year stop block."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    fit, stop = inner_masks(pf.wb.train.query_dates, pf.wb.train_end)
    loader = DataLoader(TensorDataset(*_Batches(pf, "train", fit).tensors), batch_size=batch_size, shuffle=True)
    stop_t = _Batches(pf, "train", stop).tensors
    cm, cs = torch.from_numpy(pf.cell_mean), torch.from_numpy(pf.cell_std)
    model = LSTMPhysics(n_features=pf.X["train"].shape[2], head_stats=pf.head_stats)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, stale, curve, finite = float("inf"), None, 0, [], True
    for _ in range(max_epochs):
        model.train()
        for b in loader:
            opt.zero_grad()
            loss = physics_loss(model, b, pf, hot_weight, cm, cs)
            finite &= bool(torch.isfinite(loss))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = sum(physics_loss(model, tuple(t[i:i + 512] for t in stop_t), pf, hot_weight, cm, cs).item()
                    * len(stop_t[0][i:i + 512]) for i in range(0, len(stop_t[0]), 512)) / len(stop_t[0])
        curve.append(v)
        if v < best - 1e-6:
            best, stale = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if log is not None:
        log.update(val_curve=curve, finite=finite)
    if best_state is None:
        raise RuntimeError("no finite early-stopping loss in any epoch")
    model.load_state_dict(best_state)
    return model
