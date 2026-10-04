"""
Rolling-origin folds and per-fold datasets for LSTM-type models (plan v5, Week 2).

Folds (validation blocks of 3 years; training = every year before the block):
    f1: val 2007-2009    f2: val 2010-2012    f3: val 2013-2015    f4: val 2016-2018
f4 is the original v1 split (train 1980-2015, val 2016-2018). The test period (2019+) is
dropped before anything else is computed, so no fold can read it.

Windows come only from the frozen splits/window_index_v1.parquet, assigned with the v1
convention (prepare_datasets.py): a window belongs to the block containing its query date,
and is dropped if its 5-day forecast would cross out of that block (training windows must
end by the training end date; validation windows by the block end). Inputs may reach back
into earlier years -- that is past information.

Everything that depends on which years are "training years" is recomputed PER FOLD from
that fold's training years only: day-of-year climatology (pipeline/climatology.py),
anomaly channels, heatwave labels and z-score normalisation.

Inputs: 10 base weather features + the history channels of the TARGET variable:
    target t_max        -> clim_mean_t_max, clim_std_t_max, t_max_anomaly  (= v1's 13 inputs)
    target wbgt_bom_max -> wbgt_bom_max, clim_mean_wbgt_bom_max, clim_std_wbgt_bom_max,
                           wbgt_bom_max_anomaly                             (14 inputs)
so swapping the target (A1' -> A2) swaps only the target and its own history.

Labels: heatwave days are ALWAYS defined from Tmax (official tiers are Tmax-based), for any
target, so labels cannot differ between A1' and A2:
    v1: hot = Tmax anomaly > 1.5 sigma (year-round), episodes span >= 3 days
    v2: pipeline/labels_v2.py (season-gated; Tmax >= 40 C & anomaly >= 3 C, or >= 45 C)
The hot mask weights the loss; the stratum (normal/unusual/extreme) is for evaluation.

Anomaly target (A2r): the model predicts target - climatology, normalised; predictions are
converted back by adding the climatology of each forecast day.

Early stopping (inner_split): v2 runs early-stop on the LAST 2 TRAINING YEARS of the fold
(an inner block), never on the validation block that is then reported. Choosing the
checkpoint on the reported block is optimistic by a model-dependent amount (v1 did this;
A1_repro keeps it only to reproduce v1).

Known caveats (documented, not bugs):
  - hot_weight=20 was tuned under v1 labels; v2 marks fewer, season-only hot days
    (3.7-3.9% of train days vs 4.5-4.7%), so A1 -> A1' changes the loss weighting too.
    The Week-3 hot_weight re-sweep addresses this.
  - With a WBGT target the strata are still Tmax-based, so humid-heat days outside
    Mar 15-Jul 31 count as "normal". RMSE of different targets is not comparable; compare
    runs through skill scores against references computed on the same target.
  - v1 labels: an episode can straddle a block edge, so the v1 stratum of the last val
    days may depend on data just after the block (evaluation labels only; v1 repro only).
  - The pre-2019 acceptance events are labelled with in-sample climatology.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.climatology import apply_climatology, doy_climatology
from pipeline.labels_v2 import episodes, label_frame
from training.data import _load_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
DAILY_V2_PATH = REPO_ROOT / "datasets_v2" / "all_daily_v2.parquet"
TEST_START = pd.Timestamp("2019-01-01")
FOLDS = {
    "f1": (pd.Timestamp("2007-01-01"), pd.Timestamp("2009-12-31")),
    "f2": (pd.Timestamp("2010-01-01"), pd.Timestamp("2012-12-31")),
    "f3": (pd.Timestamp("2013-01-01"), pd.Timestamp("2015-12-31")),
    "f4": (pd.Timestamp("2016-01-01"), pd.Timestamp("2018-12-31")),
}
PRIMARY_FOLD = "f4"
BASE_FEATURES = [
    "t_max", "t_min", "t_mean", "relative_humidity_mean", "wind_speed_mean",
    "surface_pressure_mean", "shortwave_radiation_sum", "doy_sin", "doy_cos", "years_since_1980",
]
TARGETS = ("t_max", "wbgt_bom_max")
LABEL_VERSIONS = ("v1", "v2")
INPUT_DAYS, FORECAST_DAYS = 14, 5


@dataclass
class SplitArrays:
    X: np.ndarray  # (N, 14, F) normalised inputs, float32
    y: np.ndarray  # (N, 5) normalised model target (raw or anomaly), float32
    y_raw: np.ndarray  # (N, 5) raw target values, deg C
    clim_target: np.ndarray  # (N, 5) climatology of the target on each forecast day
    hot: np.ndarray  # (N, 5) bool, loss weighting mask
    stratum: np.ndarray  # (N, 5) 'normal' / 'unusual' / 'extreme'
    query_dates: pd.Series
    persist: np.ndarray = field(default=None)  # (N,) target on the last input day (persistence reference)

    def subset(self, mask: np.ndarray) -> "SplitArrays":
        """Rows of this split where mask is True."""
        mask = np.asarray(mask, dtype=bool)
        return SplitArrays(
            X=self.X[mask], y=self.y[mask], y_raw=self.y_raw[mask], clim_target=self.clim_target[mask],
            hot=self.hot[mask], stratum=self.stratum[mask],
            query_dates=self.query_dates[mask].reset_index(drop=True),
            persist=None if self.persist is None else self.persist[mask],
        )


@dataclass
class FoldData:
    fold: str
    target: str
    labels: str
    anomaly_target: bool
    feature_columns: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    y_mean: float
    y_std: float
    train: SplitArrays
    val: SplitArrays
    train_end: pd.Timestamp = field(default=None)

    def to_raw(self, y_norm: np.ndarray, split: str) -> np.ndarray:
        """Normalised model output -> raw target (deg C) for the given split. Kept in the
        output's precision (float32 for model outputs), like training.data.denormalize_y."""
        y = y_norm * self.y_std + self.y_mean
        return y + getattr(self, split).clim_target if self.anomaly_target else y


def fold_bounds(fold: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(train_end, val_start, val_end)."""
    if fold not in FOLDS:
        raise ValueError(f"unknown fold {fold!r}; choose from {list(FOLDS)}")
    val_start, val_end = FOLDS[fold]
    return val_start - pd.Timedelta(days=1), val_start, val_end


def fold_windows(fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train windows, val windows) from the frozen index, v1 assignment convention."""
    train_end, val_start, val_end = fold_bounds(fold)
    w = _load_windows()
    w = w[w["forecast_end"] < TEST_START]
    train = w[(w["query_date"] <= train_end) & (w["forecast_end"] <= train_end)]
    val = w[(w["query_date"] >= val_start) & (w["query_date"] <= val_end) & (w["forecast_end"] <= val_end)]
    return train.reset_index(drop=True), val.reset_index(drop=True)


def inner_split(train: SplitArrays, train_end: pd.Timestamp, years: int = 2) -> tuple[SplitArrays, SplitArrays]:
    """(fit, stop): the stop block is the last `years` training years; fit windows must
    finish their forecast before the stop block starts (windows crossing it are dropped)."""
    stop_start = train_end - pd.DateOffset(years=years) + pd.Timedelta(days=1)
    q = pd.DatetimeIndex(train.query_dates)
    fit = q + pd.Timedelta(days=FORECAST_DAYS - 1) < stop_start
    stop = q >= stop_start
    return train.subset(fit), train.subset(stop)


def _load_daily_pre_test() -> pd.DataFrame:
    # The test period is never loaded into a fold: rows from 2019 on are filtered at read time.
    daily = pd.read_parquet(DAILY_V2_PATH, filters=[("date", "<", TEST_START)]).set_index("date").sort_index()
    return daily[daily.index < TEST_START]


def _v1_labels(dates, t_max, mean, std) -> tuple[np.ndarray, np.ndarray]:
    """v1 hot mask and stratum (training.data.build_split_target_stratum convention)."""
    sigma = (t_max - mean) / std
    hot = sigma > 1.5
    ids, _ = episodes(dates, hot, min_span_days=3)
    in_ep = ids > 0
    stratum = np.full(len(t_max), "normal", dtype="<U7")
    stratum[((sigma > 1.0) & (sigma <= 1.5)) | (hot & ~in_ep)] = "unusual"
    stratum[hot & in_ep] = "extreme"
    return hot, stratum


def fold_daily(fold: str, target: str, labels: str) -> tuple[pd.DataFrame, list[str]]:
    """Daily frame (pre-test dates only) with inputs, target climatology, hot mask and
    stratum computed from this fold's training years; plus the input column list."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if labels not in LABEL_VERSIONS:
        raise ValueError(f"labels must be one of {LABEL_VERSIONS}")
    train_end, _, _ = fold_bounds(fold)
    d = _load_daily_pre_test().copy()
    train_mask = d.index <= train_end

    t_clim = doy_climatology(d["t_max"], train_mask)
    t_mean, t_std = apply_climatology(d.index, t_clim)
    d["clim_mean_t_max"], d["clim_std_t_max"] = t_mean, t_std
    d["t_max_anomaly"] = d["t_max"].to_numpy() - t_mean
    if target == "t_max":
        target_channels = ["clim_mean_t_max", "clim_std_t_max", "t_max_anomaly"]
    else:
        clim = doy_climatology(d[target], train_mask)
        m, s = apply_climatology(d.index, clim)
        d[f"clim_mean_{target}"], d[f"clim_std_{target}"] = m, s
        d[f"{target}_anomaly"] = d[target].to_numpy() - m
        target_channels = [target, f"clim_mean_{target}", f"clim_std_{target}", f"{target}_anomaly"]

    if labels == "v1":
        d["hot"], d["stratum"] = _v1_labels(d.index, d["t_max"].to_numpy(), t_mean, t_std)
    else:
        lab = label_frame(d.index, d["t_max"], d["t_max_anomaly"])
        d["hot"], d["stratum"] = lab["hot_v2"].to_numpy(), lab["stratum_v2"].to_numpy()
    return d, BASE_FEATURES + target_channels


def _split_arrays(d: pd.DataFrame, windows: pd.DataFrame, features: list[str], target: str,
                  anomaly_target: bool, f_mean, f_std, y_mean, y_std) -> SplitArrays:
    pos = d.index.get_indexer(pd.DatetimeIndex(windows["query_date"]))
    if (pos < INPUT_DAYS).any() or (pos < 0).any():
        raise ValueError("window query dates missing from the daily table")
    in_idx = pos[:, None] + np.arange(-INPUT_DAYS, 0)[None, :]
    out_idx = pos[:, None] + np.arange(FORECAST_DAYS)[None, :]
    # Same float32 arithmetic as training/data.py (raw float32 arrays, float32 stats), so the
    # primary fold with v1 settings reproduces v1's arrays bit-for-bit.
    feats = d[features].to_numpy(dtype=np.float32)
    X = ((feats[in_idx] - f_mean) / f_std).astype(np.float32)
    y_raw = d[target].to_numpy(dtype=np.float32)[out_idx]
    clim_col = "clim_mean_t_max" if target == "t_max" else f"clim_mean_{target}"
    clim_target = d[clim_col].to_numpy(dtype=np.float64)[out_idx]
    y_model = (y_raw - clim_target).astype(np.float32) if anomaly_target else y_raw
    return SplitArrays(
        X=X,
        y=((y_model - y_mean) / y_std).astype(np.float32),
        y_raw=y_raw,
        clim_target=clim_target,
        hot=d["hot"].to_numpy(dtype=bool)[out_idx],
        stratum=d["stratum"].to_numpy()[out_idx],
        query_dates=windows["query_date"].reset_index(drop=True),
        persist=d[target].to_numpy(dtype=np.float64)[pos - 1],
    )


def build_fold(fold: str, target: str = "t_max", labels: str = "v1", anomaly_target: bool = False) -> FoldData:
    """All arrays for one fold; normalisation fitted on the fold's training-year daily rows
    (the v1 convention: every training day, not only window days)."""
    d, features = fold_daily(fold, target, labels)
    train_end, _, _ = fold_bounds(fold)
    train_rows = d[d.index <= train_end]
    f_mean = train_rows[features].to_numpy(dtype=np.float64).mean(axis=0)
    f_std = train_rows[features].to_numpy(dtype=np.float64).std(axis=0)
    f_std[f_std == 0] = 1.0
    clim_col = "clim_mean_t_max" if target == "t_max" else f"clim_mean_{target}"
    y_series = train_rows[target] - train_rows[clim_col] if anomaly_target else train_rows[target]
    y_mean, y_std = float(y_series.mean()), float(y_series.std(ddof=0)) or 1.0
    # v1 stored float32 stats; match that so the primary fold reproduces v1 bit-for-bit.
    f_mean, f_std = f_mean.astype(np.float32), f_std.astype(np.float32)

    train_w, val_w = fold_windows(fold)
    args = (features, target, anomaly_target, f_mean, f_std, y_mean, y_std)
    return FoldData(
        fold=fold, target=target, labels=labels, anomaly_target=anomaly_target,
        feature_columns=features, feature_mean=f_mean, feature_std=f_std, y_mean=y_mean, y_std=y_std,
        train=_split_arrays(d, train_w, *args), val=_split_arrays(d, val_w, *args), train_end=train_end,
    )
