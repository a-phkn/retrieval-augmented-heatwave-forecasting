"""
Upstream node inputs and daily graph edges for the graph backbone runs (plan v5, Week 4;
pre-registered 2026-10-07 in context/decisions.md, "Graph backbone runs and gates").

Per fold, from that fold's TRAINING years only:
  * 12 daily inputs per upstream point: 11 variables (UPSTREAM_VARS) plus the Tmax
    standardised anomaly (day-of-year climatology), each z-scored per point and variable;
  * daily edges: wind-gated advective weights (pipeline.graph.advective_weights) from each
    source point's daily mean wind vector; Delhi (node 0) has no wind direction in its daily
    table, so its outgoing edges use the mean wind vector of its 8 neighbours.

Windows use the same 14 input days as the Delhi arrays (query_date - 14 .. query_date - 1).
The model uses day t-1's edges for the message into day t (models/dstgnn.py), so a window's
edges also come from its input days only. Nothing from the forecast days, the fold's
validation block or 2019+ is used to scale anything; 2019+ rows are never read.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.climatology import apply_climatology, doy_climatology
from pipeline.download_era5_upstream import NODES, node_id
from pipeline.graph import advective_weights, geographic_weights, neighbour_mask, node_coords
from retrieval.fold_retrieval import UPSTREAM_PATH
from training.folds import INPUT_DAYS, TEST_START, fold_bounds

UPSTREAM_VARS = ("dew_point_2m_mean", "relative_humidity_2m_mean", "shortwave_radiation_sum",
                 "soil_moisture_0_to_7cm_mean", "surface_pressure_mean", "temperature_2m_max",
                 "temperature_2m_mean", "temperature_2m_min", "wind_speed_10m_mean", "wind_u_10m", "wind_v_10m")
NODE_IDS = [node_id(a, o) for a, o in NODES]
N_UP_FEATURES = len(UPSTREAM_VARS) + 1  # + Tmax standardised anomaly


def load_upstream() -> pd.DataFrame:
    """Daily upstream table, pre-2019 rows only (filtered at read time)."""
    cols = [f"{v}__{n}" for v in UPSTREAM_VARS for n in NODE_IDS]
    up = pd.read_parquet(UPSTREAM_PATH, columns=cols, filters=[("date", "<", TEST_START)])
    up = up[up.index < TEST_START]
    full = pd.date_range(up.index.min(), up.index.max(), freq="D")
    if not up.index.equals(full) or up.isna().any().any():
        raise ValueError("upstream table must be complete and gap-free")
    return up


def wind_from_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(speed m/s, meteorological direction the wind comes FROM, degrees) from u/v."""
    speed = np.hypot(u, v)
    direction = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return speed, direction


@dataclass
class UpstreamDaily:
    dates: pd.DatetimeIndex  # daily index of the arrays below
    x: np.ndarray  # (days, 27, 12) float32, z-scored on the fold's training years
    adj: np.ndarray  # (days, 28, 28) float32 advective weights, [day, j, i] = edge j -> i
    static_adj: np.ndarray  # (28, 28) geographic weights (C3)
    mask: np.ndarray  # (28, 28) bool neighbour mask

    def positions(self, query_dates) -> np.ndarray:
        """(N, 14) daily positions of each window's input days."""
        q = self.dates.get_indexer(pd.DatetimeIndex(query_dates))
        if (q < INPUT_DAYS).any():
            raise ValueError("window input days missing from the upstream table")
        return q[:, None] + np.arange(-INPUT_DAYS, 0)[None, :]


def build_upstream(fold: str, up: pd.DataFrame | None = None) -> UpstreamDaily:
    """Upstream inputs and daily edges for one fold (training-year statistics only)."""
    up = load_upstream() if up is None else up
    train_end, _, _ = fold_bounds(fold)
    train = np.asarray(up.index <= train_end)
    feats = []
    for n in NODE_IDS:
        cols = [up[f"{v}__{n}"].to_numpy(np.float64) for v in UPSTREAM_VARS]
        tmax = up[f"temperature_2m_max__{n}"]
        m, s = apply_climatology(up.index, doy_climatology(tmax, train))
        cols.append((tmax.to_numpy() - m) / s)
        feats.append(np.stack(cols, axis=1))
    x = np.stack(feats, axis=1)  # (days, 27, 12)
    mu, sd = x[train].mean(axis=0), x[train].std(axis=0)
    sd[sd == 0] = 1.0
    x = ((x - mu) / sd).astype(np.float32)

    coords = node_coords()
    mask = neighbour_mask(coords)
    u = np.stack([up[f"wind_u_10m__{n}"].to_numpy(np.float64) for n in NODE_IDS], axis=1)
    v = np.stack([up[f"wind_v_10m__{n}"].to_numpy(np.float64) for n in NODE_IDS], axis=1)
    delhi_nb = mask[0, 1:] | mask[1:, 0]  # Delhi's neighbours among the upstream points
    u = np.concatenate([u[:, delhi_nb].mean(axis=1, keepdims=True), u], axis=1)
    v = np.concatenate([v[:, delhi_nb].mean(axis=1, keepdims=True), v], axis=1)
    speed, direction = wind_from_uv(u, v)
    adj = advective_weights(speed, direction, coords).astype(np.float32)
    return UpstreamDaily(dates=pd.DatetimeIndex(up.index), x=x, adj=adj,
                         static_adj=geographic_weights(coords).astype(np.float32), mask=mask)
