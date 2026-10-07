"""
Graph geometry for the regional "upstream" DSTGNN (plan v5, Option 1).

Nodes: index 0 = Delhi (centre of the 9 ERA5 cells), then the 27 upstream points of
pipeline/download_era5_upstream.NODES in that fixed order.

Edges j -> i (source j, destination i) exist only between NEIGHBOURS: nodes at most
NEIGHBOUR_KM = 370 km apart: each lattice point's 8 surrounding points (sides ~190-220 km,
diagonals ~295-300 km; the next lattice distance is 377 km) and Delhi's 8 nearest points
(<= 347.5 km; the next is 394.4 km). The cutoff sits inside both gaps, so the graph does not
change if Delhi's approximate coordinates move by up to 0.15 degrees. With one message-passing hop per day, information then moves at
most ~one lattice step (~200-300 km) per day, i.e. ~2.5-3.5 m/s of advection, a typical
pre-monsoon north-westerly. (A dense graph would let a point 1,000 km away reach Delhi in
one day, which no air parcel does.) Two kinds of weight:
  geographic   w_ji = exp(-d_ji / L): fixed, distance only (control C3).
  advective    w_ji(t) = max(0, cos(theta_j(t) - beta_ji)) * (s_j(t) / 5 m/s) * exp(-d_ji / L)
               theta_j = direction the wind at j blows TOWARDS (meteorological "from"
               direction + 180), beta_ji = compass bearing from j to i, s_j = wind speed,
               divided by a 5 m/s reference so the weights are dimensionless and of order 1,
               like the geographic and learned (adaptive) weights. Air at j only "sends" to
               neighbours downwind of it, more strongly when the wind is stronger and the
               nodes are closer (the C4 / DSTGNN dynamic edges).
Self-loops are 0 (a node's own state enters the model separately).
"""
from __future__ import annotations

import numpy as np

from pipeline.download_era5_upstream import NODES

DELHI = (28.61, 77.21)  # approx. centre of the 9 Delhi ERA5 cells (v1 download)
EARTH_RADIUS_KM = 6371.0
LENGTH_SCALE_KM = 500.0
NEIGHBOUR_KM = 370.0
REFERENCE_WIND_MS = 5.0


def node_coords() -> np.ndarray:
    """(28, 2) lat, lon in degrees; row 0 = Delhi."""
    return np.array([DELHI, *NODES], dtype=np.float64)


def distance_km(coords: np.ndarray) -> np.ndarray:
    """(N, N) great-circle (haversine) distances."""
    lat, lon = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    dlat = lat[None, :] - lat[:, None]
    dlon = lon[None, :] - lon[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(coords: np.ndarray) -> np.ndarray:
    """(N, N) initial compass bearing from node j (row) to node i (column), 0 = north, clockwise."""
    lat, lon = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    dlon = lon[None, :] - lon[:, None]
    x = np.sin(dlon) * np.cos(lat[None, :])
    y = np.cos(lat[:, None]) * np.sin(lat[None, :]) - np.sin(lat[:, None]) * np.cos(lat[None, :]) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360.0


def neighbour_mask(coords: np.ndarray, max_km: float = NEIGHBOUR_KM) -> np.ndarray:
    """(N, N) bool: True where an edge may exist (distance <= max_km), no self-loops."""
    m = distance_km(coords) <= max_km
    np.fill_diagonal(m, False)
    return m


def geographic_weights(coords: np.ndarray, length_km: float = LENGTH_SCALE_KM,
                       max_km: float = NEIGHBOUR_KM) -> np.ndarray:
    """(N, N) fixed edge weights exp(-d / L) between neighbours; source rows, destination columns."""
    return np.exp(-distance_km(coords) / length_km) * neighbour_mask(coords, max_km)


def advective_weights(wind_speed: np.ndarray, wind_dir_from: np.ndarray, coords: np.ndarray,
                      length_km: float = LENGTH_SCALE_KM, max_km: float = NEIGHBOUR_KM) -> np.ndarray:
    """Wind-gated edge weights between neighbours, for each time step.

    wind_speed, wind_dir_from: (..., N) per node (m/s; meteorological degrees the wind comes
    FROM). Returns (..., N, N): [..., j, i] = weight of edge j -> i (dimensionless)."""
    toward = (np.asarray(wind_dir_from) + 180.0) % 360.0
    beta = bearing_deg(coords)  # (N, N), j -> i
    align = np.clip(np.cos(np.radians(toward[..., :, None] - beta)), 0.0, None)
    speed = np.asarray(wind_speed)[..., :, None] / REFERENCE_WIND_MS
    return align * speed * np.exp(-distance_km(coords) / length_km) * neighbour_mask(coords, max_km)
