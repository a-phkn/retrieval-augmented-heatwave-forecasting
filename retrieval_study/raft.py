"""
RAFT retrieval (Han et al. 2025, "Retrieval Augmented Time Series Forecasting", arXiv 2505.04163, Sec. 3.2-3.3),
written for daily windows (pre-registered in context/decisions_retrieval.md, RS-1).

- Key: a candidate's input window of the target variable; query: the query's input window.
- Normalisation: RAFT subtracts each window's last value. Pearson correlation ignores offsets and scale anyway; the
  subtraction matters for the values.
- Value: the days after a candidate's window minus that window's last value (so it is a change, not a level).
- Similarity: Pearson correlation; the top m candidates are weighted by softmax(rho / tau) with tau = 0.1.
- Periods: keys and values are average-pooled over p = 1, 2 and 4 days and retrieved separately for each p.
  Keys are pooled counting back from the window's end (the most recent days line up); values counting forward
  from the first forecast day. A partial block is dropped.
- Eligibility: a candidate whose window starts fewer than OVERLAP_DAYS days from the query's is excluded (RAFT
  excludes overlapping windows; for past candidates this is also the project's chronological rule).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training.folds import FORECAST_DAYS, INPUT_DAYS

PERIODS = (1, 2, 4)
TEMPERATURE = 0.1
M_GRID = (1, 5, 10, 20)
M_MAX = max(M_GRID)
OVERLAP_DAYS = INPUT_DAYS + FORECAST_DAYS  # two windows closer than this share at least one day
CHUNK = 1024


def pool(x: np.ndarray, p: int, align: str) -> np.ndarray:
    """Average-pool the last axis in blocks of p days. align="end" counts back from the last day (used for keys),
    align="start" counts forward from the first (used for values); a partial block is dropped."""
    n = x.shape[-1] // p
    if n == 0:
        raise ValueError(f"cannot pool {x.shape[-1]} days in blocks of {p}")
    if align == "end":
        part = x[..., x.shape[-1] - n * p:]
    elif align == "start":
        part = x[..., : n * p]
    else:
        raise ValueError("align must be 'end' or 'start'")
    return part.reshape(*x.shape[:-1], n, p).mean(axis=-1)


def standardise_rows(x: np.ndarray) -> np.ndarray:
    """Each row minus its mean, divided by its SD, so that row_a @ row_b / n is their Pearson correlation. A constant
    row becomes all zeros (correlation 0 with everything)."""
    x = np.asarray(x, dtype=np.float64)
    c = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt((c ** 2).mean(axis=1, keepdims=True))
    return np.where(sd > 0, c / np.where(sd > 0, sd, 1.0), 0.0)


def offset_values(future: np.ndarray, last: np.ndarray) -> np.ndarray:
    """(N, F) the days after each window minus the window's last value."""
    return np.asarray(future, dtype=np.float64) - np.asarray(last, dtype=np.float64)[:, None]


@dataclass
class Neighbours:
    """Per period: (n, M_MAX) candidate indices (best first; -1 = none) and their Pearson correlations."""
    idx: dict[int, np.ndarray]
    sim: dict[int, np.ndarray]


def eligible(q_day: np.ndarray, c_day: np.ndarray, overlap: int = OVERLAP_DAYS) -> np.ndarray:
    """(n, N) True where the candidate window does not overlap the query window (windows `overlap` steps long)."""
    return np.abs(q_day[:, None] - c_day[None, :]) >= overlap


def neighbours(q_keys: np.ndarray, q_day: np.ndarray, c_keys: np.ndarray, c_day: np.ndarray,
               periods: tuple[int, ...] = PERIODS, m_max: int = M_MAX, seed: int | None = None,
               overlap: int = OVERLAP_DAYS, chunk: int = CHUNK) -> Neighbours:
    """RAFT's top-m_max candidates per query and period by Pearson correlation of the pooled keys. With a seed, the
    m_max are instead drawn at random from the eligible candidates (the random control; same draw for every
    period), and keep their actual correlations so that the weighting machinery is identical.
    q_keys: (n, L) query windows; c_keys: (N, L) candidate windows; *_day: integer time-step numbers (days, or
    hours for hourly data) of each window's first forecast step; overlap: the window length in those steps."""
    q_day, c_day = np.asarray(q_day, dtype=np.int64), np.asarray(c_day, dtype=np.int64)
    rng = np.random.default_rng(seed) if seed is not None else None
    std_q = {p: standardise_rows(pool(q_keys, p, "end")) for p in periods}
    std_c = {p: standardise_rows(pool(c_keys, p, "end")) for p in periods}
    n = len(q_keys)
    idx = {p: np.full((n, m_max), -1, dtype=np.int64) for p in periods}
    sim = {p: np.full((n, m_max), np.nan) for p in periods}
    k = min(m_max, len(c_keys))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        ok = eligible(q_day[start:stop], c_day, overlap)
        if rng is not None:  # a uniform random draw without replacement among the eligible candidates
            draw = _top_k(np.where(ok, rng.random(ok.shape), -np.inf), k)
        for p in periods:
            rho = std_q[p][start:stop] @ std_c[p].T / std_q[p].shape[1]
            top = _top_k(np.where(ok, rho, -np.inf), k) if rng is None else draw
            ok_top = np.take_along_axis(ok, np.where(top >= 0, top, 0), axis=1) & (top >= 0)
            idx[p][start:stop, :k] = np.where(ok_top, top, -1)
            sim[p][start:stop, :k] = np.where(ok_top, np.take_along_axis(rho, np.where(top >= 0, top, 0), axis=1),
                                              np.nan)
    return Neighbours(idx=idx, sim=sim)


def _top_k(score: np.ndarray, k: int) -> np.ndarray:
    """(n, k) column indices of each row's k largest finite scores, largest first; -1 where a row has fewer."""
    top = np.argpartition(-score, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(score, top, axis=1), axis=1, kind="stable")
    top = np.take_along_axis(top, order, axis=1)
    return np.where(np.isfinite(np.take_along_axis(score, top, axis=1)), top, -1)


def weights(sim: np.ndarray, m: int, tau: float = TEMPERATURE) -> np.ndarray:
    """(n, m) softmax(rho / tau) over each query's first m neighbours (missing neighbours get weight 0; a query with
    none gets all zeros)."""
    s = sim[:, :m]
    z = np.where(np.isfinite(s), s / tau, -np.inf)
    zmax = np.max(np.where(np.isfinite(z), z, -np.inf), axis=1, keepdims=True)
    zmax = np.where(np.isfinite(zmax), zmax, 0.0)
    e = np.where(np.isfinite(z), np.exp(z - zmax), 0.0)
    tot = e.sum(axis=1, keepdims=True)
    return np.where(tot > 0, e / np.where(tot > 0, tot, 1.0), 0.0)


def retrieved_values(nb: Neighbours, c_values: np.ndarray, m: int, tau: float = TEMPERATURE) -> np.ndarray:
    """(n, sum over periods of F // p) RAFT's weighted values: per period, the softmax-weighted average of the top m
    neighbours' pooled values (pooled from the first forecast day), concatenated over periods (8 for F = 5)."""
    if m > next(iter(nb.idx.values())).shape[1]:
        raise ValueError("m exceeds the number of stored neighbours")
    out = []
    for p, idx in nb.idx.items():
        v = pool(np.asarray(c_values, dtype=np.float64), p, "start")  # (N, F // p)
        w = weights(nb.sim[p], m, tau)  # (n, m)
        safe = np.where(idx[:, :m] >= 0, idx[:, :m], 0)
        out.append(np.einsum("nm,nmf->nf", w, v[safe]))
    return np.concatenate(out, axis=1)


def n_values(periods: tuple[int, ...] = PERIODS, horizon: int = FORECAST_DAYS) -> int:
    return sum(horizon // p for p in periods)
