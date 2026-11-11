"""
Step 2 screens (RS-2, pre-registered in context/decisions_retrieval.md before any of them ran): cheap tests of
retrieval types that might help our best model, before any network is retrained.

The screen is RAFT's add-on mode, generalised. A ridge maps a retrieval's features to C3-hop's error (truth minus
the 10-seed ensemble forecast). It is fitted on the networks' saved inner-block forecasts (each fold's last 2 training
years, never trained on), with its strength chosen leave-one-year-out, then added to the validation forecasts. If a
retrieval carries no information about C3-hop's errors, retraining C3-hop with it is not expected to help either.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retrieval_study.testbed import ALPHAS, LEADS, FoldData, _ensemble_matrix, long_rows, ridge_path, ridge_predict

PRED_COLS = [f"pred_{i}" for i in LEADS + 1]
ACT_COLS = [f"actual_{i}" for i in LEADS + 1]


@dataclass
class NetworkForecasts:
    """One network's 10-seed ensemble forecasts on a fold: the inner block (fit) and the validation block."""
    rid: str
    inner: pd.DataFrame  # indexed by query date: pred_1..5, actual_1..5
    val: pd.DataFrame

    @classmethod
    def load(cls, fd: FoldData, pred_dir, rid: str) -> "NetworkForecasts":
        inner = _ensemble_matrix(pd.read_parquet(pred_dir / "inner" / rid / f"{fd.fold}.parquet"))
        val = _ensemble_matrix(pd.read_parquet(pred_dir / rid / f"{fd.fold}.parquet"))
        if not val.index.equals(pd.DatetimeIndex(fd.val.q)):
            raise ValueError("validation forecasts do not match the fold's validation windows")
        if not inner.index.isin(fd.train.q).all():
            raise ValueError("an inner-block forecast has no training window")
        return cls(rid, inner, val)

    def error(self, which: str) -> np.ndarray:
        df = self.inner if which == "inner" else self.val
        return df[ACT_COLS].to_numpy() - df[PRED_COLS].to_numpy()


def addon_correction(nf: NetworkForecasts, x_inner: np.ndarray, x_val: np.ndarray,
                     alpha: float | None = None) -> tuple[np.ndarray, float]:
    """Corrected validation forecasts (n_val, 5) and the ridge strength used. x_inner is aligned with nf.inner's rows,
    x_val with nf.val's. Rows whose features are not finite are left out of the fit and get no correction."""
    err = nf.error("inner")
    ok = np.isfinite(x_inner).all(axis=1)
    x, e, years = x_inner[ok], err[ok], nf.inner.index.year.to_numpy()[ok]
    if alpha is None:
        sq = np.zeros(len(ALPHAS))
        for y in np.unique(years):
            tr, te = years != y, years == y
            for i, model in enumerate(ridge_path(x[tr], e[tr])):
                sq[i] += np.sum((ridge_predict(model, x[te]) - e[te]) ** 2)
        alpha = ALPHAS[int(np.argmin(sq))]
    model = ridge_path(x, e, (alpha,))[0]
    good = np.isfinite(x_val).all(axis=1)
    corr = np.where(good[:, None], ridge_predict(model, np.where(np.isfinite(x_val), x_val, 0.0)), 0.0)
    return nf.val[PRED_COLS].to_numpy() + corr, alpha


def screen_frames(fd: FoldData, nf: NetworkForecasts, real: tuple[np.ndarray, np.ndarray],
                  random: list[tuple[np.ndarray, np.ndarray]], name: str) -> dict[str, pd.DataFrame]:
    """Long frames: the network alone, + the retrieval, + its random control (one seed per element of `random`, with
    the real arm's ridge strength)."""
    out = {"alone": long_rows(fd, nf.val[PRED_COLS].to_numpy(), nf.rid)}
    pred, alpha = addon_correction(nf, *real)
    out["+real"] = long_rows(fd, pred, f"{nf.rid}+{name}")
    parts = [long_rows(fd, addon_correction(nf, *r, alpha=alpha)[0], f"{nf.rid}+{name}-rand", seed=s)
             for s, r in enumerate(random)]
    out["+rand"] = pd.concat(parts, ignore_index=True) if parts else None
    return out


# ------------------------------------------------------------------ candidate retrievals


def standardised_anomaly(fd: FoldData) -> np.ndarray:
    """The target's standardised anomaly per day (the fold's training-year climatology)."""
    from retrieval_study.testbed import clim_column

    d = fd.d
    std_col = "clim_std_t_max" if fd.target == "t_max" else f"clim_std_{fd.target}"
    return ((d[fd.target] - d[clim_column(fd.target)]) / d[std_col]).to_numpy(dtype=np.float64)


def analogue_signal(fd: FoldData, idx: np.ndarray) -> np.ndarray:
    """(n, 5) the mean standardised anomaly of each query's analogues' next 5 days (the AnEn signal of the
    information check); rows with a missing analogue are NaN. idx: positions among fd.train's windows."""
    z = standardised_anomaly(fd)
    ok = (idx >= 0).all(axis=1)
    fut = z[fd.train.pos[np.where(idx >= 0, idx, 0)][..., None] + LEADS]  # (n, K, 5)
    return np.where(ok[:, None], fut.mean(axis=1), np.nan)


def project_retrieval(fd: FoldData, mode: str, query_dates, seed: int | None = None, k: int = 5) -> np.ndarray:
    """The project's own retrieval (retrieval.fold_retrieval.FoldRetriever) for the given queries: the analogue
    signal of mode "synoptic" (Rz, the 500 hPa pattern), "region" (Rg) or "sim" (R0); mode "rand" is their random
    control (R0-rand), as in the runs."""
    idx = fd.fr.retrieve(pd.DatetimeIndex(query_dates), mode, k, seed=seed).idx
    return analogue_signal(fd, idx)


def _nearest(q_keys: np.ndarray, q_day: np.ndarray, c_keys: np.ndarray, c_day: np.ndarray, overlap: int, m: int,
             seed: int | None, chunk: int = 128) -> np.ndarray:
    """(n, m) indices of each query's m nearest eligible candidates by RMS distance of the keys (or m random eligible
    candidates when a seed is given); -1 where fewer exist."""
    from retrieval_study import raft

    rng = np.random.default_rng(seed) if seed is not None else None
    c2 = np.sum(c_keys ** 2, axis=1)
    out = np.full((len(q_keys), m), -1, dtype=np.int64)
    for start in range(0, len(q_keys), chunk):
        stop = min(start + chunk, len(q_keys))
        ok = raft.eligible(q_day[start:stop], c_day, overlap)
        if rng is None:
            q = q_keys[start:stop]
            dist2 = np.sum(q ** 2, axis=1)[:, None] + c2[None, :] - 2.0 * q @ c_keys.T
            score = np.where(ok, -dist2, -np.inf)
        else:
            score = np.where(ok, rng.random(ok.shape), -np.inf)
        out[start:stop] = raft._top_k(score, m)
    return out


LONG_DAYS = 60
M_SCREEN = 10


def long_window(fd: FoldData, query_dates, seed: int | None = None, m: int = M_SCREEN) -> np.ndarray:
    """(n, 5) LW: the mean standardised anomaly of the next 5 days after the m training windows whose last
    LONG_DAYS days of standardised anomaly are closest (RMS) to the query's. Season-to-date context that the
    networks' 14-day inputs do not see."""
    z = standardised_anomaly(fd)
    q = pd.DatetimeIndex(query_dates)
    qpos = fd.d.index.get_indexer(q)
    has_history = qpos >= LONG_DAYS  # earlier queries get NaN (their key would wrap round the array)
    keep = fd.train.pos >= LONG_DAYS
    c_pos, c_day = fd.train.pos[keep], fd.train.day[keep]
    back = np.arange(-LONG_DAYS, 0)
    out = np.full((len(q), len(LEADS)), np.nan)
    qp = qpos[has_history]
    nb = _nearest(z[qp[:, None] + back], q[has_history].values.astype("datetime64[D]").astype(np.int64),
                  z[c_pos[:, None] + back], c_day, LONG_DAYS + len(LEADS), m, seed)
    fut = z[c_pos[np.where(nb >= 0, nb, 0)][..., None] + LEADS]
    out[has_history] = np.where((nb >= 0).all(axis=1)[:, None], fut.mean(axis=1), np.nan)
    return out


def site_anomalies(fd: FoldData) -> np.ndarray:
    """(days, 28) daily standardised Tmax anomalies at Delhi (column 0) and the 27 upstream points, climatology from
    the fold's training years only; pre-2019 rows only."""
    from pipeline.climatology import apply_climatology, doy_climatology
    from retrieval.fold_retrieval import UPSTREAM_PATH
    from training.folds import TEST_START, fold_bounds
    from training.graph_data import NODE_IDS

    cols = [f"temperature_2m_max__{n}" for n in NODE_IDS]
    up = pd.read_parquet(UPSTREAM_PATH, columns=cols, filters=[("date", "<", TEST_START)])
    up = up[up.index < TEST_START].reindex(fd.d.index)
    if up.isna().any().any():
        raise ValueError("upstream Tmax missing for some days")
    train = np.asarray(fd.d.index <= fold_bounds(fd.fold)[0])
    out = [standardised_anomaly(fd)]
    for c in cols:
        mean, sd = apply_climatology(up.index, doy_climatology(up[c].astype(np.float64), train))
        out.append((up[c].to_numpy(dtype=np.float64) - mean) / sd)
    return np.column_stack(out)


def cross_site(fd: FoldData, query_dates, seed: int | None = None, m: int = M_SCREEN,
               delhi_only: bool = False) -> np.ndarray:
    """(n, 5) CS (Tmax only): the mean standardised anomaly of the next 5 days after the m windows closest (RMS of
    the 14-day standardised anomaly) to Delhi's, searched over the training windows of Delhi AND the 27 upstream
    points (each in its own standardised anomaly). delhi_only=True searches Delhi's windows alone (the same
    machinery, to isolate the effect of the larger pool)."""
    if fd.target != "t_max":
        raise ValueError("the cross-site pool exists for Tmax only")
    za = site_anomalies(fd)
    sites = [0] if delhi_only else list(range(za.shape[1]))
    back = np.arange(-14, 0)
    keys = np.concatenate([za[fd.train.pos[:, None] + back, s] for s in sites])
    vals = np.concatenate([za[fd.train.pos[:, None] + LEADS, s] for s in sites])
    days = np.tile(fd.train.day, len(sites))
    q = pd.DatetimeIndex(query_dates)
    qpos = fd.d.index.get_indexer(q)
    nb = _nearest(za[qpos[:, None] + back, 0], q.values.astype("datetime64[D]").astype(np.int64), keys, days,
                  14 + len(LEADS), m, seed, chunk=32)
    fut = vals[np.where(nb >= 0, nb, 0)]
    return np.where((nb >= 0).all(axis=1)[:, None], fut.mean(axis=1), np.nan)


# ------------------------------------------------------------------ retrieval for ranges (KR)

K_RANGES = 50
RANGE_ALPHA = 0.2  # 80% ranges


def interval_score(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float = RANGE_ALPHA) -> np.ndarray:
    """Gneiting & Raftery (2007) interval score of the central (1 - alpha) range; lower is better."""
    return (hi - lo) + (2 / alpha) * np.clip(lo - y, 0, None) + (2 / alpha) * np.clip(y - hi, 0, None)


def knn_ranges(fd: FoldData, nf: NetworkForecasts, method: str, k: int = K_RANGES, alpha: float = RANGE_ALPHA,
               seed: int | None = None) -> dict[str, np.ndarray]:
    """80% ranges around the network's validation forecasts from its inner-block errors. method "global": every
    inner-block error of that lead; "knn": only the k inner-block situations most similar to the query (R0's key,
    cosine of v1's 17 features); "random": k random inner-block situations (the random control; needs a seed).
    Returns lo, hi (n_val, 5) and the truth."""
    err = nf.error("inner")
    pred = nf.val[PRED_COLS].to_numpy()
    y = nf.val[ACT_COLS].to_numpy()
    q_lo, q_hi = alpha / 2, 1 - alpha / 2
    if method == "global":
        lo = pred + np.quantile(err, q_lo, axis=0)[None, :]
        hi = pred + np.quantile(err, q_hi, axis=0)[None, :]
        return {"lo": lo, "hi": hi, "y": y}
    if method == "knn":
        sim = fd.fr.query_vectors(nf.val.index) @ fd.fr.query_vectors(nf.inner.index).T
        nb = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    elif method == "random":
        rng = np.random.default_rng(seed)
        nb = np.stack([rng.choice(len(err), size=k, replace=False) for _ in range(len(pred))])
    else:
        raise ValueError(f"unknown method {method!r}")
    local = err[nb]  # (n_val, k, 5)
    lo = pred + np.quantile(local, q_lo, axis=1)
    hi = pred + np.quantile(local, q_hi, axis=1)
    return {"lo": lo, "hi": hi, "y": y}
