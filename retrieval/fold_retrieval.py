"""
Fold-aware analogue retrieval for v2 (plan v5, Week 3: rungs R0, R0-rand, R1, R1-rand; Week 4: Rg).

Why a new module: v1's retrieval (retrieval/features.py, build_index.py) uses a
climatology and feature normalisation fitted on 1980-2015, which contains the validation
blocks of folds f1-f3. Here everything is rebuilt per fold from that fold's TRAINING years
only (training.folds.fold_daily): the anomaly features, the z-score normalisation and the
candidate pool. The v1 files are not changed.

Candidate pool: the fold's training windows (training.folds.fold_windows), so no
validation, later-fold or 2019+ window can ever be retrieved.

Eligibility (the v1 rules, retrieval/query.py), for every query window:
  1. chronological: candidate query_date <= query_date - 19 days, so the candidate's input
     and 5-day outcome end before the query's input window starts;
  2. split: validation queries only see training windows (true by construction);
  3. no analogue from the query's own heat episode (fold labels; a window's episode is the
     first episode among its 5 target days, 0 = none);
  dedup: at most 2 analogues per episode and >= 10 days between any two analogues.

Modes (pre-registered 2026-10-06, context/decisions.md):
  "sim"   R0: top-K by cosine similarity of v1's 17 window features.
  "rand"  R0-rand: K random eligible windows (same rules and dedup), fixed per fold and seed.
  "time"  R1: only candidates whose window-end day of year is within +-30 days of the
          query's (circular), then top-K by similarity.
  "time_rand"  R1-rand: R1's eligible pool (same +-30-day window), K drawn at random as in
          "rand". It is R1's own random control (added 2026-10-07, G3 condition 3).
  "region"  Rg: top-K by cosine similarity of the REGIONAL pattern instead of Delhi's own
          features: standardised anomalies on the last 3 input days of daily Tmax at the 27
          upstream points plus Delhi; the WBGT family adds the 27 points' daily mean dew point
          and Delhi's relative humidity (pre-registered 2026-10-07). Same pool, rules and dedup
          as R0, so R0-rand is its random control.

Run from repo root (prints a summary for one fold):
    python -m retrieval.fold_retrieval f4 v2
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.climatology import apply_climatology, doy_climatology
from retrieval.features import FEATURE_NAMES
from training.folds import FOLDS, FORECAST_DAYS, INPUT_DAYS, TEST_START, fold_bounds, fold_daily, fold_windows

MODES = ("sim", "rand", "time", "time_rand", "region")
RANDOM_MODES = ("rand", "time_rand")  # need a seed; draws are fixed per fold and seed
K_DEFAULT = 5
BUFFER_DAYS = 19  # v1: candidate_latest_start = query_date - 19 days
MIN_DAYS_APART = 10
MAX_PER_EPISODE = 2
DOY_WINDOW = 30
_TOP_POOL = 200  # most-similar candidates screened before dedup (full sort if too few survive)
REGION_LAGS = 3  # Rg: last 3 input days
UPSTREAM_PATH = Path(__file__).resolve().parents[1] / "datasets_v2" / "upstream_daily.parquet"


def window_features(d: pd.DataFrame, query_dates) -> np.ndarray:
    """(N, 17) raw features of each window's 14 input days, v1 definitions
    (retrieval/features.py) but on the fold's own climatology (d from fold_daily)."""
    pos = d.index.get_indexer(pd.DatetimeIndex(query_dates))
    if (pos < INPUT_DAYS).any():
        raise ValueError("window input days missing from the daily table")
    idx = pos[:, None] + np.arange(-INPUT_DAYS, 0)[None, :]

    def col(name: str) -> np.ndarray:
        return d[name].to_numpy(dtype=np.float64)[idx]

    sig = (d["t_max_anomaly"] / d["clim_std_t_max"]).to_numpy(dtype=np.float64)[idx]
    tmax, rh = col("t_max"), col("relative_humidity_mean")
    t = np.arange(INPUT_DAYS, dtype=np.float64) - (INPUT_DAYS - 1) / 2

    def slope(x: np.ndarray) -> np.ndarray:  # least-squares slope, as np.polyfit(t, x, 1)[0]
        return (x - x.mean(axis=1, keepdims=True)) @ t / np.sum(t**2)

    feats = np.column_stack([
        sig.mean(axis=1), sig[:, -1], slope(sig), sig.std(axis=1), sig.max(axis=1),
        (sig > 1.0).sum(axis=1), (sig > 1.5).sum(axis=1),
        tmax.mean(axis=1), slope(tmax), rh.mean(axis=1), slope(rh),
        col("wind_speed_mean").mean(axis=1), col("surface_pressure_mean").mean(axis=1),
        slope(col("surface_pressure_mean")), col("shortwave_radiation_sum").mean(axis=1),
        d["doy_sin"].to_numpy(dtype=np.float64)[pos - 1], d["doy_cos"].to_numpy(dtype=np.float64)[pos - 1],
    ])
    assert feats.shape[1] == len(FEATURE_NAMES)
    return feats


def window_episode(d: pd.DataFrame, query_dates) -> np.ndarray:
    """(N,) first non-zero episode id among each window's 5 target days (0 = none)."""
    pos = d.index.get_indexer(pd.DatetimeIndex(query_dates))
    ep = d["episode_id"].to_numpy(dtype=np.float64)[pos[:, None] + np.arange(FORECAST_DAYS)[None, :]]
    first = np.where(ep > 0, ep, np.inf).min(axis=1)
    return np.where(np.isfinite(first), first, 0).astype(np.int64)


def regional_daily(fold: str, d: pd.DataFrame, target: str) -> pd.DataFrame:
    """(dates, columns) daily standardised anomalies used by Rg, climatology from the fold's
    training years only: Tmax at the 27 upstream points plus Delhi; for a WBGT target also the
    27 points' daily mean dew point and Delhi's daily mean relative humidity. Pre-2019 rows only."""
    from pipeline.download_era5_upstream import NODES, node_id  # local: only Rg needs the upstream data

    nodes = [node_id(a, o) for a, o in NODES]
    vars_ = ["temperature_2m_max"] + (["dew_point_2m_mean"] if target != "t_max" else [])
    cols = [f"{v}__{n}" for v in vars_ for n in nodes]
    up = pd.read_parquet(UPSTREAM_PATH, columns=cols, filters=[("date", "<", TEST_START)])
    up = up[up.index < TEST_START].reindex(d.index)
    up["t_max__delhi"] = d["t_max"]
    if target != "t_max":
        up["relative_humidity_mean__delhi"] = d["relative_humidity_mean"]
    if up.isna().any().any():
        raise ValueError("regional data missing for some days")
    train_end, _, _ = fold_bounds(fold)
    train = up.index <= train_end
    out = {}
    for c in up.columns:
        m, sd = apply_climatology(up.index, doy_climatology(up[c], train))
        out[c] = (up[c].to_numpy() - m) / sd
    return pd.DataFrame(out, index=up.index)


def region_features(rz: pd.DataFrame, query_dates) -> np.ndarray:
    """(N, REGION_LAGS * columns): each column's anomaly on the last REGION_LAGS input days."""
    pos = rz.index.get_indexer(pd.DatetimeIndex(query_dates))
    if (pos < REGION_LAGS).any():
        raise ValueError("window input days missing from the regional table")
    v = rz.to_numpy(dtype=np.float64)
    return np.concatenate([v[pos - 1 - k] for k in range(REGION_LAGS)], axis=1)


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


@dataclass
class Retrieved:
    idx: np.ndarray  # (N, K) positions in the fold's training windows; -1 = no analogue
    sim: np.ndarray  # (N, K) cosine similarity (NaN where idx == -1)
    n_eligible: np.ndarray  # (N,) eligible candidates before dedup


class FoldRetriever:
    """Candidate pool, normalisation and episodes for one fold and label set."""

    def __init__(self, fold: str, labels: str, target: str = "t_max"):
        if fold not in FOLDS:
            raise ValueError(f"unknown fold {fold!r}")
        self.fold = fold
        self.target = target
        self._region = None  # (vectors, mean, std, daily table), built on first Rg request
        # target only selects which climatology channels fold_daily adds; the retrieval
        # features use Tmax channels for every target (the v1 feature set).
        self.d, _ = fold_daily(fold, target, labels)
        self.train_w, self.val_w = fold_windows(fold)
        q = pd.DatetimeIndex(self.train_w["query_date"])
        if not q.is_monotonic_increasing:
            raise ValueError("training windows must be sorted by query date")
        self.cand_dates = q
        self._cand_day = q.values.astype("datetime64[D]").astype(np.int64)
        raw = window_features(self.d, q)
        self.mean, self.std = raw.mean(axis=0), raw.std(axis=0)  # training windows only
        self.std[self.std == 0] = 1.0
        self.vectors = _unit((raw - self.mean) / self.std)
        self.cand_episode = window_episode(self.d, q)
        self.cand_doy = (q - pd.Timedelta(days=1)).dayofyear.to_numpy()

    def query_vectors(self, query_dates) -> np.ndarray:
        return _unit((window_features(self.d, query_dates) - self.mean) / self.std)

    def _region_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        if self._region is None:
            rz = regional_daily(self.fold, self.d, self.target)
            raw = region_features(rz, self.cand_dates)
            mean, std = raw.mean(axis=0), raw.std(axis=0)  # training windows only
            std[std == 0] = 1.0
            self._region = (_unit((raw - mean) / std), mean, std, rz)
        return self._region

    def region_query_vectors(self, query_dates) -> np.ndarray:
        _, mean, std, rz = self._region_state()
        return _unit((region_features(rz, query_dates) - mean) / std)

    def retrieve(self, query_dates, mode: str = "sim", k: int = K_DEFAULT, seed: int | None = None) -> Retrieved:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode in RANDOM_MODES and seed is None:
            raise ValueError(f"mode {mode!r} needs a seed")
        q = pd.DatetimeIndex(query_dates)
        if mode == "region":
            qv, cand_vectors = self.region_query_vectors(q), self._region_state()[0]
        else:
            qv, cand_vectors = self.query_vectors(q), self.vectors
        q_ep = window_episode(self.d, q)
        q_doy = (q - pd.Timedelta(days=1)).dayofyear.to_numpy()
        n_cand = np.searchsorted(self.cand_dates.values, (q - pd.Timedelta(days=BUFFER_DAYS)).values, side="right")
        rng = np.random.default_rng([list(FOLDS).index(self.fold), seed, 20261006]) if mode in RANDOM_MODES else None

        n = len(q)
        idx = np.full((n, k), -1, dtype=np.int64)
        sim = np.full((n, k), np.nan)
        n_elig = np.zeros(n, dtype=np.int64)
        for start in range(0, n, 1024):
            stop = min(start + 1024, n)
            sims = qv[start:stop] @ cand_vectors.T
            for r in range(stop - start):
                i = start + r
                elig = np.zeros(len(self.cand_dates), dtype=bool)
                elig[: n_cand[i]] = True
                if q_ep[i] > 0:
                    elig &= self.cand_episode != q_ep[i]
                if mode in ("time", "time_rand"):
                    diff = np.abs(self.cand_doy - q_doy[i])
                    elig &= np.minimum(diff, 365 - diff) <= DOY_WINDOW
                n_elig[i] = elig.sum()
                if n_elig[i] == 0:
                    continue
                row = sims[r]
                if mode in RANDOM_MODES:
                    order = rng.permutation(np.flatnonzero(elig))
                    chosen = self._dedup(order, k)
                else:
                    masked = np.where(elig, row, -np.inf)
                    top = min(_TOP_POOL, n_elig[i])
                    part = np.argpartition(-masked, top - 1)[:top]
                    chosen = self._dedup(part[np.argsort(-masked[part], kind="stable")], k)
                    if len(chosen) < k and n_elig[i] > top:
                        full = np.flatnonzero(elig)
                        chosen = self._dedup(full[np.argsort(-row[full], kind="stable")], k)
                idx[i, : len(chosen)] = chosen
                sim[i, : len(chosen)] = row[chosen]
        return Retrieved(idx=idx, sim=sim, n_eligible=n_elig)

    def _dedup(self, ranked: np.ndarray, k: int) -> list[int]:
        """v1 dedup: max 2 per episode, >= 10 days between any two chosen analogues."""
        chosen, days, counts = [], [], {}
        for c in ranked:
            day = self._cand_day[c]
            if any(abs(day - x) < MIN_DAYS_APART for x in days):
                continue
            ep = self.cand_episode[c]
            if ep > 0:
                if counts.get(ep, 0) >= MAX_PER_EPISODE:
                    continue
                counts[ep] = counts.get(ep, 0) + 1
            chosen.append(int(c))
            days.append(day)
            if len(chosen) == k:
                break
        return chosen


def anen_forecast(d: pd.DataFrame, target: str, query_dates, analogue_dates: np.ndarray) -> np.ndarray:
    """(N, 5) analogue-ensemble forecast (as evaluation/anen_v1.py, any target): the query's
    climatology plus its clim_std times the mean standardised anomaly of the analogues'
    own 5-day outcomes. analogue_dates: (N, K) datetime64, NaT where missing (a query with
    no analogue gets the climatology)."""
    mean_col, std_col = (("clim_mean_t_max", "clim_std_t_max") if target == "t_max"
                         else (f"clim_mean_{target}", f"clim_std_{target}"))
    z = ((d[target] - d[mean_col]) / d[std_col]).to_numpy(dtype=np.float64)
    leads = np.arange(FORECAST_DAYS)
    a = pd.DatetimeIndex(np.asarray(analogue_dates).ravel())
    pos = d.index.get_indexer(a).reshape(np.asarray(analogue_dates).shape)
    valid = pos >= 0
    za = np.where(valid[..., None], z[np.where(valid, pos, 0)[..., None] + leads], 0.0)  # (N, K, 5)
    n = valid.sum(axis=1)[:, None]
    zbar = za.sum(axis=1) / np.maximum(n, 1)  # 0 (climatology) when a query has no analogue
    qpos = d.index.get_indexer(pd.DatetimeIndex(query_dates))[:, None] + leads
    return d[mean_col].to_numpy()[qpos] + d[std_col].to_numpy()[qpos] * zbar


def _summary(fold: str, labels: str) -> None:
    fr = FoldRetriever(fold, labels)
    for mode in MODES:
        r = fr.retrieve(fr.val_w["query_date"], mode=mode, seed=0)
        full = (r.idx >= 0).sum(axis=1)
        print(f"{fold} {labels} {mode:4s}: val queries {len(full)}, mean analogues {full.mean():.2f}, "
              f"mean similarity {np.nanmean(r.sim):+.3f}, min eligible {r.n_eligible.min()}")


if __name__ == "__main__":
    _summary(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("f4", "v2")))
