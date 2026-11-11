"""
RS-1a/b/c testbed (pre-registered in context/decisions_retrieval.md): does retrieval help, and does its help depend on
what the base model already knows?

Per family and development fold:
- Windows: the fold's training windows (also the candidate pool) and its validation windows, from
  training.folds.fold_windows; daily table from retrieval.fold_retrieval.FoldRetriever (training-year statistics).
- Bases (what the ridge sees): B0 the target's 14 input days minus the last one; B1 + season; B2 + every Delhi input
  channel; B3 + the regional readings Rg matches on.
- Arms (extra inputs): none; RAFT (retrieval_study.raft); RAFT-rand; R0 (the project's retrieval, FoldRetriever mode
  "sim", K = 5, mean raw level of the analogues' next 5 days minus the query's last value); R0-rand (mode "rand").
- Model: ridge per base and arm (multi-output, standardised columns, unpenalised intercept) predicting the next 5
  days minus the last input value. Ridge strength (and RAFT's m) chosen on the fold's last 2 training years, then
  refitted on all training windows and scored on the validation block.
- RS-1c: RAFT's add-on mode on the saved network forecasts (C3-hop and the Delhi-only LSTM, 10-seed ensemble mean):
  a ridge maps RAFT's values to the network's error, fitted on the networks' inner-block forecasts (the last 2
  training years, never trained on), settings chosen leave-one-year-out within those 2 years.

Development data only: nothing from 2019 onwards is read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from evaluation.stats import make_cluster_ids
from retrieval.fold_retrieval import FoldRetriever, region_features
from retrieval_study import raft
from training.folds import BASE_FEATURES, FORECAST_DAYS, INPUT_DAYS, fold_bounds

FAMILIES = {"Tmax": ("t_max", "v2", "A1prime_hw5"), "WBGT (physical)": ("wbgt_lj_max", "wbgt", "A2Lr_hw5")}
BASES = ("B0", "B1", "B2", "B3")
BASE_NAMES = {"B0": "own history only", "B1": "+ season", "B2": "+ all Delhi inputs", "B3": "+ region"}
ARMS = ("none", "RAFT", "RAFT-rand", "R0", "R0-rand")
RANDOM_ARMS = {"RAFT-rand": "RAFT", "R0-rand": "R0"}
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
RANDOM_SEEDS = tuple(range(10))
K_R0 = 5
INNER_YEARS = 2
LEADS = np.arange(FORECAST_DAYS)


def target_channels(target: str) -> list[str]:
    """The target's own input channels, as training.folds.fold_daily adds them."""
    if target == "t_max":
        return ["clim_mean_t_max", "clim_std_t_max", "t_max_anomaly"]
    return [target, f"clim_mean_{target}", f"clim_std_{target}", f"{target}_anomaly"]


def clim_column(target: str) -> str:
    return "clim_mean_t_max" if target == "t_max" else f"clim_mean_{target}"


@dataclass
class Windows:
    q: pd.DatetimeIndex
    pos: np.ndarray  # daily-table position of each window's first forecast day
    inputs: np.ndarray  # (n, 14) the target on the input days
    last: np.ndarray  # (n,) the target on the last input day
    future: np.ndarray  # (n, 5) the target on the forecast days (the truth)
    day: np.ndarray  # (n,) integer day number of the first forecast day


def make_windows(d: pd.DataFrame, target: str, query_dates) -> Windows:
    q = pd.DatetimeIndex(query_dates)
    pos = d.index.get_indexer(q)
    if (pos < INPUT_DAYS).any() or (pos + FORECAST_DAYS > len(d)).any() or (pos < 0).any():
        raise ValueError("window days missing from the daily table")
    t = d[target].to_numpy(dtype=np.float64)
    inputs = t[pos[:, None] + np.arange(-INPUT_DAYS, 0)[None, :]]
    future = t[pos[:, None] + LEADS[None, :]]
    if np.isnan(inputs).any() or np.isnan(future).any():
        raise ValueError("missing target values in a window")
    return Windows(q=q, pos=pos, inputs=inputs, last=inputs[:, -1].copy(), future=future,
                   day=q.values.astype("datetime64[D]").astype(np.int64))


# ------------------------------------------------------------------ ridge


def ridge_path(x_fit: np.ndarray, y_fit: np.ndarray, alphas=ALPHAS) -> list[tuple]:
    """Multi-output ridge on standardised columns (fit-set mean and SD), unpenalised intercept, for every alpha.
    Returns [(mu, sd, coef, y_mean)] in the order of alphas."""
    mu, sd = x_fit.mean(axis=0), x_fit.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    a = (x_fit - mu) / sd
    y_mean = y_fit.mean(axis=0)
    gram, rhs = a.T @ a, a.T @ (y_fit - y_mean)
    eye = np.eye(a.shape[1])
    return [(mu, sd, np.linalg.solve(gram + alpha * eye, rhs), y_mean) for alpha in alphas]


def ridge_predict(model: tuple, x: np.ndarray) -> np.ndarray:
    mu, sd, coef, y_mean = model
    return ((x - mu) / sd) @ coef + y_mean


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def select_fit_predict(x_tr: np.ndarray, x_va: np.ndarray, w_tr: Windows, w_va: Windows, use: np.ndarray,
                       fit: np.ndarray, stop: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Ridge strength by RMSE (deg C) on the stop rows after fitting on the fit rows; then refit on all usable training
    rows. The ridge predicts the next 5 days minus the last input value. Returns (validation forecasts in deg C,
    chosen alpha, its stop-block RMSE)."""
    y_tr = w_tr.future - w_tr.last[:, None]
    f, s = use & fit, use & stop
    scores = [rmse(ridge_predict(m, x_tr[s]) + w_tr.last[s, None], w_tr.future[s])
              for m in ridge_path(x_tr[f], y_tr[f])]
    best = int(np.argmin(scores))
    alpha = ALPHAS[best]
    model = ridge_path(x_tr[use], y_tr[use], (alpha,))[0]
    return ridge_predict(model, x_va) + w_va.last[:, None], alpha, scores[best]


# ------------------------------------------------------------------ one fold and family


@dataclass
class FoldData:
    fold: str
    family: str
    target: str
    labels: str
    fr: FoldRetriever
    d: pd.DataFrame
    train: Windows
    val: Windows
    fit: np.ndarray  # training windows whose forecast ends before the inner (stop) block
    stop: np.ndarray  # training windows in the inner block (the last 2 training years)
    _cache: dict = field(default_factory=dict)

    @classmethod
    def build(cls, fold: str, family: str) -> "FoldData":
        target, labels, _ = FAMILIES[family]
        fr = FoldRetriever(fold, labels, target)
        d = fr.d
        train = make_windows(d, target, fr.cand_dates)
        val = make_windows(d, target, pd.DatetimeIndex(fr.val_w["query_date"]))
        train_end, _, _ = fold_bounds(fold)
        stop_start = train_end - pd.DateOffset(years=INNER_YEARS) + pd.Timedelta(days=1)
        fit = np.asarray(train.q + pd.Timedelta(days=FORECAST_DAYS - 1) < stop_start)
        stop = np.asarray(train.q >= stop_start)
        return cls(fold, family, target, labels, fr, d, train, val, fit, stop)

    def windows(self, which: str) -> Windows:
        return self.train if which == "train" else self.val

    # ---------------- bases
    def base(self, name: str, which: str) -> np.ndarray:
        key = ("base", name, which)
        if key in self._cache:
            return self._cache[key]
        w = self.windows(which)
        parts = [w.inputs - w.last[:, None]]  # B0: RAFT's normalised input
        if name in ("B1", "B2", "B3"):
            clim = self.d[clim_column(self.target)].to_numpy(dtype=np.float64)[w.pos[:, None] + LEADS[None, :]]
            parts += [self.d["doy_sin"].to_numpy(dtype=np.float64)[w.pos][:, None],
                      self.d["doy_cos"].to_numpy(dtype=np.float64)[w.pos][:, None],
                      clim - w.last[:, None]]
        if name in ("B2", "B3"):
            cols = list(BASE_FEATURES) + target_channels(self.target)
            x = self.d[cols].to_numpy(dtype=np.float64)[w.pos[:, None] + np.arange(-INPUT_DAYS, 0)[None, :]]
            parts.append(x.reshape(len(w.q), -1))
        if name == "B3":
            parts.append(region_features(self.fr._region_state()[3], w.q))
        out = np.column_stack(parts)
        self._cache[key] = out
        return out

    # ---------------- RAFT
    def raft_neighbours(self, seed: int | None = None) -> tuple[raft.Neighbours, raft.Neighbours]:
        key = ("raft", seed)
        if key not in self._cache:
            c = self.train
            s_tr = None if seed is None else [_fold_seed(self.fold), seed, 1]
            s_va = None if seed is None else [_fold_seed(self.fold), seed, 2]
            self._cache[key] = (raft.neighbours(c.inputs, c.day, c.inputs, c.day, seed=_seed(s_tr)),
                                raft.neighbours(self.val.inputs, self.val.day, c.inputs, c.day, seed=_seed(s_va)))
        return self._cache[key]

    def raft_values(self, m: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        nb_tr, nb_va = self.raft_neighbours(seed)
        cv = raft.offset_values(self.train.future, self.train.last)
        return raft.retrieved_values(nb_tr, cv, m), raft.retrieved_values(nb_va, cv, m)

    # ---------------- the project's retrieval (R0)
    def r0_idx(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        key = ("r0", seed)
        if key not in self._cache:
            mode = "sim" if seed is None else "rand"
            self._cache[key] = (self.fr.retrieve(self.train.q, mode, K_R0, seed=seed).idx,
                                self.fr.retrieve(self.val.q, mode, K_R0, seed=seed).idx)
        return self._cache[key]

    def r0_values(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(train values, val values, train rows with a full set of K analogues). Values: the mean raw level of the
        K analogues' next 5 days minus the query's last input value."""
        out, full = [], None
        for which, idx in zip(("train", "val"), self.r0_idx(seed)):
            w = self.windows(which)
            ok = (idx >= 0).all(axis=1)
            fut = self.train.future[np.where(idx >= 0, idx, 0)]  # (n, K, 5)
            vals = np.where(ok[:, None], fut.mean(axis=1) - w.last[:, None], 0.0)
            out.append(vals)
            if which == "train":
                full = ok
            elif not ok.all():
                raise ValueError(f"{self.fold}: a validation query has fewer than {K_R0} analogues")
        return out[0], out[1], full


def _fold_seed(fold: str) -> int:
    return {"f1": 1, "f2": 2, "f3": 3, "f4": 4}[fold]


def _seed(parts):
    return None if parts is None else int(np.random.SeedSequence(parts).generate_state(1)[0])


# ------------------------------------------------------------------ arms


def run_arm(fd: FoldData, base: str, arm: str, seed: int | None = None, m_fixed: int | None = None) -> dict:
    """Validation forecasts of one base and arm (one seed for the random arms). Returns {"pred": (n_val, 5),
    "alpha", "m", "stop_rmse"}."""
    x_tr, x_va = fd.base(base, "train"), fd.base(base, "val")
    use = np.ones(len(fd.train.q), dtype=bool)
    if arm == "none":
        pred, alpha, score = select_fit_predict(x_tr, x_va, fd.train, fd.val, use, fd.fit, fd.stop)
        return {"pred": pred, "alpha": alpha, "m": None, "stop_rmse": score}
    if arm in ("RAFT", "RAFT-rand"):
        grid = (m_fixed,) if m_fixed is not None else raft.M_GRID
        best = None
        for m in grid:
            r_tr, r_va = fd.raft_values(m, seed if arm == "RAFT-rand" else None)
            pred, alpha, score = select_fit_predict(np.column_stack([x_tr, r_tr]), np.column_stack([x_va, r_va]),
                                                    fd.train, fd.val, use, fd.fit, fd.stop)
            if best is None or score < best["stop_rmse"]:
                best = {"pred": pred, "alpha": alpha, "m": m, "stop_rmse": score}
        return best
    if arm in ("R0", "R0-rand"):
        r_tr, r_va, full = fd.r0_values(seed if arm == "R0-rand" else None)
        pred, alpha, score = select_fit_predict(np.column_stack([x_tr, r_tr]), np.column_stack([x_va, r_va]),
                                                fd.train, fd.val, full, fd.fit, fd.stop)
        return {"pred": pred, "alpha": alpha, "m": K_R0, "stop_rmse": score}
    raise ValueError(f"unknown arm {arm!r}")


def long_rows(fd: FoldData, pred: np.ndarray, model: str, seed: int = 0) -> pd.DataFrame:
    """One row per validation window and lead, in the project's prediction-file layout."""
    w = fd.val
    n = len(w.q)
    q = np.repeat(w.q.to_numpy(), FORECAST_DAYS)
    lead = np.tile(LEADS + 1, n)
    target_date = q + (lead - 1).astype("timedelta64[D]")
    stratum = fd.d["stratum"].to_numpy()[w.pos[:, None] + LEADS[None, :]].reshape(-1)
    df = pd.DataFrame({"model": model, "seed": seed, "query_date": q, "lead": lead, "target_date": target_date,
                       "pred": pred.reshape(-1), "actual": w.future.reshape(-1), "stratum": stratum})
    df["error"] = df["pred"] - df["actual"]
    df["cluster_id"] = make_cluster_ids(df["target_date"])
    return df.assign(fold=fd.fold)


def ladder(fd: FoldData) -> tuple[dict[tuple[str, str], pd.DataFrame], dict]:
    """Every base x arm on one fold: {(base, arm): long frame (all seeds)} and the chosen settings."""
    frames, settings = {}, {}
    for base in BASES:
        for arm in ("none", "RAFT", "R0"):
            r = run_arm(fd, base, arm)
            frames[(base, arm)] = long_rows(fd, r["pred"], f"{base}/{arm}")
            settings[(base, arm)] = {k: r[k] for k in ("alpha", "m", "stop_rmse")}
        for arm, real in RANDOM_ARMS.items():
            parts, chosen = [], []
            for s in RANDOM_SEEDS:
                r = run_arm(fd, base, arm, seed=s, m_fixed=settings[(base, real)]["m"] if arm == "RAFT-rand" else None)
                parts.append(long_rows(fd, r["pred"], f"{base}/{arm}", seed=s))
                chosen.append(r["alpha"])
            frames[(base, arm)] = pd.concat(parts, ignore_index=True)
            settings[(base, arm)] = {"alpha": chosen, "m": settings[(base, real)]["m"]}
    return frames, settings


# ------------------------------------------------------------------ RS-1c: RAFT's add-on mode on the networks


def _ensemble_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-mean forecast per query date: columns pred_1..5 and actual_1..5, indexed by query date."""
    g = df.groupby(["query_date", "lead"]).agg(pred=("pred", "mean"), actual=("actual", "first"))
    wide = g.unstack("lead")
    wide.columns = [f"{a}_{lead}" for a, lead in wide.columns]
    return wide.sort_index()


def addon(fd: FoldData, inner: pd.DataFrame, val: pd.DataFrame, seed: int | None = None,
          setting: tuple[float, int] | None = None) -> tuple[np.ndarray, tuple[float, int]]:
    """RAFT's values -> the network's error, fitted on the inner-block ensemble forecasts (settings by
    leave-one-year-out over its 2 years unless given), applied to the validation ensemble forecasts. Returns the
    corrected validation forecasts (n_val, 5) in the order of fd.val and the (alpha, m) used."""
    inn, va = _ensemble_matrix(inner), _ensemble_matrix(val)
    if not va.index.equals(pd.DatetimeIndex(fd.val.q)):
        raise ValueError("validation forecasts do not match the fold's validation windows")
    tr_pos = pd.Series(np.arange(len(fd.train.q)), index=fd.train.q)
    rows = tr_pos.reindex(inn.index)
    if rows.isna().any():
        raise ValueError("an inner-block forecast has no training window")
    rows = rows.to_numpy(dtype=np.int64)
    pred_cols, act_cols = [f"pred_{i}" for i in LEADS + 1], [f"actual_{i}" for i in LEADS + 1]
    err_in = inn[act_cols].to_numpy() - inn[pred_cols].to_numpy()
    years = inn.index.year.to_numpy()

    def features(m: int) -> tuple[np.ndarray, np.ndarray]:
        r_tr, r_va = fd.raft_values(m, seed)
        return r_tr[rows], r_va

    if setting is None:
        best = None
        for m in raft.M_GRID:
            x_in, _ = features(m)
            sq = np.zeros(len(ALPHAS))
            for y in np.unique(years):
                tr, te = years != y, years == y
                for i, model in enumerate(ridge_path(x_in[tr], err_in[tr])):
                    sq[i] += np.sum((ridge_predict(model, x_in[te]) - err_in[te]) ** 2)
            i = int(np.argmin(sq))
            if best is None or sq[i] < best[0]:
                best = (sq[i], ALPHAS[i], m)
        setting = (best[1], best[2])
    alpha, m = setting
    x_in, x_va = features(m)
    model = ridge_path(x_in, err_in, (alpha,))[0]
    return va[pred_cols].to_numpy() + ridge_predict(model, x_va), setting


def network_rows(fd: FoldData, df: pd.DataFrame, model: str) -> pd.DataFrame:
    """The network's 10-seed ensemble mean as one long frame in fd.val's order."""
    va = _ensemble_matrix(df)
    return long_rows(fd, va[[f"pred_{i}" for i in LEADS + 1]].to_numpy(), model)
