"""
Paired comparison statistics for forecast models (plan v5, Phase 0).

Why this exists: forecast windows overlap (consecutive windows share 4 of 5
target days) and hot days come in multi-day episodes, so individual windows /
forecast-day instances are NOT independent. The earlier window-level bootstrap
(evaluation/hw_sweep/bootstrap_ci_15_vs_20.json) treated them as independent,
which makes confidence intervals far too narrow (2-3.6x on the val split).

PRIMARY TEST -- paired_cluster_test (the headline test; if it and dm_test
disagree, report this one and mention the other):
    delta = metric(child) - metric(parent), metric = RMSE or MAE, with a
    delete-one-cluster jackknife standard error (CV3J: centred on the mean of
    the jackknife replicates) and Student-t
    critical values with G-1 degrees of freedom (G = number of clusters).
    Recommended for few clusters by MacKinnon, Nielsen & Webb (2023), J.
    Econometrics 232(2). A percentile cluster bootstrap was tried first and
    rejected: on simulated overlapping-window data with hot-season episodes it
    gave 13% false positives at G=9 and 18-43% on extreme-only subsets (nominal
    5%); this jackknife-t gave 3.5-6.4% in the same simulations.
    With fewer than MIN_CLUSTERS clusters no CI / p-value is returned (NaN).

    Seeds: a model is evaluated over several training seeds; its metric is the
    mean over seeds of the per-seed RMSE/MAE (repo convention). Seed-to-seed
    variance of each model's mean (var_s / n_seeds) is added to the cluster
    variance. Seeds are treated as independent between models, because the same
    seed number does not give comparable initialisations across architectures.
    This is mildly conservative (the seed x cluster interaction is counted twice),
    so power simulations must use this function as configured. A model given as
    a single seed contributes no seed variance (result.seed_variance_included
    says whether any model's seed variance was added); that is correct for deterministic
    baselines but understates training noise for a single-seed neural model.

SECONDARY TEST -- dm_test:
    Diebold-Mariano test of equal predictive accuracy on one time-ordered loss
    series per model (average over seeds first), with a Newey-West (Bartlett)
    long-run variance and an automatic, data-driven lag (Andrews 1991, AR(1)
    plug-in; never below h-1). With lag fixed at h-1 the test over-rejected at
    8-19% on persistent errors; with the automatic lag, 3.5-7.7%. Use it as a
    cross-check on direction/significance, not as the headline test.

Clusters: make_cluster_ids -> year x season blocks (Jan 1-Mar 14, Mar 15-Jul 31,
Aug 1-Dec 31), keyed on the forecast (target) date for instance-level errors.

Conventions: delta = child - parent; negative delta = child better (lower error).
Errors are (prediction - truth) per instance, flattened over windows x lead days
(e.g. `errors.ravel()` on an (n_windows, 5) array). A model with several seeds
is passed as a LIST of 1-D arrays, one per seed; 2-D arrays are rejected so
that lead days can never be mistaken for seeds.

References: Diebold & Mariano (1995) JBES 13(3); Newey & West (1987)
Econometrica 55(3); Andrews (1991) Econometrica 59(3); MacKinnon, Nielsen &
Webb (2023) J. Econometrics 232(2).
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

METRICS = ("rmse", "mae")
MIN_CLUSTERS = 5  # below this, no CI / p-value is reported
FEW_CLUSTERS = 10  # below this, results are flagged as fragile

_SEASON_CODES = {"pre": 0, "hot": 1, "post": 2}


@dataclass(frozen=True)
class ClusterTestResult:
    metric: str
    parent: float  # parent's metric (mean over seeds of per-seed metric)
    child: float
    delta: float  # child - parent; negative = child better
    se: float  # jackknife (+ seed) standard error of delta
    ci_low: float  # NaN if n_clusters < MIN_CLUSTERS
    ci_high: float
    ci_level: float
    p_value: float  # two-sided, t(G-1); NaN if n_clusters < MIN_CLUSTERS
    df: int  # G - 1
    n_instances: int
    n_clusters: int
    clusters_child_better: int  # descriptive sign count across clusters
    n_seeds_parent: int
    n_seeds_child: int
    seed_variance_included: bool  # True if any model's seed variance was added (models with 1 seed add none)
    few_clusters: bool  # n_clusters < FEW_CLUSTERS

    @property
    def significant(self) -> bool:
        """CI excludes zero (False when no CI could be computed)."""
        return bool(self.ci_low > 0 or self.ci_high < 0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        return d


@dataclass(frozen=True)
class DMResult:
    statistic: float  # negative = model A has lower loss; NaN if degenerate
    p_value: float  # two-sided, t(n-1)
    mean_diff: float  # mean of d_t = loss_a - loss_b
    lag: int  # HAC lag actually used
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- clusters


def make_cluster_ids(dates) -> np.ndarray:
    """Year x season block id per date: int(year * 10 + season_code).

    Seasons: 0 = Jan 1 - Mar 14, 1 = Mar 15 - Jul 31 (the labels-v2 heatwave
    season), 2 = Aug 1 - Dec 31. Hot-season episodes (labels v2) therefore never
    span two clusters. Off-season warm spells (labels v1) can straddle the
    Mar 14/15 boundary; that leaks very little dependence (measured adjacent-
    cluster error correlation ~0) but is worth knowing.
    Pass the forecast (target) date for instance-level errors.
    """
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    if d.hasnans:
        raise ValueError("dates contain NaT")
    md = d.month * 100 + d.day
    season = np.where(md < 315, 0, np.where(md <= 731, 1, 2))
    return (d.year.to_numpy() * 10 + season).astype(np.int64)


# ---------------------------------------------------------------- input handling


def _check_pandas_alignment(*inputs) -> None:
    """Pairing is by position. All Series among the inputs -- including Series
    inside a list of seeds -- must share one index; a Series with a non-default
    index next to plain arrays can't be checked, so warn."""
    flat = []
    for x in inputs:
        flat.extend(x if isinstance(x, (list, tuple)) else [x])
    series = [x for x in flat if isinstance(x, pd.Series)]
    for s in series[1:]:
        if not s.index.equals(series[0].index):
            raise ValueError("pandas inputs have different indexes; align them before comparing")
    default_index = series and series[0].index.equals(pd.RangeIndex(len(series[0])))
    if series and len(series) < len(flat) and not default_index:
        warnings.warn(
            "Mixing a pandas Series that has a custom index with plain arrays: pairing is by "
            "position and cannot be checked. Pass all inputs as aligned Series or all as arrays.",
            stacklevel=3,
        )


def _as_seed_matrix(errors, name: str) -> np.ndarray:
    """1-D array (one seed) or list/tuple of 1-D arrays (one per seed)
    -> float64 (n_seeds, n_instances)."""
    if isinstance(errors, (list, tuple)):
        rows = [np.asarray(e, dtype=np.float64) for e in errors]
        if not rows or any(r.ndim != 1 for r in rows):
            raise ValueError(
                f"{name}: a list always means SEEDS and must contain 1-D arrays, one per seed. "
                "For a single seed given as a Python list of numbers, pass np.asarray(x)."
            )
        if len({r.shape[0] for r in rows}) != 1:
            raise ValueError(f"{name}: all seeds must have the same number of instances")
        arr = np.stack(rows)
    else:
        arr = np.asarray(errors, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                f"{name} has shape {arr.shape}. Pass ONE seed as a 1-D array of per-instance "
                "errors (flatten windows x lead days with .ravel()), or SEVERAL seeds as a list "
                "of 1-D arrays. 2-D arrays are rejected so lead days can't be mistaken for seeds."
            )
        arr = arr[None, :]
    if arr.shape[1] == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    with np.errstate(over="ignore"):
        squares_finite = np.all(np.isfinite(arr**2))
    if not squares_finite:
        raise ValueError(f"{name} overflows when squared")
    return arr


# ---------------------------------------------------------------- primary test


def _metric(loss_sum: np.ndarray, count, metric: str) -> np.ndarray:
    mean_loss = loss_sum / count
    return np.sqrt(mean_loss) if metric == "rmse" else mean_loss


def paired_cluster_test(
    *,
    parent,
    child,
    cluster_ids,
    metric: str = "rmse",
    ci_level: float = 0.95,
    include_seed_variance: bool = True,
) -> ClusterTestResult:
    """Paired test of delta = metric(child) - metric(parent) with a cluster-
    jackknife (CV3J) standard error and t(G-1) inference.

    parent, child: forecast errors on the SAME instances in the same order:
        a 1-D array (one seed) or a list of 1-D arrays (one per seed); seed
        counts may differ between the two models.
    cluster_ids: (n_instances,) cluster label per instance (make_cluster_ids).
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be in (0, 1)")
    _check_pandas_alignment(parent, child, cluster_ids)
    e_p = _as_seed_matrix(parent, "parent")
    e_c = _as_seed_matrix(child, "child")
    if pd.api.types.is_datetime64_any_dtype(np.asarray(cluster_ids)):
        warnings.warn(
            "cluster_ids are dates: each day becomes its own cluster, which ignores the "
            "dependence between overlapping windows. Use make_cluster_ids(dates).",
            stacklevel=2,
        )
    clusters = np.asarray(cluster_ids, dtype=object)
    if clusters.ndim != 1 or not (e_p.shape[1] == e_c.shape[1] == clusters.shape[0]):
        raise ValueError(
            "parent, child and cluster_ids must cover the same instances: "
            f"got {e_p.shape[1]}, {e_c.shape[1]}, {clusters.shape}"
        )
    if pd.isnull(clusters).any():
        raise ValueError("cluster_ids contains missing values")

    codes, _ = pd.factorize(clusters)  # works for mixed int/str labels
    n_clusters = int(codes.max()) + 1
    if n_clusters < 2:
        raise ValueError("need at least 2 clusters")
    count = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    if count.mean() < 10:
        warnings.warn(
            f"{n_clusters} clusters for {clusters.shape[0]} instances (mean {count.mean():.1f} per "
            "cluster): clusters this small cannot absorb the dependence between overlapping "
            "windows. Use make_cluster_ids (year x season blocks).",
            stacklevel=2,
        )

    def per_cluster_loss(e: np.ndarray) -> np.ndarray:  # (n_seeds, G)
        loss = e**2 if metric == "rmse" else np.abs(e)
        return np.stack([np.bincount(codes, weights=row, minlength=n_clusters) for row in loss])

    loss_p, loss_c = per_cluster_loss(e_p), per_cluster_loss(e_c)
    tot_p, tot_c, n_tot = loss_p.sum(axis=1), loss_c.sum(axis=1), count.sum()

    seed_metric_p = _metric(tot_p, n_tot, metric)  # (n_seeds,)
    seed_metric_c = _metric(tot_c, n_tot, metric)
    m_p, m_c = float(seed_metric_p.mean()), float(seed_metric_c.mean())
    delta = m_c - m_p

    # Delete-one-cluster jackknife of delta (each model: mean over seeds).
    keep_n = n_tot - count  # (G,)
    jack_p = _metric(tot_p[:, None] - loss_p, keep_n[None, :], metric).mean(axis=0)  # (G,)
    jack_c = _metric(tot_c[:, None] - loss_c, keep_n[None, :], metric).mean(axis=0)
    jack = jack_c - jack_p
    var = (n_clusters - 1) / n_clusters * float(np.sum((jack - jack.mean()) ** 2))
    if include_seed_variance:
        for sm in (seed_metric_p, seed_metric_c):
            if sm.size > 1:
                var += float(np.var(sm, ddof=1)) / sm.size
    se = float(np.sqrt(var))

    # Descriptive: in how many clusters is the child better (seed-mean metric)?
    clus_p = _metric(loss_p, count[None, :], metric).mean(axis=0)
    clus_c = _metric(loss_c, count[None, :], metric).mean(axis=0)
    child_better = int(np.sum(clus_c < clus_p))

    df = n_clusters - 1
    if n_clusters < MIN_CLUSTERS:
        warnings.warn(
            f"Only {n_clusters} clusters (< {MIN_CLUSTERS}): no CI or p-value reported. "
            "Report the per-cluster sign count instead, or pool more data (rolling-origin folds).",
            stacklevel=2,
        )
        ci_low = ci_high = p_value = float("nan")
    elif se == 0.0:
        # Identical models (no difference), or a constant offset in every cluster
        # with no sampling variability to test against (NaN, as in dm_test).
        if delta == 0.0:
            ci_low = ci_high = 0.0
            p_value = 1.0
        else:
            ci_low = ci_high = p_value = float("nan")
    else:
        crit = sp_stats.t.ppf(0.5 + ci_level / 2, df)
        ci_low, ci_high = delta - crit * se, delta + crit * se
        p_value = float(2 * sp_stats.t.sf(abs(delta / se), df))

    return ClusterTestResult(
        metric=metric,
        parent=m_p,
        child=m_c,
        delta=float(delta),
        se=se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        ci_level=ci_level,
        p_value=float(p_value),
        df=df,
        n_instances=int(clusters.shape[0]),
        n_clusters=n_clusters,
        clusters_child_better=child_better,
        n_seeds_parent=e_p.shape[0],
        n_seeds_child=e_c.shape[0],
        seed_variance_included=bool(include_seed_variance and (e_p.shape[0] > 1 or e_c.shape[0] > 1)),
        few_clusters=n_clusters < FEW_CLUSTERS,
    )


# ---------------------------------------------------------------- secondary test


def newey_west_variance(d, lag: int) -> float:
    """Long-run variance, Bartlett kernel: gamma_0 + 2 * sum_k (1 - k/(lag+1)) gamma_k."""
    x = np.asarray(d, dtype=np.float64)
    x = x - x.mean()
    n = x.shape[0]
    var = float(np.dot(x, x) / n)
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * float(np.dot(x[k:], x[:-k]) / n)
    return var


def andrews_lag(d, horizon: int = 5) -> int:
    """Automatic Bartlett bandwidth (Andrews 1991, AR(1) plug-in), never below
    horizon - 1 and never above n // 4. Returned as lag = ceil(S_T); with weights
    1 - k/(lag+1) this smooths slightly more than Andrews' S_T (conservative)."""
    x = np.asarray(d, dtype=np.float64)
    x = x - x.mean()
    n = x.shape[0]
    denom = float(np.dot(x[:-1], x[:-1]))
    rho = 0.0 if denom == 0 else float(np.clip(np.dot(x[1:], x[:-1]) / denom, -0.97, 0.97))
    alpha = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    lag = int(np.ceil(1.1447 * (alpha * n) ** (1 / 3)))
    return int(min(max(horizon - 1, lag), n // 4))


def dm_test(loss_a, loss_b, lag: int | None = None, horizon: int = 5, dates=None) -> DMResult:
    """Diebold-Mariano test of equal expected loss (secondary check).

    loss_a, loss_b: per-period losses of two models on the same time-ordered
        periods (e.g. per-window MSE, ordered by query date; average seeds first).
    lag: HAC lag; None (default) = andrews_lag(d, horizon).
    dates: optional period dates; if given, they must be strictly increasing.
    Statistic < 0 means model A has the lower loss.
    """
    _check_pandas_alignment(loss_a, loss_b)
    if dates is not None:
        d_idx = pd.DatetimeIndex(pd.to_datetime(dates))
        if len(d_idx) != len(loss_a) or not d_idx.is_monotonic_increasing or not d_idx.is_unique:
            raise ValueError("dates must match the losses in length and be strictly increasing (sort by date first)")
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"loss_a and loss_b must be 1-D and the same length; got {a.shape}, {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("losses contain NaN or inf")
    n = a.shape[0]
    if n < 4 * horizon:
        raise ValueError(f"series too short (n={n}) for horizon={horizon}")

    d = a - b
    mean_d = float(d.mean())
    if lag is None:
        lag = andrews_lag(d, horizon)
    if lag < 0 or lag >= n:
        raise ValueError(f"invalid lag {lag} for n={n}")

    spread = float(np.ptp(d))
    if spread <= 1e-12 * max(1.0, abs(mean_d)):
        # Constant differential: identical models (no difference) or a
        # deterministic offset (no sampling variability to test against).
        if mean_d == 0.0:
            return DMResult(statistic=0.0, p_value=1.0, mean_diff=0.0, lag=lag, n=n)
        return DMResult(statistic=float("nan"), p_value=float("nan"), mean_diff=mean_d, lag=lag, n=n)

    long_run_var = newey_west_variance(d, lag)
    stat = float(mean_d / np.sqrt(long_run_var / n))
    p_value = float(2 * sp_stats.t.sf(abs(stat), df=n - 1))
    return DMResult(statistic=stat, p_value=p_value, mean_diff=mean_d, lag=lag, n=n)
