"""
RS-5 (pre-registered in context/decisions_retrieval.md): retrieval keyed on a global model's forecast, an analogue
ensemble (Delle Monache et al. 2013) for Delhi's daily maximum temperature, 2016-2018 (development years only).

- Forecasts: ECMWF IFS HRES from WeatherBench 2 (1.5° store, 2 m temperature), runs started 12 UTC on q - 1, the
  four 6-hourly values (00, 06, 12, 18 UTC) of each target day, as an area-weighted mean over the cells overlapping
  the 9-cell Delhi box. Only the box means are kept (DATA_PATH); nothing from 2019 on is downloaded.
- Analogue ensemble: for each query and lead, the K = 20 forecasts of the same lead in the other two years (within
  ±45 days of the same day of year) closest to today's 4-value forecast; the forecast is the mean ERA5 Tmax that
  followed them. Random control: 20 random archive members.
- Compared with C3-hop (fold f4, 10-seed mean) and with HRES's own daily maximum after the GraphCast yardstick's
  per-month correction (fitted on ERA5 1980-2015).

Run from repo root:  python -m retrieval_study.hres_anen [--fetch-only]  ->  retrieval_study/results/step5.{md,json}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = ("https://storage.googleapis.com/weatherbench2/datasets/hres/"
         "2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr")
DATA_PATH = REPO_ROOT / "retrieval_study" / "data" / "hres_2016_2018_delhi_t2m.parquet"
OUT_DIR = REPO_ROOT / "retrieval_study" / "results"
FIRST_INIT = pd.Timestamp("2016-01-01 12:00")
LAST_INIT = pd.Timestamp("2018-12-26 12:00")  # its day-5 target is 2018-12-31: nothing from 2019
LEADS_H = tuple(range(12, 127, 6))  # 12 .. 126 h: the 4 six-hourly values of target days 1-5
K = 20
SEASON_DAYS = 45
SEEDS = tuple(range(10))
N_PAIRS = 200


# ------------------------------------------------------------------ download (box means only)


def box_weights(lat: np.ndarray, lon: np.ndarray, box) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices of the cells overlapping the box and their area weights (lat_idx, lon_idx, (n_lat, n_lon)), for a
    regular grid of cell centres given in any order."""
    from evaluation.graphcast_yardstick import overlap_weights

    dlat, dlon = float(np.abs(np.diff(lat)).mean()), float(np.abs(np.diff(lon)).mean())
    ilat = np.flatnonzero((lat + dlat / 2 > box[0]) & (lat - dlat / 2 < box[1]))
    ilon = np.flatnonzero((lon + dlon / 2 > box[2]) & (lon - dlon / 2 < box[3]))
    ilat, ilon = ilat[np.argsort(lat[ilat])], ilon[np.argsort(lon[ilon])]
    w = overlap_weights(np.r_[lat[ilat] - dlat / 2, lat[ilat[-1]] + dlat / 2],
                        np.r_[lon[ilon] - dlon / 2, lon[ilon[-1]] + dlon / 2], box)
    return ilat, ilon, w


def fetch(path: Path = DATA_PATH, workers: int = 6) -> pd.DataFrame:
    """Box-mean 2 m temperature (deg C) per (init, lead_hours) for the 12 UTC runs FIRST_INIT..LAST_INIT; only the
    chunks holding LEADS_H are read. Cached in `path`."""
    if path.exists():
        return pd.read_parquet(path)
    from concurrent.futures import ThreadPoolExecutor

    import numcodecs
    import requests

    from evaluation.graphcast_yardstick import BOX

    sess = requests.Session()

    def get(name: str) -> bytes:
        for attempt in range(6):
            try:
                r = sess.get(f"{STORE}/{name}", timeout=120)
                r.raise_for_status()
                return r.content
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
        raise AssertionError

    meta = json.loads(get(".zmetadata"))["metadata"]
    za, attrs = meta["2m_temperature/.zarray"], meta["2m_temperature/.zattrs"]
    if attrs["_ARRAY_DIMENSIONS"] != ["time", "prediction_timedelta", "longitude", "latitude"] or za["chunks"][0] != 1:
        raise ValueError(f"unexpected layout: {attrs['_ARRAY_DIMENSIONS']} chunks {za['chunks']}")
    if not meta["time/.zattrs"]["units"].startswith("hours since 2016-01-01"):
        raise ValueError("unexpected time units")

    def coord(name: str) -> np.ndarray:
        z = meta[f"{name}/.zarray"]
        return np.frombuffer(numcodecs.Blosc().decode(get(f"{name}/0")), dtype=z["dtype"])

    times = pd.DatetimeIndex(pd.Timestamp("2016-01-01") + pd.to_timedelta(coord("time"), unit="h"))
    leads = coord("prediction_timedelta")
    lat, lon = coord("latitude"), coord("longitude")
    ilat, ilon, w = box_weights(lat, lon, BOX)
    inits = pd.date_range(FIRST_INIT, LAST_INIT, freq="D")
    if not inits.isin(times).all():
        raise ValueError("a 12 UTC run is missing from the store")
    lead_idx = {int(h): int(np.flatnonzero(leads == h)[0]) for h in LEADS_H}
    lc = za["chunks"][1]
    plan: dict[tuple[int, int], list[tuple[pd.Timestamp, int, int]]] = {}
    for init in inits:
        ti = times.get_loc(init)
        for h, li in lead_idx.items():
            plan.setdefault((ti, li // lc), []).append((init, h, li % lc))

    def box_means(key: tuple[int, int]) -> list[tuple[pd.Timestamp, int, float]]:
        ti, ci = key
        a = np.frombuffer(numcodecs.Blosc().decode(get(f"2m_temperature/{ti}.{ci}.0.0")), dtype=za["dtype"])
        a = a.reshape(za["chunks"])[0]  # (lead, lon, lat)
        return [(init, h, float((a[li][np.ix_(ilon, ilat)].T * w).sum()) - 273.15) for init, h, li in plan[key]]

    rows, keys = [], sorted(plan)
    print(f"HRES: {len(inits)} runs, {len(keys)} chunks to read", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, part in enumerate(pool.map(box_means, keys), 1):
            rows += part
            if i % 300 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)} chunks", flush=True)
    out = pd.DataFrame(rows, columns=["init", "lead_hours", "t2m"]).sort_values(["init", "lead_hours"])
    if out["t2m"].isna().any() or len(out) != len(inits) * len(LEADS_H):
        raise ValueError("missing HRES values in the extract")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


# ------------------------------------------------------------------ the analogue ensemble


def keys_table(hres: pd.DataFrame, query_dates: pd.DatetimeIndex) -> np.ndarray:
    """(n_queries, 5, 4): for each query date q and lead L, the run started 12 UTC on q - 1, its 2 m temperature at
    00, 06, 12 and 18 UTC of the target day q + L - 1."""
    s = hres.set_index(["init", "lead_hours"])["t2m"]
    out = np.full((len(query_dates), 5, 4), np.nan)
    for i, q in enumerate(query_dates):
        init = q - pd.Timedelta(hours=12)
        for L in range(1, 6):
            hours = [12 + 24 * (L - 1) + 6 * k for k in range(4)]
            out[i, L - 1] = [s.get((init, h), np.nan) for h in hours]
    return out


def doy_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a[:, None] - b[None, :])
    return np.minimum(d, 365 - d)


def analogue_ensemble(keys: np.ndarray, truth: np.ndarray, target_doy: np.ndarray, target_year: np.ndarray,
                      k: int = K, season: int = SEASON_DAYS, seed: int | None = None) -> np.ndarray:
    """(n,) forecasts for one lead: the mean truth of the k archive members nearest to each query's key (Euclidean on
    keys standardised per component), from target days in other years within ±season days of the same day of year;
    with a seed, k random members of that archive instead."""
    sd = keys.std(axis=0)
    z = keys / np.where(sd > 0, sd, 1.0)
    allowed = (target_year[:, None] != target_year[None, :]) & (doy_distance(target_doy, target_doy) <= season)
    if seed is None:
        dist = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))
        score = np.where(allowed, -dist, -np.inf)
    else:
        score = np.where(allowed, np.random.default_rng(seed).random(allowed.shape), -np.inf)
    top = np.argpartition(-score, k - 1, axis=1)[:, :k]
    if not np.isfinite(np.take_along_axis(score, top, axis=1)).all():
        raise ValueError("an archive has fewer than k members")
    return truth[top].mean(axis=1)


def key_value_spearman(keys: np.ndarray, anomaly: np.ndarray, target_doy: np.ndarray, target_year: np.ndarray,
                       seed: int = 20261111) -> float:
    """Spearman correlation, over random allowed (query, member) pairs, between key similarity (minus the distance)
    and outcome similarity (minus the absolute difference of the Tmax anomalies)."""
    from scipy.stats import spearmanr

    sd = keys.std(axis=0)
    z = keys / np.where(sd > 0, sd, 1.0)
    allowed = (target_year[:, None] != target_year[None, :]) & (doy_distance(target_doy, target_doy) <= SEASON_DAYS)
    rng = np.random.default_rng(seed)
    ks, vs = [], []
    for i in range(len(z)):
        j = rng.choice(np.flatnonzero(allowed[i]), size=min(N_PAIRS, int(allowed[i].sum())), replace=False)
        ks.append(-np.sqrt(((z[j] - z[i]) ** 2).sum(axis=1)))
        vs.append(-np.abs(anomaly[j] - anomaly[i]))
    return float(spearmanr(np.concatenate(ks), np.concatenate(vs)).statistic)


def run() -> dict:
    from evaluation.compare_v2 import compare, ensemble, rmse_by
    from evaluation.graphcast_yardstick import apply_correction, era5_daily_true_and_sampled, fit_correction
    from evaluation.stats import make_cluster_ids
    from training.folds import fold_daily

    hres = fetch()
    c3 = ensemble(pd.read_parquet(REPO_ROOT / "predictions_v2" / "A1prime_hw5_C3hop" / "f4.parquet"))
    c3 = c3[c3["query_date"] >= FIRST_INIT.normalize() + pd.Timedelta(days=1)].sort_values(["query_date", "lead"])
    q = pd.DatetimeIndex(sorted(c3["query_date"].unique()))
    keys = keys_table(hres, q)
    if np.isnan(keys).any():
        raise ValueError("a query has no HRES forecast")
    d, _ = fold_daily("f4", "t_max", "v2")
    coef = fit_correction(era5_daily_true_and_sampled())  # the yardstick's per-month max correction, ERA5 1980-2015
    frames = {"C3-hop": [], "HRES corrected": [], "AnEn": [], "AnEn-random": []}
    rho = {}
    for L in range(1, 6):
        target = q + pd.Timedelta(days=L - 1)
        truth = d["t_max"].reindex(target).to_numpy(dtype=np.float64)
        anomaly = truth - d["clim_mean_t_max"].reindex(target).to_numpy(dtype=np.float64)
        kL = keys[:, L - 1, :]
        doy, year = target.dayofyear.to_numpy(), target.year.to_numpy()
        preds = {"HRES corrected": apply_correction(kL.max(axis=1), target.month.to_numpy(), coef),
                 "AnEn": analogue_ensemble(kL, truth, doy, year)}
        rho[L] = key_value_spearman(kL, anomaly, doy, year)
        base = pd.DataFrame({"query_date": q, "lead": L, "target_date": target, "actual": truth,
                             "stratum": d["stratum"].reindex(target).to_numpy(), "cluster_id": make_cluster_ids(target)})
        for name, p in preds.items():
            frames[name].append(base.assign(model=name, seed=0, pred=p, error=p - truth))
        for s in SEEDS:
            p = analogue_ensemble(kL, truth, doy, year, seed=s)
            frames["AnEn-random"].append(base.assign(model="AnEn-random", seed=s, pred=p, error=p - truth))
        c = c3[c3["lead"] == L].set_index("query_date").reindex(q)
        if not np.allclose(c["actual"].to_numpy(), truth):
            raise ValueError("C3-hop's truths differ from the daily table")
        frames["C3-hop"].append(base.assign(model="C3-hop", seed=0, pred=c["pred"].to_numpy(),
                                            error=c["pred"].to_numpy() - truth))
    df = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
    res = {"queries": int(len(q)), "first_query": str(q.min().date()), "last_query": str(q.max().date()),
           "rmse": {}, "bias": {}, "by_lead": {}, "tests": {}, "key_value_spearman_by_lead": rho}
    for name, f in df.items():
        res["rmse"][name] = {s: rmse_by(f if s == "all" else f[f["stratum"] == s]) for s in ("all", "extreme")}
        res["bias"][name] = {s: float((f if s == "all" else f[f["stratum"] == s])["error"].mean()) for s in ("all", "extreme")}
        res["by_lead"][name] = [rmse_by(f[f["lead"] == L]) for L in range(1, 6)]
    for a, b in (("C3-hop", "AnEn"), ("AnEn-random", "AnEn"), ("HRES corrected", "AnEn"), ("C3-hop", "HRES corrected")):
        for s in ("all", "extreme"):
            t = compare(df[a], df[b], s)
            res["tests"][f"{b} vs {a}|{s}"] = {k: t[k] for k in ("parent", "child", "delta", "ci_low", "ci_high",
                                                                 "n_clusters")}
    res["extract_sha256"] = __import__("hashlib").sha256(DATA_PATH.read_bytes()).hexdigest()
    return res


def write_report(res: dict) -> None:
    def d(t: dict) -> str:
        if not np.isfinite(t["ci_low"]):
            return f"{t['delta']:+.3f} (no CI, {t['n_clusters']} clusters)"
        mark = " **(CI < 0)**" if t["ci_high"] < 0 else (" (CI > 0)" if t["ci_low"] > 0 else "")
        return f"{t['delta']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]{mark}"

    L = ["# Retrieval study, step 5: retrieval keyed on a global model's forecast (RS-5, Tmax)", "",
         "Pre-registered in `context/decisions_retrieval.md` (RS-5) before any forecast value was downloaded. ECMWF IFS "
         f"HRES (WeatherBench 2, 1.5°), runs started 12 UTC the day before; query dates {res['first_query']} to "
         f"{res['last_query']} ({res['queries']} windows x 5 leads, fold f4's validation years). Analogue ensemble: "
         f"the mean ERA5 Tmax that followed the {K} most similar past HRES forecasts of the same lead (other years, "
         f"±{SEASON_DAYS} days). RMSE in °C. 9 year x season clusters on all days (fragile); heatwave days have too "
         "few for a CI. Descriptive.", "",
         "| Forecast | RMSE all days | Bias all days | RMSE heatwave days | Bias heatwave days |", "|---|---|---|---|---|"]
    for name in ("C3-hop", "HRES corrected", "AnEn", "AnEn-random"):
        L.append(f"| {name} | {res['rmse'][name]['all']:.3f} | {res['bias'][name]['all']:+.2f} | "
                 f"{res['rmse'][name]['extreme']:.3f} | {res['bias'][name]['extreme']:+.2f} |")
    L += ["", "| Comparison | All days | Heatwave days |", "|---|---|---|"]
    pairs = {}
    for key, t in res["tests"].items():
        pairs.setdefault(key.split("|")[0], {})[key.split("|")[1]] = t
    for name, t in pairs.items():
        L.append(f"| {name} | {d(t['all'])} | {d(t['extreme'])} |")
    L += ["", "| Lead | 1 | 2 | 3 | 4 | 5 |", "|---|---|---|---|---|---|"]
    for name in ("C3-hop", "HRES corrected", "AnEn"):
        L.append(f"| {name} | " + " | ".join(f"{x:.3f}" for x in res["by_lead"][name]) + " |")
    L.append("| Key-value Spearman (forecast key) | " + " | ".join(
        f"{res['key_value_spearman_by_lead'][str(i)] if str(i) in res['key_value_spearman_by_lead'] else res['key_value_spearman_by_lead'][i]:+.2f}"
        for i in range(1, 6)) + " |")
    L += ["", f"HRES box-mean extract: `{DATA_PATH.relative_to(REPO_ROOT).as_posix()}`, SHA-256 {res['extract_sha256']}."]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "step5.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="RS-5: analogue ensemble keyed on HRES forecasts.")
    ap.add_argument("--fetch-only", action="store_true")
    args = ap.parse_args(argv)
    if args.fetch_only:
        fetch()
        print(f"wrote {DATA_PATH}")
        return
    res = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "step5.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    write_report(res)
    print(f"wrote {OUT_DIR / 'step5.md'}")


if __name__ == "__main__":
    main()
