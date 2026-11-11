"""
RS-1e (pre-registered in context/decisions_retrieval.md): RAFT in its own kind of setting, on our data. This is a
positive control for the daily null result.

Delhi's hourly 2 m temperature (the 9-cell mean, IST hours). Rows from 2019 on are dropped right after reading. Train
on 1980-2015 (settings chosen on 2014-2015), validate on 2016-2018. Queries are issued every 6 hours. Windows are 12 or
96 hours; the forecast is the next 48 hourly values.
- Bases: H0, the window minus its last value; H1, H0 plus the phase (sine and cosine of the issue hour and of the day
  of year).
- Arms: none, RAFT (retrieval_study.raft over periods of 1, 2 and 4 hours), RAFT-rand (3 seeds).
- Model: ridge as in the daily testbed. Score: RMSE over the 48 values. Paired cluster jackknife over year x season
  clusters; 9 on validation, so the result is descriptive.

Run from repo root:  python -m retrieval_study.hourly  ->  retrieval_study/results/step1e.{md,json}
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.compare_v2 import compare, rmse_by
from evaluation.stats import make_cluster_ids
from retrieval_study import raft
from retrieval_study.testbed import ALPHAS, ridge_path, ridge_predict, rmse

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "retrieval_study" / "results"
TEST_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2016-01-01")  # exclusive
INNER_START = pd.Timestamp("2014-01-01")
HORIZON = 48
WINDOWS = (12, 96)
ISSUE_HOURS = (0, 6, 12, 18)
CAND_STRIDE = 3
N_TRAIN_QUERIES = 15000
RANDOM_SEEDS = (0, 1, 2)
QUERY_SEED = 20261111
CHUNK = 256
BASES = ("H0", "H1")


def delhi_hourly() -> pd.Series:
    """The 9-cell mean hourly temperature before 2019 (later rows are dropped as soon as each cell is read)."""
    from pipeline.hourly_features import N_CELLS, load_cell_hourly

    cells = []
    for c in range(1, N_CELLS + 1):
        h = load_cell_hourly(c)
        h = h[h["time"] < TEST_START]
        cells.append(h.set_index("time")["t"].astype(np.float64))
    s = pd.concat(cells, axis=1).mean(axis=1)
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    if not s.index.equals(full) or s.isna().any():
        raise ValueError("the hourly series must be complete")
    return s


@dataclass
class HourlyWindows:
    pos: np.ndarray  # position of each window's first forecast hour
    time: pd.DatetimeIndex
    inputs: np.ndarray  # (n, S)
    last: np.ndarray
    future: np.ndarray  # (n, HORIZON)


def make_windows(x: np.ndarray, times: pd.DatetimeIndex, pos: np.ndarray, s: int) -> HourlyWindows:
    pos = np.asarray(pos, dtype=np.int64)
    if (pos < s).any() or (pos + HORIZON > len(x)).any():
        raise ValueError("window hours outside the series")
    inputs = x[pos[:, None] + np.arange(-s, 0)[None, :]]
    future = x[pos[:, None] + np.arange(HORIZON)[None, :]]
    return HourlyWindows(pos=pos, time=times[pos], inputs=inputs, last=inputs[:, -1].copy(), future=future)


def plan(times: pd.DatetimeIndex, s: int) -> dict[str, np.ndarray]:
    """Positions of the candidate windows and of the training (fit, stop) and validation queries for window s."""
    n = len(times)
    p = np.arange(n)
    hour = times.hour.to_numpy()
    end = times + pd.Timedelta(hours=HORIZON)  # exclusive end of each window's forecast
    starts_ok = (p >= s) & (p + HORIZON <= n)
    in_train = starts_ok & (end <= TRAIN_END)
    issue = np.isin(hour, ISSUE_HOURS)
    cands = np.flatnonzero(in_train & (hour % CAND_STRIDE == 0))
    early = np.flatnonzero(in_train & issue & (end <= INNER_START))
    inner = np.flatnonzero(in_train & issue & (times >= INNER_START))
    rng = np.random.default_rng(QUERY_SEED)
    fit = np.sort(rng.choice(early, size=min(N_TRAIN_QUERIES, len(early)), replace=False))
    val = np.flatnonzero(starts_ok & issue & (times >= TRAIN_END) & (end <= TEST_START))
    return {"cand": cands, "fit": fit, "stop": inner, "val": val}


def phase(times: pd.DatetimeIndex) -> np.ndarray:
    h = 2 * np.pi * times.hour.to_numpy() / 24
    d = 2 * np.pi * (times.dayofyear.to_numpy() - 1) / 365.25
    return np.column_stack([np.sin(h), np.cos(h), np.sin(d), np.cos(d)])


def base(w: HourlyWindows, name: str) -> np.ndarray:
    x = w.inputs - w.last[:, None]
    return x if name == "H0" else np.column_stack([x, phase(w.time)])


def long_frame(w: HourlyWindows, pred: np.ndarray, model: str, seed: int = 0) -> pd.DataFrame:
    n = len(w.pos)
    issue = np.repeat(w.time.to_numpy(), HORIZON)
    lead = np.tile(np.arange(1, HORIZON + 1), n)
    target = issue + (lead - 1).astype("timedelta64[h]")
    df = pd.DataFrame({"model": model, "seed": seed, "query_date": issue, "lead": lead, "target_date": target,
                       "pred": pred.reshape(-1), "actual": w.future.reshape(-1), "stratum": "normal"})
    df["error"] = df["pred"] - df["actual"]
    df["cluster_id"] = make_cluster_ids(pd.DatetimeIndex(df["target_date"]).normalize())
    return df


def fit_predict(x_tr: np.ndarray, x_va: np.ndarray, w_tr: HourlyWindows, w_va: HourlyWindows,
                fit: np.ndarray, stop: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Ridge strength by RMSE on the stop rows, then refit on fit + stop (as retrieval_study.testbed)."""
    y = w_tr.future - w_tr.last[:, None]
    scores = [rmse(ridge_predict(m, x_tr[stop]) + w_tr.last[stop, None], w_tr.future[stop])
              for m in ridge_path(x_tr[fit], y[fit])]
    best = int(np.argmin(scores))
    model = ridge_path(x_tr, y, (ALPHAS[best],))[0]
    return ridge_predict(model, x_va) + w_va.last[:, None], ALPHAS[best], scores[best]


def run_window(x: np.ndarray, times: pd.DatetimeIndex, s: int) -> dict:
    pl = plan(times, s)
    tr_pos = np.concatenate([pl["fit"], pl["stop"]])
    fit = np.arange(len(tr_pos)) < len(pl["fit"])
    stop = ~fit
    w_tr, w_va, w_c = (make_windows(x, times, p, s) for p in (tr_pos, pl["val"], pl["cand"]))
    c_values = raft.offset_values(w_c.future, w_c.last)
    overlap = s + HORIZON

    def nb(seed):
        return (raft.neighbours(w_tr.inputs, w_tr.pos, w_c.inputs, w_c.pos, seed=seed, overlap=overlap, chunk=CHUNK),
                raft.neighbours(w_va.inputs, w_va.pos, w_c.inputs, w_c.pos, seed=seed, overlap=overlap, chunk=CHUNK))

    real = nb(None)
    rand = {sd: nb(int(np.random.SeedSequence([QUERY_SEED, s, sd]).generate_state(1)[0])) for sd in RANDOM_SEEDS}
    frames, settings = {}, {}
    for b in BASES:
        xb_tr, xb_va = base(w_tr, b), base(w_va, b)
        pred, alpha, score = fit_predict(xb_tr, xb_va, w_tr, w_va, fit, stop)
        frames[(b, "none")] = long_frame(w_va, pred, f"S{s}/{b}/none")
        settings[(b, "none")] = {"alpha": alpha, "stop_rmse": score}
        best = None
        for m in raft.M_GRID:
            r_tr, r_va = (raft.retrieved_values(n, c_values, m) for n in real)
            out = fit_predict(np.column_stack([xb_tr, r_tr]), np.column_stack([xb_va, r_va]), w_tr, w_va, fit, stop)
            if best is None or out[2] < best[2]:
                best = (*out, m)
        frames[(b, "RAFT")] = long_frame(w_va, best[0], f"S{s}/{b}/RAFT")
        settings[(b, "RAFT")] = {"alpha": best[1], "stop_rmse": best[2], "m": best[3]}
        parts = []
        for sd, n in rand.items():
            r_tr, r_va = (raft.retrieved_values(z, c_values, best[3]) for z in n)
            out = fit_predict(np.column_stack([xb_tr, r_tr]), np.column_stack([xb_va, r_va]), w_tr, w_va, fit, stop)
            parts.append(long_frame(w_va, out[0], f"S{s}/{b}/RAFT-rand", seed=sd))
        frames[(b, "RAFT-rand")] = pd.concat(parts, ignore_index=True)
    res = {"queries": {k: int(len(v)) for k, v in pl.items()}, "rmse": {}, "tests": {},
           "settings": {f"{b}/{a}": v for (b, a), v in settings.items()}}
    for (b, a), df in frames.items():
        res["rmse"][f"{b}/{a}"] = rmse_by(df)
    for b in BASES:
        for arm in ("RAFT", "RAFT-rand"):
            res["tests"][f"{b}|{arm} vs none"] = _test(frames[(b, "none")], frames[(b, arm)])
        res["tests"][f"{b}|RAFT vs RAFT-rand"] = _test(frames[(b, "RAFT-rand")], frames[(b, "RAFT")])
    return res


def _test(parent: pd.DataFrame, child: pd.DataFrame) -> dict:
    r = compare(parent, child, "all")
    return {k: r[k] for k in ("parent", "child", "delta", "ci_low", "ci_high", "p_value", "n_clusters", "few_clusters")}


def predictions(res: dict) -> dict:
    t12, t96 = res["S12"]["tests"], res["S96"]["tests"]
    p8 = t12["H0|RAFT vs none"]["ci_high"] < 0 and t12["H0|RAFT vs RAFT-rand"]["ci_high"] < 0
    g12, g96 = -t12["H0|RAFT vs none"]["delta"], -t96["H0|RAFT vs none"]["delta"]
    p9a = g12 > 0 and g12 >= 2 * max(g96, 0.0)
    p9b = all(-res[f"S{s}"]["tests"]["H1|RAFT vs none"]["delta"] < -res[f"S{s}"]["tests"]["H0|RAFT vs none"]["delta"]
              for s in WINDOWS)
    return {"P8": bool(p8), "P9 (gain at 12 h at least twice the gain at 96 h)": bool(p9a),
            "P9 (smaller gain once the phase is given)": bool(p9b)}


def _d(t: dict) -> str:
    if not np.isfinite(t["ci_low"]):
        return f"{t['delta']:+.4f} (no CI)"
    mark = " **(CI < 0)**" if t["ci_high"] < 0 else (" (CI > 0)" if t["ci_low"] > 0 else "")
    return f"{t['delta']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]{mark}"


def write_report(res: dict) -> None:
    L = ["# Retrieval study, step 1e: RAFT in its own kind of setting (hourly, positive control)", "",
         "Pre-registered in `context/decisions_retrieval.md` (RS-1e) before it was run. Delhi's hourly 2 m temperature "
         "(9-cell mean); train 1980-2015, validate 2016-2018; queries every 6 hours; forecast = the next 48 hourly "
         "values. RMSE in °C over the 48 values. Δ = RMSE of the second minus the first (negative = retrieval helps). "
         "95% CIs: paired cluster jackknife, 9 year x season clusters (fragile; descriptive).", ""]
    for s in WINDOWS:
        r = res[f"S{s}"]
        L += [f"## Window {s} hours", "",
              f"Queries: {r['queries']['fit']} fitting, {r['queries']['stop']} for choosing settings, "
              f"{r['queries']['val']} validation; {r['queries']['cand']} candidate windows.", "",
              "| Base | RMSE none | RMSE RAFT | RAFT vs none | RAFT-rand vs none | RAFT vs RAFT-rand | RAFT's m |",
              "|---|---|---|---|---|---|---|"]
        for b in BASES:
            t = r["tests"]
            label = "H0 (window only)" if b == "H0" else "H1 (window + phase)"
            L.append(f"| {label} | {r['rmse'][f'{b}/none']:.4f} | {r['rmse'][f'{b}/RAFT']:.4f} | "
                     f"{_d(t[f'{b}|RAFT vs none'])} | {_d(t[f'{b}|RAFT-rand vs none'])} | {_d(t[f'{b}|RAFT vs RAFT-rand'])} | "
                     f"{r['settings'][f'{b}/RAFT']['m']} |")
        L.append("")
    L += ["## The predictions (automatic reading)", "", "| Prediction | Holds |", "|---|---|"]
    L += [f"| {k} | {'yes' if v else 'no'} |" for k, v in res["predictions"].items()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "step1e.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    series = delhi_hourly()
    x, times = series.to_numpy(), series.index
    res = {}
    for s in WINDOWS:
        res[f"S{s}"] = run_window(x, times, s)
        print(f"window {s} h done ({time.time() - t0:.0f} s)", flush=True)
    res["predictions"] = predictions(res)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "step1e.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    write_report(res)
    print(f"wrote {OUT_DIR / 'step1e.md'}")


if __name__ == "__main__":
    main()
