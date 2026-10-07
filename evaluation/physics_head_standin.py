"""
Physics-head design study (plan v5, Week 4 preparation): exact Liljegren WBGT vs a learned
stand-in. Measures the numbers the team needs before choosing (memory: user leans exact,
accepts the stand-in only on evidence). No training code is touched.

The target the models forecast is wbgt_lj_max = mean over Delhi's 9 cells of each cell's
daily MAXIMUM of HOURLY Liljegren WBGT. A physics head predicts a few ingredients per day,
not 9 hourly profiles, so even the exact formula has a representation error. Three numbers:

  R1  representation, peak hour: the exact formula applied to the 9-cell MEAN hourly inputs,
      taking the day's maximum (= the formula at the peak hour of the mean profile), vs the
      true target. The floor for a head that predicts peak-hour ingredients, exact or not.
  R2  representation, fixed hour: the exact formula on the 9-cell mean inputs at one fixed
      hour (the most common peak hour) vs the true target. A simpler head (no peak-hour choice).
  S   stand-in fidelity: a small neural network trained to reproduce the exact formula from
      its inputs (air temperature, humidity, pressure, 10 m wind, global radiation, direct
      fraction, sun angle), compared with the exact formula on the same inputs. This is the
      exact-vs-stand-in question itself.

Data: hourly raw ERA5 for the 9 cells; rows from 2019 on are dropped as soon as they are
read (test lock). Stand-in trained on 1980-2015 hours (early stopping on 2014-2015), scored
on 2016-2018. Hot days = WBGT-label days of fold f4 (Mar 15-Sep 30, target >= 36.207 C).

Run from repo root:  python -m evaluation.physics_head_standin
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from pipeline.build_wbgt_liljegren import IST_OFFSET, load_cell_radiation
from pipeline.hourly_features import load_cell_hourly
from pipeline.labels_v2 import in_wbgt_season
from pipeline.wbgt_liljegren import mean_sunlit_cosz, wbgt
from training.folds import TEST_START

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "evaluation_v2"
TRAIN_END, STOP_START = pd.Timestamp("2015-12-31 23:00"), pd.Timestamp("2014-01-01")
HOT_THRESHOLD = 36.207  # configs/wbgt_label.json, fold f4
INPUTS = ["t", "rh", "pressure", "wind", "ghi", "fdir", "cosz"]
THREADS = 2  # leave the CPU to the training queue


def cell_hourly(cell: int) -> pd.DataFrame:
    """Hourly inputs and exact WBGT of one cell, pre-2019 rows only."""
    met = load_cell_hourly(cell)
    rad, (lat, lon) = load_cell_radiation(cell)
    df = met.merge(rad, on="time", how="inner").dropna()
    df = df[df["time"] < TEST_START].reset_index(drop=True)  # test lock: drop 2019+ before any computation
    utc = (df["time"] - IST_OFFSET).to_numpy(dtype="datetime64[ns]")
    df["cosz"] = np.concatenate([mean_sunlit_cosz(c, lat, lon) for c in np.array_split(utc, max(1, len(utc) // 50_000))])
    df["direct"] = np.clip(df["ghi"].to_numpy() - df["diffuse"].to_numpy(), 0.0, None)
    df["wbgt"] = wbgt(df["t"], df["rh"], df["pressure"], df["wind"], df["ghi"], df["direct"], df["cosz"])["wbgt"]
    return df


def exact(df: pd.DataFrame) -> np.ndarray:
    return wbgt(df["t"], df["rh"], df["pressure"], df["wind"], df["ghi"], df["direct"], df["cosz"])["wbgt"]


def stats(err: np.ndarray) -> dict:
    a = np.abs(err)
    return {"n": int(len(err)), "rmse": float(np.sqrt(np.mean(err**2))), "bias": float(np.mean(err)),
            "p99_abs": float(np.quantile(a, 0.99)), "max_abs": float(a.max())}


class StandIn(nn.Module):
    def __init__(self, n_in: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_standin(X: np.ndarray, y: np.ndarray, stop: np.ndarray, seed: int = 0, epochs: int = 60):
    """Small MLP reproducing the exact formula; returns a numpy predict function."""
    torch.manual_seed(seed)
    mu, sd = X[~stop].mean(axis=0), X[~stop].std(axis=0)
    ym, ys = y[~stop].mean(), y[~stop].std()
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32)
    tr = torch.from_numpy(np.flatnonzero(~stop))
    st = torch.from_numpy(np.flatnonzero(stop))
    model = StandIn(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for b in tr[torch.randperm(len(tr))].split(4096):
            opt.zero_grad()
            loss = torch.mean((model(xt[b]) - yt[b]) ** 2)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = torch.mean((model(xt[st]) - yt[st]) ** 2).item()
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)

    def predict(Xn: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return model(torch.tensor((Xn - mu) / sd, dtype=torch.float32)).numpy() * ys + ym

    return predict


def main() -> None:
    torch.set_num_threads(THREADS)
    t0 = time.time()
    cells = [cell_hourly(c) for c in range(1, 10)]
    # true target: mean over cells of each cell's daily max (complete days only), as build_wbgt_liljegren
    per_cell = []
    for c in cells:
        g = c.assign(date=c["time"].dt.normalize()).groupby("date")["wbgt"]
        per_cell.append(g.max()[g.size() == 24])
    target = pd.concat(per_cell, axis=1).dropna().mean(axis=1)
    ref = pd.read_parquet(REPO_ROOT / "datasets_v2/wbgt_liljegren_daily.parquet",
                          filters=[("date", "<", TEST_START)]).set_index("date")["wbgt_lj_max"]
    check = float(np.abs(target - ref.reindex(target.index)).max())

    # 9-cell mean hourly profile (what a single-profile head would predict)
    cols = ["t", "rh", "pressure", "wind", "ghi", "direct", "cosz"]
    mean = pd.concat([c.set_index("time")[cols] for c in cells]).groupby(level=0).mean()
    mean["fdir"] = np.clip(np.divide(mean["direct"], mean["ghi"], out=np.zeros(len(mean)), where=mean["ghi"] > 0), 0, 1)
    mean["wbgt_exact"] = exact(mean)
    mean["date"], mean["hour"] = mean.index.normalize(), mean.index.hour
    peak_idx = mean.groupby("date")["wbgt_exact"].idxmax()
    peak = mean.loc[peak_idx].set_index("date")
    peak = peak.loc[peak.index.isin(target.index)]
    fixed_hour = int(peak["hour"].mode().iloc[0])
    fixed = mean[mean["hour"] == fixed_hour].set_index("date").reindex(peak.index)

    days = peak.index
    ev = (days >= "2016-01-01")
    hot = ev & in_wbgt_season(days) & (target.reindex(days).to_numpy() >= HOT_THRESHOLD)
    tgt = target.reindex(days).to_numpy()
    res = {"check_target_vs_dataset_max_abs": check, "fixed_hour_ist": fixed_hour,
           "peak_hour_share": {int(h): float(v) for h, v in peak["hour"].value_counts(normalize=True).head(5).items()}}
    res["R1_peak_hour_exact_vs_target"] = {"eval_all_days": stats(peak["wbgt_exact"].to_numpy()[ev] - tgt[ev]),
                                           "eval_hot_days": stats(peak["wbgt_exact"].to_numpy()[hot] - tgt[hot])}
    fx = fixed["wbgt_exact"].to_numpy()
    res["R2_fixed_hour_exact_vs_target"] = {"eval_all_days": stats(fx[ev] - tgt[ev]), "eval_hot_days": stats(fx[hot] - tgt[hot])}

    # stand-in: reproduce the exact formula from its inputs (hourly 9-cell mean rows)
    rows = mean[mean.index <= pd.Timestamp("2018-12-31 23:00")]
    X, y = rows[INPUTS].to_numpy(np.float64), rows["wbgt_exact"].to_numpy()
    is_train = rows.index <= TRAIN_END
    stop = (rows.index >= STOP_START)[is_train]
    predict = fit_standin(X[is_train], y[is_train], stop)
    ev_rows = ~is_train
    sur = predict(X[ev_rows])
    res["S_standin_vs_exact"] = {"eval_all_hours": stats(sur - y[ev_rows])}
    pk = peak.loc[ev]
    pk_hot = peak.loc[hot]
    res["S_standin_vs_exact"]["eval_peak_hours"] = stats(predict(pk[INPUTS].to_numpy()) - pk["wbgt_exact"].to_numpy())
    res["S_standin_vs_exact"]["eval_peak_hours_hot_days"] = stats(predict(pk_hot[INPUTS].to_numpy()) - pk_hot["wbgt_exact"].to_numpy())
    res["S_end_to_end_peak_hour_vs_target"] = {
        "eval_all_days": stats(predict(pk[INPUTS].to_numpy()) - tgt[ev]),
        "eval_hot_days": stats(predict(pk_hot[INPUTS].to_numpy()) - tgt[hot])}
    res["n_train_hours"], res["n_eval_days"], res["n_eval_hot_days"] = int(is_train.sum()), int(ev.sum()), int(hot.sum())
    res["seconds"] = round(time.time() - t0, 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "physics_head_standin.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    f = lambda s: f"RMSE {s['rmse']:.3f}, bias {s['bias']:+.3f}, 99% of errors within {s['p99_abs']:.3f}, max {s['max_abs']:.3f}"  # noqa: E731
    L = ["# Physics head: exact formula vs learned stand-in (design study)", "",
         "Generated by `python -m evaluation.physics_head_standin`. Errors in °C, scored on 2016-2018 (stand-in trained "
         f"on 1980-2015); no 2019+ data. Hot days = WBGT-label days (n={res['n_eval_hot_days']} of {res['n_eval_days']}). "
         f"Sanity check: rebuilt target vs dataset max difference {check:.2e} °C.", "",
         "## S. Does the stand-in reproduce the exact formula? (the exact-vs-stand-in question)", "",
         f"- All hours: {f(res['S_standin_vs_exact']['eval_all_hours'])}",
         f"- Peak hours: {f(res['S_standin_vs_exact']['eval_peak_hours'])}",
         f"- Peak hours on hot days: {f(res['S_standin_vs_exact']['eval_peak_hours_hot_days'])}", "",
         "## R. What any daily physics head loses (exact formula, perfect ingredients)", "",
         f"- R1, ingredients of the 9-cell average at the day's peak hour: all days {f(res['R1_peak_hour_exact_vs_target']['eval_all_days'])}; "
         f"hot days {f(res['R1_peak_hour_exact_vs_target']['eval_hot_days'])}",
         f"- R2, the same at a fixed hour ({fixed_hour}:00 IST, the most common peak): all days "
         f"{f(res['R2_fixed_hour_exact_vs_target']['eval_all_days'])}; hot days {f(res['R2_fixed_hour_exact_vs_target']['eval_hot_days'])}",
         f"- Peak-hour shares: {res['peak_hour_share']}", "",
         "## Stand-in end to end (stand-in at the peak hour vs the true target)", "",
         f"- All days: {f(res['S_end_to_end_peak_hour_vs_target']['eval_all_days'])}",
         f"- Hot days: {f(res['S_end_to_end_peak_hour_vs_target']['eval_hot_days'])}", "",
         "Reading: if S is much smaller than R1, the choice between exact and stand-in barely matters for accuracy, "
         "because the daily representation (R) dominates; exactness is then about faithfulness, not skill."]
    (OUT_DIR / "physics_head_standin.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[4:]))


if __name__ == "__main__":
    main()
