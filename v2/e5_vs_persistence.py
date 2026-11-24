"""
Post-hoc, descriptive (added 24 Nov AFTER the E5 result and AFTER e5_baselines showed persistence matching E5 at
lead 1): paired comparison of the C3-hop trigger against the persistence trigger on the SAME validation days.
Both triggers use the same CRC procedure, alpha, folds and event (rest-hours >= R). Model fire = share of the 10
seeds that fire (as in calibration_v2.e5); persistence fire = 0/1.
Differences are model minus persistence; uncertainty is the delete-one-cluster (year x season) jackknife, CV3J,
t with G-1 df, two-sided. Not part of the Holm family; reported as context for the E5 claim.
Writes results_v2/e5_vs_persistence.json.

Run from repo root:  python -m v2.e5_vs_persistence
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from evaluation.stats import make_cluster_ids
from v2.calibration_v2 import ALPHA, in_season
from v2.e5_baselines import FOLD_YEARS, scores
from v2.risk_control import crc_threshold
from v2.run_calibration_v2 import FOLDS, R, RESULTS, RUNS, load, rest_hours


def model_fire(inner: pd.DataFrame, val: pd.DataFrame, ev: pd.Series, lead: int, alpha: float) -> pd.Series:
    out = []
    for f in FOLDS:
        i, v = inner[(inner["fold"] == f) & (inner["lead"] == lead)], val[(val["fold"] == f) & (val["lead"] == lead)]
        for seed, vs in v.groupby("seed"):
            c = i[(i["seed"] == seed) & in_season(i["target_date"]) & i["target_date"].isin(ev.index)]
            lam = crc_threshold(c["pred"].to_numpy(), ev.reindex(c["target_date"]).to_numpy(), alpha)
            vs = vs[in_season(vs["target_date"]) & vs["target_date"].isin(ev.index)]
            out.append(pd.DataFrame({"target_date": vs["target_date"].to_numpy(), "fire": vs["pred"].to_numpy() >= lam}))
    return pd.concat(out).groupby("target_date")["fire"].mean()


def persistence_fire(rest: pd.Series, ev: pd.Series, lead: int, alpha: float) -> pd.Series:
    sc = scores(rest, "persistence", lead)
    out = []
    for (i0, i1), (v0, v1) in FOLD_YEARS.values():
        def pick(a, b):
            return (rest.index.year >= a) & (rest.index.year <= b) & in_season(rest.index) & sc.notna().to_numpy()
        mc, mv = pick(i0, i1), pick(v0, v1)
        lam = crc_threshold(sc[mc].to_numpy(), ev[mc].to_numpy(), alpha)
        out.append((sc[mv] >= lam).astype(float))
    return pd.concat(out)


def _stats(d: pd.DataFrame) -> np.ndarray:
    e, q = d[d["event"]], d[~d["event"]]
    miss_m, miss_p = 1 - e["m"].mean(), 1 - e["p"].mean()
    far_m, far_p = q["m"].mean(), q["p"].mean()
    return np.array([miss_m - miss_p, far_m - far_p, (1 - miss_m - far_m) - (1 - miss_p - far_p),
                     miss_m, miss_p, far_m, far_p])


def compare(rest: pd.Series, m: pd.Series, p: pd.Series, lead: int) -> dict:
    ev = rest >= R
    d = pd.DataFrame({"m": m, "p": p}).dropna()
    d["event"] = ev.reindex(d.index).to_numpy()
    d = d.rename_axis("target_date").reset_index()
    cl = make_cluster_ids(d["target_date"])
    ids = np.unique(cl)
    G = len(ids)
    est = _stats(d)
    reps = np.array([_stats(d[cl != g]) for g in ids])
    se = np.sqrt((G - 1) / G * ((reps - reps.mean(0)) ** 2).sum(0))
    names = ["miss_diff", "false_alarm_diff", "peirce_diff"]
    res = {"lead": lead, "days": int(len(d)), "event_days": int(d["event"].sum()), "n_clusters": G,
           "model": {"miss": est[3], "false_alarm": est[5]}, "persistence": {"miss": est[4], "false_alarm": est[6]}}
    for k, n in enumerate(names):
        t = est[k] / se[k]
        half = sp_stats.t.ppf(0.975, G - 1) * se[k]
        res[n] = {"estimate": float(est[k]), "se": float(se[k]), "ci95": [float(est[k] - half), float(est[k] + half)],
                  "p_two_sided": float(2 * sp_stats.t.sf(abs(t), G - 1))}
    return res


def main(alpha: float = ALPHA) -> int:
    rest = rest_hours()
    rest = rest[rest.index.year <= 2018]
    ev = rest >= R
    val, inner = load(RUNS["WBGT"])
    out = [compare(rest, model_fire(inner, val, ev, lead, alpha), persistence_fire(rest, ev, lead, alpha), lead)
           for lead in range(1, 6)]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e5_vs_persistence.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in out:
        print(f"lead {r['lead']}: miss model {r['model']['miss']:.3f} vs pers {r['persistence']['miss']:.3f} | "
              f"FA model {r['model']['false_alarm']:.3f} vs pers {r['persistence']['false_alarm']:.3f} | "
              + " | ".join(f"{n} {r[n]['estimate']:+.3f} [{r[n]['ci95'][0]:+.3f},{r[n]['ci95'][1]:+.3f}] p={r[n]['p_two_sided']:.3f}"
                           for n in ("miss_diff", "false_alarm_diff", "peirce_diff")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
