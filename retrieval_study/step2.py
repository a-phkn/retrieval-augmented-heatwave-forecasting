"""
RS-2 runner (pre-registered in context/decisions_retrieval.md): screens for retrieval that brings information C3-hop
does not see (Rz, the 500 hPa pattern; LW, season-to-date analogues), and retrieval for ranges (KR). Development
folds only.

Run from repo root:
    python -m retrieval_study.step2 --family Tmax     -> retrieval_study/results/step2_Tmax.json
    python -m retrieval_study.step2 --family WBGT     -> retrieval_study/results/step2_WBGT.json
    python -m retrieval_study.step2 --report          -> retrieval_study/results/step2.{md,json}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.stats import make_cluster_ids, paired_cluster_test
from retrieval.fold_retrieval import synoptic_features
from retrieval_study import screens as sc
from retrieval_study.step1 import _rmse, _test
from retrieval_study.testbed import FAMILIES, LEADS, FoldData, long_rows, select_fit_predict
from training.folds import FOLDS

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "retrieval_study" / "results"
PRED_DIR = REPO_ROOT / "predictions_v2"
SHORT = {"Tmax": "Tmax", "WBGT": "WBGT (physical)"}
STRATA = ("all", "extreme")
SEEDS = tuple(range(10))
SIGNALS = ("Rz", "LW")


# ------------------------------------------------------------------ features for every training window and the validation windows


def rz_direct(fd: FoldData, q) -> np.ndarray:
    """The 20 principal-component scores of the 3-day 500 hPa anomaly fields Rz matches on."""
    _, mean, comps, za, w = fd.fr._synoptic_state()
    return (synoptic_features(za, w, pd.DatetimeIndex(q)) - mean) @ comps.T


def lw_direct(fd: FoldData, q) -> np.ndarray:
    """Mean standardised anomaly over days -60..-46, -45..-31 and -30..-15 before the first forecast day."""
    z = sc.standardised_anomaly(fd)
    pos = fd.d.index.get_indexer(pd.DatetimeIndex(q))
    blocks = ((-60, -45), (-45, -30), (-30, -14))
    return np.column_stack([np.mean(z[pos[:, None] + np.arange(a, b)], axis=1) for a, b in blocks])


def signals(fd: FoldData) -> dict[str, dict]:
    """{name: {"real": (train, val), "rand": [(train, val) per seed], "direct": (train, val)}}"""
    tq, vq = fd.train.q, fd.val.q
    out = {"Rz": {"real": (sc.project_retrieval(fd, "synoptic", tq), sc.project_retrieval(fd, "synoptic", vq)),
                  "rand": [(sc.project_retrieval(fd, "rand", tq, seed=s), sc.project_retrieval(fd, "rand", vq, seed=s))
                           for s in SEEDS],
                  "direct": (rz_direct(fd, tq), rz_direct(fd, vq))},
           "LW": {"real": (sc.long_window(fd, tq), sc.long_window(fd, vq)),
                  "rand": [(sc.long_window(fd, tq, seed=1000 + s), sc.long_window(fd, vq, seed=2000 + s)) for s in SEEDS],
                  "direct": (_lw_direct_or_nan(fd, tq), _lw_direct_or_nan(fd, vq))}}
    return out


def _lw_direct_or_nan(fd: FoldData, q) -> np.ndarray:
    pos = fd.d.index.get_indexer(pd.DatetimeIndex(q))
    ok = pos >= sc.LONG_DAYS
    out = np.full((len(pos), 3), np.nan)
    out[ok] = lw_direct(fd, pd.DatetimeIndex(q)[ok])
    return out


def _finite_rows(*arrays: np.ndarray) -> np.ndarray:
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a).all(axis=1)
    return ok


# ------------------------------------------------------------------ one fold


def run_fold(fd: FoldData) -> dict:
    sig = signals(fd)
    x_tr, x_va = fd.base("B3", "train"), fd.base("B3", "val")
    every_train = [a for s in sig.values() for a in [s["real"][0], s["direct"][0]] + [r[0] for r in s["rand"]]]
    use = _finite_rows(*every_train)  # the same training rows for every arm
    for s in sig.values():
        for a in [s["real"][1], s["direct"][1]] + [r[1] for r in s["rand"]]:
            if not np.isfinite(a).all():
                raise ValueError(f"{fd.fold}: a validation query lacks a signal")
    info = {"B3": [long_rows(fd, select_fit_predict(x_tr, x_va, fd.train, fd.val, use, fd.fit, fd.stop)[0], "B3")]}
    for name, s in sig.items():
        for arm, pair in (("real", s["real"]), ("direct", s["direct"])):
            pred = select_fit_predict(np.column_stack([x_tr, pair[0]]), np.column_stack([x_va, pair[1]]),
                                      fd.train, fd.val, use, fd.fit, fd.stop)[0]
            info[f"B3+{name}{'' if arm == 'real' else '-direct'}"] = [long_rows(fd, pred, f"B3+{name}-{arm}")]
        info[f"B3+{name}-rand"] = [long_rows(fd, select_fit_predict(np.column_stack([x_tr, r[0]]),
                                                                    np.column_stack([x_va, r[1]]), fd.train, fd.val,
                                                                    use, fd.fit, fd.stop)[0], f"B3+{name}-rand", seed=i)
                                   for i, r in enumerate(s["rand"])]
    # relevance screen: the add-on to C3-hop's ensemble forecasts
    _, _, ctrl = FAMILIES[fd.family]
    nf = sc.NetworkForecasts.load(fd, PRED_DIR, f"{ctrl}_C3hop")
    rows = pd.Series(np.arange(len(fd.train.q)), index=fd.train.q).reindex(nf.inner.index).to_numpy(dtype=np.int64)
    rel = {}
    for name, s in sig.items():
        fr = sc.screen_frames(fd, nf, (s["real"][0][rows], s["real"][1]), [(r[0][rows], r[1]) for r in s["rand"]], name)
        rel[name] = fr
        pred, _ = sc.addon_correction(nf, s["direct"][0][rows], s["direct"][1])
        rel[name]["+direct"] = long_rows(fd, pred, f"{nf.rid}+{name}-direct")
    # KR: retrieval for ranges
    kr = {"global": [sc.knn_ranges(fd, nf, "global")], "knn": [sc.knn_ranges(fd, nf, "knn")],
          "random": [sc.knn_ranges(fd, nf, "random", seed=7000 + s) for s in SEEDS]}
    stratum = fd.d["stratum"].to_numpy()[fd.val.pos[:, None] + LEADS[None, :]]
    target_dates = fd.val.q.to_numpy()[:, None] + LEADS[None, :].astype("timedelta64[D]")
    return {"info": info, "rel": rel, "kr": kr, "kr_stratum": stratum, "kr_dates": target_dates}


# ------------------------------------------------------------------ pooling and tests


def _pool(parts: list[list[pd.DataFrame]]) -> pd.DataFrame:
    return pd.concat([df for p in parts for df in p], ignore_index=True)


def kr_summary(folds: list[dict]) -> dict:
    """Coverage, mean width and interval score per method and stratum; paired cluster jackknife of the mean interval
    score (knn vs global, knn vs random)."""
    stratum = np.concatenate([f["kr_stratum"] for f in folds])
    clusters = make_cluster_ids(pd.DatetimeIndex(np.concatenate([f["kr_dates"] for f in folds]).ravel())).reshape(
        stratum.shape)
    per = {}
    for method in ("global", "knn", "random"):
        reps = []
        for i in range(len(folds[0]["kr"][method])):
            r = {k: np.concatenate([f["kr"][method][i][k] for f in folds]) for k in ("lo", "hi", "y")}
            reps.append({"score": sc.interval_score(r["lo"], r["hi"], r["y"]), "cover": (r["lo"] <= r["y"]) & (r["y"] <= r["hi"]),
                         "width": r["hi"] - r["lo"]})
        per[method] = reps
    out = {"summary": {}, "tests": {}}
    for s in STRATA:
        mask = np.ones_like(stratum, bool) if s == "all" else stratum == "extreme"
        for method, reps in per.items():
            out["summary"][f"{method}|{s}"] = {
                "coverage": float(np.mean([r["cover"][mask].mean() for r in reps])),
                "width": float(np.mean([r["width"][mask].mean() for r in reps])),
                "interval_score": float(np.mean([r["score"][mask].mean() for r in reps])), "rows": int(mask.sum())}
        ids = clusters[mask]
        for other in ("global", "random"):
            res = paired_cluster_test(parent=[r["score"][mask] for r in per[other]], child=per["knn"][0]["score"][mask],
                                      cluster_ids=ids, metric="mae")
            out["tests"][f"knn vs {other}|{s}"] = {k: v for k, v in res.to_dict().items()
                                                   if k in ("parent", "child", "delta", "ci_low", "ci_high", "n_clusters")}
    return out


def run_family(short: str) -> dict:
    family = SHORT[short]
    folds = []
    for fold in FOLDS:
        t0 = time.time()
        folds.append(run_fold(FoldData.build(fold, family)))
        print(f"{short} {fold}: done in {time.time() - t0:.0f} s", flush=True)
    info = {k: _pool([f["info"][k] for f in folds]) for k in folds[0]["info"]}
    res = {"family": family, "info": {"rmse": {k: _rmse(v) for k, v in info.items()}, "tests": {}}, "rel": {}}
    for name in SIGNALS:
        for s in STRATA:
            t = res["info"]["tests"]
            t[f"{name} vs B3|{s}"] = _test(info["B3"], info[f"B3+{name}"], s)
            t[f"{name} vs random|{s}"] = _test(info[f"B3+{name}-rand"], info[f"B3+{name}"], s)
            t[f"{name}-direct vs B3|{s}"] = _test(info["B3"], info[f"B3+{name}-direct"], s)
            t[f"{name} vs {name}-direct|{s}"] = _test(info[f"B3+{name}-direct"], info[f"B3+{name}"], s)
        rel = {k: pd.concat([f["rel"][name][k] for f in folds], ignore_index=True)
               for k in ("alone", "+real", "+rand", "+direct")}
        r = {"rmse": {k: _rmse(v) for k, v in rel.items()}, "tests": {}}
        for s in STRATA:
            r["tests"][f"+{name} vs alone|{s}"] = _test(rel["alone"], rel["+real"], s)
            r["tests"][f"+{name} vs +random|{s}"] = _test(rel["+rand"], rel["+real"], s)
            r["tests"][f"+{name}-direct vs alone|{s}"] = _test(rel["alone"], rel["+direct"], s)
        res["rel"][name] = r
    res["kr"] = kr_summary(folds)
    return res


# ------------------------------------------------------------------ verdicts and report


def verdicts(res: dict) -> dict:
    t, out = res["info"]["tests"], {}
    for name in SIGNALS:
        r = res["rel"][name]["tests"]
        passed = False
        for s in STRATA:
            info_ok = t[f"{name} vs B3|{s}"]["ci_high"] < 0 and t[f"{name} vs random|{s}"]["ci_high"] < 0
            if s == "extreme":
                info_ok = info_ok and t[f"{name} vs B3|all"]["delta"] <= 0.005
            rel_ok = r[f"+{name} vs +random|{s}"]["ci_high"] < 0
            passed = passed or (info_ok and rel_ok)
        out[f"{name} passes"] = bool(passed)
    k = res["kr"]
    out["KR passes"] = bool(k["tests"]["knn vs global|extreme"]["ci_high"] < 0 and k["tests"]["knn vs random|extreme"]["ci_high"] < 0
                            and k["summary"]["knn|all"]["interval_score"] <= 1.02 * k["summary"]["global|all"]["interval_score"])
    return out


def _d(t: dict) -> str:
    if not np.isfinite(t["ci_low"]):
        return f"{t['delta']:+.3f} (no CI)"
    mark = " **(CI < 0)**" if t["ci_high"] < 0 else (" (CI > 0)" if t["ci_low"] > 0 else "")
    return f"{t['delta']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]{mark}"


def write_report(results: dict) -> None:
    names = {"Rz": "Rz (500 hPa pattern analogues)", "LW": "LW (60-day season-to-date analogues)"}
    L = ["# Retrieval study, step 2: screens (RS-2)", "",
         "Pre-registered in `context/decisions_retrieval.md` (RS-2) before any screen was run. Development folds only "
         "(validation 2007-2018 pooled). RMSE in °C; Δ = RMSE of the second minus the first (negative = helps); 95% "
         "CIs from the paired cluster jackknife. Skipped by RS-1's results: RAFT inside C3-hop, a cross-site pool, a "
         "learned similarity (reasons in the decision log).", ""]
    for short, res in results.items():
        L += [f"## {res['family']}", "", "### Information screen (ridge on B3: what C3-hop roughly sees)", "",
              "| Signal | Days | vs B3 alone | vs its random control | direct version vs B3 alone | signal vs direct version |",
              "|---|---|---|---|---|---|"]
        t = res["info"]["tests"]
        for name in SIGNALS:
            for s in STRATA:
                L.append(f"| {names[name]} | {'all days' if s == 'all' else 'heatwave days'} | {_d(t[f'{name} vs B3|{s}'])} | "
                         f"{_d(t[f'{name} vs random|{s}'])} | {_d(t[f'{name}-direct vs B3|{s}'])} | "
                         f"{_d(t[f'{name} vs {name}-direct|{s}'])} |")
        L += ["", "### Relevance screen (add-on to C3-hop's 10-seed forecast)", "",
              "| Signal | Days | C3-hop RMSE | + signal vs alone | + signal vs + random | + direct version vs alone |",
              "|---|---|---|---|---|---|"]
        for name in SIGNALS:
            r = res["rel"][name]
            for s in STRATA:
                L.append(f"| {names[name]} | {'all days' if s == 'all' else 'heatwave days'} | {r['rmse']['alone'][s]:.3f} | "
                         f"{_d(r['tests'][f'+{name} vs alone|{s}'])} | {_d(r['tests'][f'+{name} vs +random|{s}'])} | "
                         f"{_d(r['tests'][f'+{name}-direct vs alone|{s}'])} |")
        k = res["kr"]
        L += ["", "### KR: retrieval for 80% ranges around C3-hop's forecast", "",
              "| Ranges from | Days | Coverage | Mean width | Interval score |", "|---|---|---|---|---|"]
        labels = {"global": "all inner-block errors", "knn": "the 50 most similar situations (retrieval)",
                  "random": "50 random situations (10 seeds)"}
        for s in STRATA:
            for m in ("global", "knn", "random"):
                v = k["summary"][f"{m}|{s}"]
                L.append(f"| {labels[m]} | {'all days' if s == 'all' else 'heatwave days'} | {100 * v['coverage']:.0f}% | "
                         f"{v['width']:.2f} | {v['interval_score']:.3f} |")
        L += ["", "| Comparison (mean interval score) | All days | Heatwave days |", "|---|---|---|"]
        for other in ("global", "random"):
            L.append(f"| knn vs {other} | {_d(k['tests'][f'knn vs {other}|all'])} | {_d(k['tests'][f'knn vs {other}|extreme'])} |")
        v = res["verdicts"]
        L += ["", "### Pre-registered pass rule", "", "| Candidate | Passes |", "|---|---|"]
        L += [f"| {key.replace(' passes', '')} | {'yes' if ok else 'no'} |" for key, ok in v.items()]
        L.append("")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "step2.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Retrieval study, step 2 (RS-2).")
    ap.add_argument("--family", choices=sorted(SHORT))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.family:
        res = run_family(args.family)
        res["verdicts"] = verdicts(res)
        (OUT_DIR / f"step2_{args.family}.json").write_text(json.dumps(res, indent=2, default=float) + "\n",
                                                           encoding="utf-8")
        print(f"wrote {OUT_DIR / f'step2_{args.family}.json'}")
    if args.report:
        results = {s: json.loads((OUT_DIR / f"step2_{s}.json").read_text(encoding="utf-8")) for s in SHORT}
        (OUT_DIR / "step2.json").write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
        write_report(results)
        print(f"wrote {OUT_DIR / 'step2.md'}")


if __name__ == "__main__":
    main()
