"""
Scoring of the one-time test run (docs/CONFIRMATORY_PROTOCOL.md). Reads only predictions_test/ (written by
training.test_run) and training-years label thresholds; refuses unless the test lock is open.

Primary hypotheses (WBGT, Holm correction over the four, family-wise alpha 0.05; "supported" = Holm-adjusted
p < 0.05 and delta < 0), paired cluster-jackknife on year x season clusters as in development:
  H1  C3-hop vs the Delhi-only control (A2Lr_hw5), all days
  H2  C3-hop vs the control, extreme days
  H3  C3-hop vs ridge_hw5, extreme days
  H4  LSTM + Rg (A2Lr_hw5_Rg) vs the control, all days (added 2026-10-19, user)
Exploratory (no confirmatory claim): the same for Tmax; C3-hop vs the plain ridge (all days); by lead; bias / SD
split; hot-day Brier (recalibrated on the early-stopping block 2017-2018); range forecasts (CRPS, coverage);
forecast-conditioned hot days; C2 / U1 / C3-pool vs C3-hop; LSTM + Rg vs LSTM; C3-hop + Rg vs C3-hop; the Rz
side-check (2019-2021, all days); case studies of the 2019, 2022 and 2024 heat events. Truth = ERA5; the last ~3
months of the period were preliminary ERA5T when downloaded (disclosed).

Run from repo root (after the test run):  python -m evaluation.test_evaluation  ->  evaluation_test/test_results.{md,json}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluation.calibration_check import apply_recal, brier, fit_recal, probability_frames
from evaluation.compare_v2 import (
    _ci, _fmt_p, _tmax_clim_by_date, _wbgt_thresholds, compare, ensemble, extreme_bias, forecast_conditioned,
    forecast_conditioned_wbgt, rmse_by,
)
from evaluation.range_forecasts import Z50, Z80, crps_empirical, crps_gaussian
from training import test_lock
from training.folds import FORECAST_DAYS, TEST_FOLD, fold_daily
from training.test_run import PRED_TEST, RZ_FOLD, RZ_RUN

OUT_DIR = PRED_TEST.parent / "evaluation_test"
FAMILIES = {"WBGT (physical)": ("A2Lr_hw5", "wbgt_lj_max", "wbgt"), "Tmax": ("A1prime_hw5", "t_max", "v2")}
ALPHA = 0.05
# Case-study windows, fixed in the protocol before the test period was opened (widely reported heat events)
CASES = {"2019 (late May - mid June)": ("2019-05-25", "2019-06-15"), "2022 (April - mid May)": ("2022-04-01", "2022-05-15"),
         "2024 (mid May - June)": ("2024-05-15", "2024-06-20")}


def load(run_id: str, fold: str = TEST_FOLD, suffix: str = "") -> pd.DataFrame:
    return pd.read_parquet(PRED_TEST / run_id / f"{fold}{suffix}.parquet").assign(fold=fold)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (step-down, monotone, capped at 1)."""
    names = sorted(pvals, key=pvals.get)
    m, out, running = len(names), {}, 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[n]))
        out[n] = running
    return out


def primary() -> dict:
    c3, ctrl, rhw, rg = load("A2Lr_hw5_C3hop"), load("A2Lr_hw5"), load("A2Lr_hw5_ridge_hw5"), load("A2Lr_hw5_Rg")
    tests = {"H1": compare(ctrl, c3, "all"), "H2": compare(ctrl, c3, "extreme"), "H3": compare(rhw, c3, "extreme"),
             "H4": compare(ctrl, rg, "all")}
    adj = holm({k: t["p_value"] for k, t in tests.items()})
    for k, t in tests.items():
        t["p_holm"] = adj[k]
        t["supported"] = bool(adj[k] < ALPHA and t["delta"] < 0)
    return tests


def _split(df: pd.DataFrame) -> dict:
    out = {}
    for s, sub in (("all", df), ("extreme", df[df["stratum"] == "extreme"])):
        g = sub.groupby("seed")["error"]
        out[s] = {"rmse": rmse_by(sub), "bias": float(g.mean().mean()), "sd": float(g.std(ddof=0).mean())}
    return out


def probabilistic(rid: str, family: str) -> dict:
    """Hot-day Brier and range forecasts, recalibrated / spread from the early-stopping block (2017-2018)."""
    val, stop = load(rid).drop(columns="fold"), load(rid, suffix="_stop").drop(columns="fold")
    coef = fit_recal(stop)
    ve, se = ensemble(apply_recal(val, coef)), ensemble(apply_recal(stop, coef))
    out = {"brier": brier(probability_frames(ve, se, TEST_FOLD, family))}
    y, mu = ve["actual"].to_numpy(float), ve["pred"].to_numpy(float)
    crps_g, crps_e, in80, in50 = (np.empty(len(ve)) for _ in range(4))
    for lead in range(1, FORECAST_DAYS + 1):
        i = (ve["lead"] == lead).to_numpy()
        r = se.loc[se["lead"] == lead, "error"].to_numpy(float)
        sd = r.std(ddof=1)
        crps_g[i] = crps_gaussian(y[i], mu[i], np.full(i.sum(), sd))
        crps_e[i] = crps_empirical(y[i], mu[i], r)
        in80[i], in50[i] = np.abs(y[i] - mu[i]) <= Z80 * sd, np.abs(y[i] - mu[i]) <= Z50 * sd
    ext = (ve["stratum"] == "extreme").to_numpy()
    out["range"] = {s: {"crps_gauss": float(crps_g[m].mean()), "crps_emp": float(crps_e[m].mean()),
                        "cover80": float(in80[m].mean()), "cover50": float(in50[m].mean())}
                    for s, m in (("all", np.ones(len(ve), bool)), ("extreme", ext))}
    return out


def cases(c3: pd.DataFrame, ctrl: pd.DataFrame) -> dict:
    out = {}
    for name, (a, b) in CASES.items():
        row = {}
        for label, df in (("C3-hop", c3), ("control", ctrl)):
            e = ensemble(df.drop(columns="fold"))
            sel = (e["target_date"] >= a) & (e["target_date"] <= b)
            for lead in (1, 3, 5):
                s = e[sel & (e["lead"] == lead)]
                row[f"{label} lead {lead}"] = {"rmse": float(np.sqrt(np.mean(s["error"] ** 2))),
                                               "bias": float(s["error"].mean()), "n": int(len(s))}
            row["observed_max"] = float(e.loc[sel, "actual"].max())
        out[name] = row
    return out


def rz_side_check() -> dict | None:
    if not (PRED_TEST / RZ_RUN / f"{RZ_FOLD}.parquet").exists():
        return None
    rz = load(RZ_RUN, RZ_FOLD)
    c3 = load("A2Lr_hw5_C3hop")
    c3 = c3[c3["query_date"].isin(set(rz["query_date"]))]
    return {"all": compare(c3, rz, "all"), "n_windows": int(rz["query_date"].nunique())}


def exploratory() -> dict:
    res = {}
    for fam, (ctrl_id, target, labels) in FAMILIES.items():
        c3, ctrl = load(f"{ctrl_id}_C3hop"), load(ctrl_id)
        r = {"rmse": {}, "tests": {}, "by_lead": {}}
        models = {"C3-hop": c3, "control": ctrl, "ridge": load(f"{ctrl_id}_ridge"), "ridge_hw5": load(f"{ctrl_id}_ridge_hw5")}
        for m in ("C2", "U1_tuned", "C3pool_tuned", "Rg", "C3hop_Rg"):
            if (PRED_TEST / f"{ctrl_id}_{m}" / f"{TEST_FOLD}.parquet").exists():
                models[m] = load(f"{ctrl_id}_{m}")
        for m, df in models.items():
            r["rmse"][m] = {**_split(df), "extreme_bias": extreme_bias(df)}
        for s in ("all", "extreme"):
            r["tests"][f"C3-hop vs control ({s})"] = compare(ctrl, c3, s)
            r["tests"][f"C3-hop vs ridge_hw5 ({s})"] = compare(models["ridge_hw5"], c3, s)
            r["tests"][f"C3-hop vs ridge ({s})"] = compare(models["ridge"], c3, s)
            for m in ("C2", "U1_tuned", "C3pool_tuned"):
                if m in models:
                    r["tests"][f"{m} vs C3-hop ({s})"] = compare(c3, models[m], s)
            if "Rg" in models:  # for WBGT, all days is H4 (primary); reported here too
                r["tests"][f"LSTM + Rg vs control ({s})"] = compare(ctrl, models["Rg"], s)
            if "C3hop_Rg" in models:
                r["tests"][f"C3-hop + Rg vs C3-hop ({s})"] = compare(c3, models["C3hop_Rg"], s)
        for lead in range(1, FORECAST_DAYS + 1):
            r["by_lead"][lead] = compare(ctrl[ctrl["lead"] == lead], c3[c3["lead"] == lead], "all")
        r["probabilistic"] = {"C3-hop": probabilistic(f"{ctrl_id}_C3hop", fam), "control": probabilistic(ctrl_id, fam)}
        fc = (lambda df: forecast_conditioned(df, _tmax_clim_by_date((TEST_FOLD,)))) if fam == "Tmax" else \
            (lambda df: forecast_conditioned_wbgt(df, _wbgt_thresholds((TEST_FOLD,))))
        r["forecast_hot"] = {m: fc(models[m]) for m in ("C3-hop", "control")}
        r["cases"] = cases(c3, ctrl)
        res[fam] = r
    res["rz_side_check"] = rz_side_check()
    return res


def _t(t: dict) -> str:
    return f"{t['delta']:+.3f} {_ci(t)} (p {_fmt_p(t['p_value'])})"


def write_report(prim: dict, expl: dict) -> None:
    L = ["# Test results, 2019-01-01 .. 2026-09-06 (one evaluation run under the frozen protocol)", "",
         "Models trained once on 1980-2016 (early stopping 2017-2018); truth = ERA5 (last ~3 months ERA5T when "
         "downloaded). Δ = first-named minus second (negative = C3-hop better). Paired cluster-jackknife, year x season.", "",
         "## Primary hypotheses (WBGT; Holm over H1-H4)", "", "| | Comparison | Δ RMSE (95% CI) | p | Holm p | Supported |",
         "|---|---|---|---|---|---|"]
    names = {"H1": "C3-hop vs Delhi-only control, all days", "H2": "C3-hop vs control, heatwave days",
             "H3": "C3-hop vs ridge_hw5, heatwave days", "H4": "LSTM + Rg vs Delhi-only control, all days"}
    for k, t in prim.items():
        L.append(f"| {k} | {names[k]} | {t['delta']:+.3f} {_ci(t)} | {_fmt_p(t['p_value'])} | {_fmt_p(t['p_holm'])} | "
                 f"{'yes' if t['supported'] else 'no'} |")
    L += ["", "## Exploratory (no confirmatory claims)", ""]
    for fam in FAMILIES:
        r = expl[fam]
        L += [f"### {fam}", "", "| Model | RMSE all / bias / SD | RMSE extreme / bias / SD |", "|---|---|---|"]
        for m, v in r["rmse"].items():
            L.append(f"| {m} | {v['all']['rmse']:.3f} / {v['all']['bias']:+.2f} / {v['all']['sd']:.2f} | "
                     f"{v['extreme']['rmse']:.3f} / {v['extreme']['bias']:+.2f} / {v['extreme']['sd']:.2f} |")
        L += ["", "| Comparison | Δ (95% CI) |", "|---|---|"] + [f"| {k} | {_t(t)} |" for k, t in r["tests"].items()]
        L += ["", "By lead (C3-hop vs control, all days): " + "; ".join(f"day {k} {_t(t)}" for k, t in r["by_lead"].items())]
        for m, p in r["probabilistic"].items():
            b, g = p["brier"], p["range"]
            L.append(f"- {m}: hot-day Brier {b['brier']:.4f} (skill {b['bss']:+.3f}, {b['n_events']} hot days); CRPS all "
                     f"{g['all']['crps_gauss']:.3f}, extreme {g['extreme']['crps_gauss']:.3f}; 80% coverage all "
                     f"{g['all']['cover80']:.2f}, extreme {g['extreme']['cover80']:.2f}")
        for m, h in r["forecast_hot"].items():
            L.append(f"- {m}: forecast hot days per seed {h['n_per_seed']:.0f}, RMSE {h['rmse']:.2f}, bias {h['bias']:+.2f}")
        L += ["", "Case studies (10-seed ensemble; RMSE / bias °C at leads 1, 3, 5):", ""]
        for name, c in r["cases"].items():
            L.append(f"- {name}: observed max {c['observed_max']:.1f} °C; " + "; ".join(
                f"{k} {v['rmse']:.2f} / {v['bias']:+.2f}" for k, v in c.items() if k != "observed_max"))
        L.append("")
    rz = expl["rz_side_check"]
    L.append("Rz side-check (WBGT, 2019-2021, all days): " +
             (f"C3-hop + Rz vs C3-hop {_t(rz['all'])} over {rz['n_windows']} windows" if rz else "not run"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "test_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    test_lock.require()
    prim, expl = primary(), exploratory()
    write_report(prim, expl)
    (OUT_DIR / "test_results.json").write_text(json.dumps({"primary": prim, "exploratory": expl}, indent=2, default=float)
                                               + "\n", encoding="utf-8")
    print({k: (round(t["delta"], 3), t["supported"]) for k, t in prim.items()})


if __name__ == "__main__":
    main()
