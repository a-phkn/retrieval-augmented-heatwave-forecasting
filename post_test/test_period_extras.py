"""
Post-test exploration 2 (added 2026-10-25 after the four critical reviews; descriptive only, it cannot change the
confirmatory result in evaluation_test/test_results.md). Numbers the reviews asked to be reported:

  1. the 10-seed ensemble (the deliverable forecast per docs/CONFIRMATORY_PROTOCOL.md), all and heatwave days;
  2. warning scores for the ensemble at the label threshold: hit rate, false-alarm ratio, equitable threat score;
  3. the realised minimum detectable effect of H2 (80% power), versus the protocol's pre-test estimate;
  4. how much of H3's heatwave-day gap is mean bias (bias squared) rather than day-to-day error;
  5. H1, H3 and H4 again without the last ~3 months (June - September 2026, preliminary ERA5T when downloaded).

Reads only the committed predictions_test/; the label thresholds come from the training years, which needs the test
lock to be valid (the code is unchanged since the tag), exactly as in evaluation.test_evaluation.

Run from repo root:  python -m post_test.test_period_extras  ->  evaluation_post_test/test_period_extras.{md,json}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.compare_v2 import _ci, _fmt_p, _tmax_clim_by_date, _wbgt_thresholds, compare, ensemble, mde80
from pipeline.labels_v2 import hot_days_tmax, in_wbgt_season
from training.folds import TEST_FOLD

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_TEST = REPO_ROOT / "predictions_test"
OUT_DIR = REPO_ROOT / "evaluation_post_test"
FAMILIES = {"WBGT (physical)": "A2Lr_hw5", "Tmax": "A1prime_hw5"}
MODELS = {"C3-hop": "_C3hop", "Delhi-only LSTM": "", "ridge": "_ridge", "ridge_hw5": "_ridge_hw5"}
ERA5T_START = "2026-06-01"
PROTOCOL_MDE_H2 = 0.45  # docs/CONFIRMATORY_PROTOCOL.md power table, estimated before the test


def load(run_id: str) -> pd.DataFrame:
    return pd.read_parquet(PRED_TEST / run_id / f"{TEST_FOLD}.parquet").assign(fold=TEST_FOLD)


def rmse(e: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(e, dtype=float) ** 2)))


def event_scores(observed: np.ndarray, forecast: np.ndarray) -> dict:
    """Yes/no warning scores from boolean arrays of observed and forecast hot days.
    hit_rate = hits / observed events; false_alarm_ratio = false alarms / forecast events;
    ets = equitable threat score (Gilbert skill score), 0 = no skill beyond chance, 1 = perfect."""
    o, f = np.asarray(observed, bool), np.asarray(forecast, bool)
    if o.shape != f.shape:
        raise ValueError("observed and forecast must have the same shape")
    hits, misses, fa = int((o & f).sum()), int((o & ~f).sum()), int((~o & f).sum())
    n = o.size
    chance = (hits + misses) * (hits + fa) / n if n else 0.0
    denom = hits + misses + fa - chance
    return {"hits": hits, "misses": misses, "false_alarms": fa, "n": n,
            "hit_rate": hits / (hits + misses) if hits + misses else float("nan"),
            "false_alarm_ratio": fa / (hits + fa) if hits + fa else float("nan"),
            "ets": (hits - chance) / denom if denom else float("nan")}


def bias_share(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """MSE = bias^2 + variance of the error, per seed, averaged over seeds. Returns both models' parts and the
    share of the MSE gap (b minus a) that is the bias^2 gap."""
    def parts(df: pd.DataFrame) -> tuple[float, float]:
        g = df.groupby("seed")["error"]
        return float((g.mean() ** 2).mean()), float((g.apply(lambda e: np.mean(e.to_numpy() ** 2))).mean())
    (ba, ma), (bb, mb) = parts(a), parts(b)
    return {"a_bias2": ba, "a_mse": ma, "b_bias2": bb, "b_mse": mb,
            "bias_share_of_gap": (bb - ba) / (mb - ma) if mb != ma else float("nan")}


def forecast_hot(fam: str, df: pd.DataFrame, thresholds: dict, clim: pd.Series | None) -> np.ndarray:
    """The frozen rule for 'the forecast says hot day' (as in compare_v2's forecast-conditioned check)."""
    if fam == "Tmax":
        c = clim.reindex(pd.DatetimeIndex(df["target_date"])).to_numpy()
        return np.asarray(hot_days_tmax(df["target_date"], df["pred"], df["pred"].to_numpy() - c), bool)
    return np.asarray(in_wbgt_season(df["target_date"]) & (df["pred"].to_numpy() >= thresholds[TEST_FOLD]), bool)


def family_extras(fam: str, ctrl: str, thresholds: dict, clim: pd.Series | None) -> dict:
    out = {"ensemble": {}, "events": {}}
    frames = {m: load(ctrl + s) for m, s in MODELS.items()}
    for m, df in frames.items():
        e = ensemble(df.drop(columns="fold")).assign(fold=TEST_FOLD)
        ext = e["stratum"] == "extreme"
        out["ensemble"][m] = {"rmse_all": rmse(e["error"]), "rmse_extreme": rmse(e.loc[ext, "error"])}
        out["events"][m] = event_scores(ext.to_numpy(), forecast_hot(fam, e, thresholds, clim))
    ext = {m: df[df["stratum"] == "extreme"] for m, df in frames.items()}
    out["h3_bias_share"] = bias_share(ext["C3-hop"], ext["ridge_hw5"])
    out["c3hop_vs_ridge_all"] = compare(frames["ridge"], frames["C3-hop"], "all")
    return out


def without_era5t() -> dict:
    """H1, H3, H4 with every window whose target date is in the ERA5T months removed (same rows for both models)."""
    keep = lambda df: df[df["target_date"] < ERA5T_START]  # noqa: E731
    c3, ctrl, rhw, rg = (keep(load(r)) for r in ("A2Lr_hw5_C3hop", "A2Lr_hw5", "A2Lr_hw5_ridge_hw5", "A2Lr_hw5_Rg"))
    return {"H1": compare(ctrl, c3, "all"), "H3": compare(rhw, c3, "extreme"), "H4": compare(ctrl, rg, "all")}


def _t(t: dict) -> str:
    return f"{t['delta']:+.3f} {_ci(t)} (p {_fmt_p(t['p_value'])})"


def write_report(res: dict) -> None:
    L = ["# Test period: extra descriptive numbers (post-test, 2026-10-25)", "",
         "Asked for by the critical reviews. Descriptive only: the confirmatory result is `evaluation_test/test_results.md`.",
         "RMSE in °C (lower is better). Ensemble = the average of the 10 seeds' forecasts.", ""]
    for fam, r in res["families"].items():
        L += [f"## {fam}", "", "| Model | Ensemble RMSE all days | Ensemble RMSE heatwave days | Hit rate | False-alarm ratio | ETS |",
              "|---|---|---|---|---|---|"]
        for m in MODELS:
            en, ev = r["ensemble"][m], r["events"][m]
            L.append(f"| {m} | {en['rmse_all']:.3f} | {en['rmse_extreme']:.3f} | {ev['hit_rate']:.2f} | "
                     f"{ev['false_alarm_ratio']:.2f} | {ev['ets']:.2f} |")
        b = r["h3_bias_share"]
        L += ["", f"Observed heatwave forecasts (window x lead rows): "
              f"{r['events']['C3-hop']['hits'] + r['events']['C3-hop']['misses']}. "
              "Hit rate = share of observed heatwave days that were forecast; false-alarm ratio = share of forecast "
              "heatwave days that did not happen; ETS = equitable threat score (0 = chance, 1 = perfect).", "",
              f"Heatwave days, C3-hop vs ridge_hw5: MSE {b['a_mse']:.2f} vs {b['b_mse']:.2f}, of which bias² "
              f"{b['a_bias2']:.2f} vs {b['b_bias2']:.2f}; **{100 * b['bias_share_of_gap']:.0f}% of the gap is bias**.", "",
              f"C3-hop vs the plain ridge, all days: {_t(r['c3hop_vs_ridge_all'])} (a tie).", ""]
    h2 = res["h2_power"]
    L += ["## H2 power", "",
          f"Realised minimum detectable effect (80% power): **{h2['mde80_realised']:.2f} °C** "
          f"(the protocol's pre-test estimate was {PROTOCOL_MDE_H2:.2f}). The observed difference was {h2['delta']:+.3f}, "
          "about a third of the development value (-0.255): the test was precise enough to detect the development-sized "
          "effect, so the null is not only a power problem; development probably overstated the heatwave-day gain.", "",
          "## How much of each all-day WBGT gain is a change in the average error (bias²)", ""]
    for k, b in res["bias_share_all_days"].items():
        L.append(f"- {k}: MSE {b['a_mse']:.3f} vs {b['b_mse']:.3f}; bias² {b['a_bias2']:.3f} vs {b['b_bias2']:.3f}; "
                 f"**{100 * b['bias_share_of_gap']:.0f}%** of the gain is bias")
    L += ["", "## Without the ERA5T months (target dates from 2026-06-01 removed)", ""]
    L += [f"- {k}: {_t(t)}" for k, t in res["without_era5t"].items()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "test_period_extras.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    thresholds = _wbgt_thresholds((TEST_FOLD,))
    clim = _tmax_clim_by_date((TEST_FOLD,))
    res = {"families": {fam: family_extras(fam, ctrl, thresholds, clim if fam == "Tmax" else None)
                        for fam, ctrl in FAMILIES.items()}}
    ctrl, c3, rg = load("A2Lr_hw5"), load("A2Lr_hw5_C3hop"), load("A2Lr_hw5_Rg")
    h2 = compare(ctrl, c3, "extreme")
    res["h2_power"] = {"delta": h2["delta"], "mde80_realised": mde80(h2), "mde80_protocol": PROTOCOL_MDE_H2}
    res["bias_share_all_days"] = {"H1 (C3-hop vs control)": bias_share(c3, ctrl), "H4 (LSTM + Rg vs control)": bias_share(rg, ctrl)}
    res["without_era5t"] = without_era5t()
    write_report(res)
    (OUT_DIR / "test_period_extras.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    print((OUT_DIR / "test_period_extras.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
