"""
P3-1 (pre-registered 2026-11-05 in context/decisions_phase3.md): our forecasts vs the archived operational forecasts
of ECMWF IFS, ECMWF AIFS and NCEP GFS for Delhi, Tmax and WBGT, days 1-5, on the test years. Post-test and
descriptive: it changes no model and no confirmatory result.

- Query date q, lead L: the target day is D = q + L - 1, built from the hourly `_previous_dayL` values of D. Each comes
  from a run started at least L x 24 h before its valid time, at the latest 12 UTC on q - 1, so before our models'
  input cutoff (the end of q - 1 IST).
- Tmax = daily maximum of the 24 hourly 2 m temperatures of the IST day per cell, then the 9-cell mean (the ERA5
  target's definition). WBGT = hourly Liljegren WBGT with the steps of pipeline/build_wbgt_liljegren.py, the same
  daily maximum and 9-cell mean; not for the experimental AIFS, which has no radiation in the archive.
- A (q, L) row whose target day lacks any hour or cell for a model is dropped from every comparison with that model.
- Compared with C3-hop and the Delhi-only LSTM, 10-seed ensemble means (C3-hop's mean of per-seed RMSE as well): RMSE
  and bias on all days, the extreme stratum and the heat season, by lead; the paired cluster jackknife of
  RMSE(global) - RMSE(C3-hop); R-point warnings for every model, with C3-hop's R-prob and R-shift (post-test 3) for
  reference.
- Timing check, once downloaded: IFS HRES 9 km from its 12 UTC run on q - 1 (Single Runs API).
- Deviation of 2026-11-10: the operational AIFS Single (`ecmwf_aifs025_single`, from 2025-02, Tmax and WBGT) is a
  fourth model. It was added after the first results were seen, and every table says so.

Run from repo root, after python -m followup.open_meteo:  python -m followup.global_models
  -> followup/results/global_models.{md,json} and followup/data/open_meteo_runs_daily.parquet
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from followup import open_meteo as om
from training.folds import FORECAST_DAYS

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "followup" / "results"
EXTRACT = REPO_ROOT / "followup" / "data" / "open_meteo_runs_daily.parquet"
PRED_TEST = REPO_ROOT / "predictions_test"
WARNING_JSON = REPO_ROOT / "evaluation_post_test" / "warning_rules.json"
ERA5_RAW = REPO_ROOT / "data" / "raw" / "era5_monthly"
KEY = ["query_date", "lead"]
SHORT = {"temperature_2m": "t", "relative_humidity_2m": "rh", "surface_pressure": "pressure",
         "wind_speed_10m": "wind", "shortwave_radiation": "ghi", "diffuse_radiation": "diffuse"}
FAMILIES = {"Tmax": ("A1prime_hw5", "tmax"), "WBGT (physical)": ("A2Lr_hw5", "wbgt")}
# name: (Open-Meteo model id, variables to load, targets, native output step from Open-Meteo's docs, read 2026-11-06)
MODELS = {
    "IFS": ("ecmwf_ifs025", om.WBGT_VARS, ("tmax", "wbgt"), "3-hourly (6-hourly after 144 h)"),
    "AIFS": ("ecmwf_aifs025", ("temperature_2m",), ("tmax",), "6-hourly"),
    "GFS": ("gfs_seamless", om.WBGT_VARS, ("tmax", "wbgt"), "hourly (3-hourly after 120 h)"),
    "AIFS Single": ("ecmwf_aifs025_single", om.WBGT_VARS, ("tmax", "wbgt"), "6-hourly"),
}
ADDED_LATER = {"AIFS Single"}  # the deviation of 2026-11-10: chosen after the first results were seen
GFS_SINCE_2021 = "GFS, Tmax since 2021"
SINGLE_NAME = "IFS HRES 9 km, 12 UTC run"
SINGLE_STEP = "hourly (3-hourly after 90 h, 6-hourly after 144 h)"


# ------------------------------------------------------------------ daily values from hourly forecasts


def _cell_mean(per_cell: pd.DataFrame) -> pd.Series:
    """9-cell mean of the days present in every cell (date index, one column per cell)."""
    return per_cell.reindex(columns=range(1, 10)).dropna().mean(axis=1).rename_axis("date")


def daily_tmax(hourly: pd.DataFrame) -> pd.Series:
    """hourly: cell, time (IST), t. The daily maximum per cell over complete IST days (24 values), then the 9-cell
    mean of the days complete in every cell."""
    h = hourly.dropna(subset=["t"]).drop_duplicates(["cell", "time"])
    g = h.groupby(["cell", h["time"].dt.normalize()])["t"]
    per_cell = g.max()[g.size() == 24].unstack("cell")
    return _cell_mean(per_cell)


def cell_wbgt_hourly(h: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """One cell's hourly Liljegren WBGT, by the steps of pipeline/build_wbgt_liljegren.cell_hourly_wbgt: the mean
    sunlit cosine zenith of the hour ending at each timestamp, direct = global - diffuse, urban wind profile."""
    from pipeline.build_wbgt_liljegren import IST_OFFSET
    from pipeline.wbgt_liljegren import mean_sunlit_cosz, wbgt

    df = h.dropna(subset=["t", "rh", "pressure", "wind", "ghi", "diffuse"]).drop_duplicates("time")
    df = df.sort_values("time").reset_index(drop=True)
    if df.empty:
        return df.assign(wbgt=[], tg=[], tnwb=[])
    utc = (df["time"] - IST_OFFSET).to_numpy(dtype="datetime64[ns]")
    cosz = np.concatenate([mean_sunlit_cosz(chunk, lat, lon) for chunk in np.array_split(utc, max(1, len(utc) // 50_000))])
    direct = np.clip(df["ghi"].to_numpy() - df["diffuse"].to_numpy(), 0.0, None)
    out = wbgt(df["t"], df["rh"], df["pressure"], df["wind"], df["ghi"], direct, cosz, urban=True)
    return df.assign(wbgt=out["wbgt"], tg=out["tg"], tnwb=out["tnwb"])


def daily_wbgt(hourly: pd.DataFrame, where: dict[int, dict]) -> tuple[pd.Series, int]:
    """hourly: cell, time (IST) and the six inputs. Daily max WBGT per cell over complete days
    (pipeline/build_wbgt_liljegren.daily_from_hourly), then the 9-cell mean. Also returns the number of hours the
    WBGT solver could not solve (dropped, so their days are incomplete)."""
    from pipeline.build_wbgt_liljegren import daily_from_hourly

    per_cell, failed = {}, 0
    for cell, h in hourly.groupby("cell"):
        w = cell_wbgt_hourly(h, where[cell]["latitude"], where[cell]["longitude"])
        if w.empty:
            per_cell[cell] = pd.Series(dtype=float)
            continue
        bad = w[["wbgt", "tg", "tnwb"]].isna().any(axis=1)
        failed += int(bad.sum())
        per_cell[cell] = daily_from_hourly(w[~bad][["time", "wbgt", "tg", "tnwb"]])["wbgt_lj_max"]
    return _cell_mean(pd.DataFrame(per_cell)), failed


def model_daily(model: str, variables: tuple[str, ...], targets: tuple[str, ...], root: Path = om.RAW_DIR
                ) -> tuple[dict[str, dict[int, pd.Series]], dict[int, dict], int]:
    """{target: {lead: daily series}} for one Previous Runs model, its grid points and the WBGT solver failures."""
    hourly, where = om.load_previous(model, variables, root)
    out, failed = {t: {} for t in targets}, 0
    for L, h in hourly.items():
        h = h.rename(columns=SHORT)
        if "tmax" in targets:
            out["tmax"][L] = daily_tmax(h)
        if "wbgt" in targets:
            out["wbgt"][L], f = daily_wbgt(h, where)
            failed += f
    return out, where, failed


def single_daily(root: Path = om.RAW_DIR) -> tuple[dict[str, dict[int, pd.Series]], dict[int, dict], int, list[str]]:
    """The same for the IFS HRES runs started 12 UTC on r = q - 1: target day r + L is lead L."""
    df, where, unavailable = om.load_single(root)
    df = df.rename(columns=SHORT)
    parts = {"tmax": {L: [] for L in range(1, FORECAST_DAYS + 1)}, "wbgt": {L: [] for L in range(1, FORECAST_DAYS + 1)}}
    failed = 0
    for run, h in df.groupby("run"):
        targets = {(run + pd.Timedelta(days=L)): L for L in range(1, FORECAST_DAYS + 1)}
        tmax = daily_tmax(h)
        wbgt, f = daily_wbgt(h, where)
        failed += f
        for name, s in (("tmax", tmax), ("wbgt", wbgt)):
            for day, value in s.items():
                if day in targets:
                    parts[name][targets[day]].append((day, value))
    out = {name: {L: pd.Series(dict(v), dtype=float).rename_axis("date") for L, v in by_lead.items()}
           for name, by_lead in parts.items()}
    return out, where, failed, unavailable


# ------------------------------------------------------------------ pairing with our forecasts


def align(ours: pd.DataFrame, by_lead: dict[int, pd.Series], name: str) -> pd.DataFrame:
    """The global model's forecast on our ensemble rows: lead L takes its lead-L value of the target day. Rows
    without a value (archive gaps, incomplete days, before the archive starts) are dropped."""
    parts = []
    for L, s in by_lead.items():
        r = ours[ours["lead"] == L]
        parts.append(r.assign(pred=s.reindex(pd.DatetimeIndex(r["target_date"])).to_numpy()))
    out = pd.concat(parts).dropna(subset=["pred"])
    out = out.assign(model=name, seed=0, error=out["pred"] - out["actual"])
    return out.sort_values(KEY).reset_index(drop=True)


def restrict(df: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """df's rows (all seeds) at the given (query date, lead) pairs, sorted like align's output."""
    return df.merge(rows[KEY], on=KEY, how="inner").sort_values(KEY + ["seed"]).reset_index(drop=True)


# ------------------------------------------------------------------ scores


def summarise(df: pd.DataFrame, season) -> dict:
    """RMSE (mean over seeds of per-seed RMSE; the plain RMSE for one forecast) and bias per stratum, RMSE by lead."""
    from evaluation.compare_v2 import rmse_by

    parts = {"all": df, "extreme": df[df["stratum"] == "extreme"], "season": df[season(df["target_date"])]}
    out = {}
    for k, v in parts.items():
        n = int(len(v) / max(1, v["seed"].nunique()))
        out[k] = {"rmse": rmse_by(v) if n else float("nan"),
                  "bias": float(v.groupby("seed")["error"].mean().mean()) if n else float("nan"), "n": n,
                  "clusters": int(v["cluster_id"].nunique())}
    out["by_lead"] = [rmse_by(df[df["lead"] == L]) if (df["lead"] == L).any() else float("nan")
                      for L in range(1, FORECAST_DAYS + 1)]
    return out


def paired_tests(c3: pd.DataFrame, g: pd.DataFrame, season) -> dict:
    """Paired cluster jackknife of RMSE(global) - RMSE(C3-hop) on the same rows; None-like entries when a stratum has
    too few clusters for the test to run (fewer than 5 give no CI, by design of evaluation/stats.py)."""
    from evaluation.compare_v2 import compare

    sc, sg = season(c3["target_date"]), season(g["target_date"])
    cases = {"all": (c3, g, "all"), "extreme": (c3, g, "extreme"),
             "season": (c3[sc].reset_index(drop=True), g[sg].reset_index(drop=True), "all")}
    out = {}
    for name, (a, b, stratum) in cases.items():
        try:
            out[name] = compare(a, b, stratum)
        except (ValueError, ZeroDivisionError) as err:
            out[name] = {"error": str(err)}
    return out


def warning_scores(points: pd.DataFrame, lstm: pd.DataFrame, probs: pd.DataFrame, g: pd.DataFrame,
                   p_star: float, delta: float) -> dict:
    """Yes/no warnings on the in-season rows where the global model has a value (post-test 3 definitions)."""
    from post_test.warning_rules import scores

    m = (points[KEY + ["thr", "o", "pred"]]
         .merge(g[KEY + ["pred"]].rename(columns={"pred": "global"}), on=KEY)
         .merge(lstm[KEY + ["pred"]].rename(columns={"pred": "lstm"}), on=KEY)
         .merge(probs[KEY + ["p"]], on=KEY))
    o, thr = m["o"].to_numpy(bool), m["thr"].to_numpy(float)
    return {"rows": int(len(m)), "events": int(o.sum()), "rules": {
        "global model, R-point": scores(o, m["global"].to_numpy() >= thr),
        "C3-hop, R-point": scores(o, m["pred"].to_numpy() >= thr),
        f"C3-hop, R-prob (p* = {p_star})": scores(o, m["p"].to_numpy() >= p_star),
        f"C3-hop, R-shift (delta = {delta} C)": scores(o, m["pred"].to_numpy() >= thr - delta),
        "LSTM, R-point": scores(o, m["lstm"].to_numpy() >= thr)}}


def family_results(family: str, nwp: dict[str, dict[int, pd.Series]]) -> dict:
    from evaluation.compare_v2 import ensemble
    from pipeline.labels_v2 import in_season, in_wbgt_season
    from post_test.warning_rules import point_rows, prob_rows

    ctrl, _ = FAMILIES[family]
    c3_rid = ctrl + "_C3hop"
    season = in_season if family == "Tmax" else in_wbgt_season
    c3_seeds = pd.read_parquet(PRED_TEST / c3_rid / "test.parquet")
    c3e, lstm_e = ensemble(c3_seeds), ensemble(pd.read_parquet(PRED_TEST / ctrl / "test.parquet"))
    points, lstm_points = point_rows(c3_rid, family, "test"), point_rows(ctrl, family, "test")
    probs = prob_rows(c3_rid, family, "test")
    choices = json.loads(WARNING_JSON.read_text(encoding="utf-8"))[family]["choices"]
    p_star, delta = choices["C3-hop R-prob"]["p_star"], choices["C3-hop R-shift"]["delta_C"]
    out = {}
    for name, by_lead in nwp.items():
        g = align(c3e, by_lead, name)
        if g.empty:
            out[name] = {"rows": 0}
            continue
        c3p, lstm_p, c3s = restrict(c3e, g), restrict(lstm_e, g), restrict(c3_seeds, g)
        out[name] = {
            "rows": int(len(g)), "clusters": int(g["cluster_id"].nunique()),
            "first_query": str(g["query_date"].min().date()), "last_query": str(g["query_date"].max().date()),
            "global": summarise(g, season), "c3hop": summarise(c3p, season),
            "c3hop_per_seed": summarise(c3s, season), "lstm": summarise(lstm_p, season),
            "test": paired_tests(c3p, g, season),
            "warnings": warning_scores(points, lstm_points, probs, g, p_star, delta),
        }
    return out


# ------------------------------------------------------------------ checks, extract, report


def grid_check(where: dict[int, dict]) -> dict:
    """Each cell's grid point and elevation against the ERA5 download's (first raw file of the cell)."""
    out = {}
    for cell, spot in where.items():
        files = sorted((ERA5_RAW / f"cell_{cell}").glob("*.json"))
        if not files:
            continue
        e = json.loads(files[0].read_text(encoding="utf-8"))
        out[cell] = {"grid": [spot["latitude"], spot["longitude"]], "era5_grid": [e["latitude"], e["longitude"]],
                     "elevation": spot["elevation"], "era5_elevation": e["elevation"]}
    return out


def extract_frame(daily: dict[str, dict[str, dict[int, pd.Series]]]) -> pd.DataFrame:
    rows = [pd.DataFrame({"model": model, "target": target, "lead": L, "date": s.index, "value": s.to_numpy()})
            for model, targets in daily.items() for target, by_lead in targets.items() for L, s in by_lead.items()]
    return pd.concat(rows, ignore_index=True).sort_values(["model", "target", "lead", "date"]).reset_index(drop=True)


def _num(x, fmt: str = ".3f") -> str:
    return "n/a" if x is None or not np.isfinite(x) else format(x, fmt)


def _label(name: str) -> str:
    return f"{name} (added after the first results)" if name in ADDED_LATER else name


def _delta(t: dict) -> str:
    if "error" in t:
        return "not testable"
    ci = "no CI (< 5 clusters)" if not np.isfinite(t["ci_low"]) else f"[{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]"
    flag = ", fragile (< 10 clusters)" if t.get("few_clusters") else ""
    return f"{t['delta']:+.3f} {ci}{flag}"


def write_report(res: dict) -> None:
    L = ["# Our forecasts vs operational global forecasts (P3-1, post-test)", "",
         "Pre-registered 2026-11-05 in `context/decisions_phase3.md` (P3-1). Descriptive: the test years were already "
         "used, so nothing here is confirmatory. RMSE and bias in °C; the global models and C3-hop / LSTM ensemble "
         "means are single forecasts, 'per-seed' is the test's primary score (mean of the 10 seeds' RMSE). Delta = "
         "RMSE(global) - RMSE(C3-hop ensemble) on the same rows, with the paired cluster-jackknife 95% CI "
         "(negative = the global model is better).", ""]
    for family, models in res["families"].items():
        L += [f"## {family}", ""]
        for stratum, title in (("all", "All days"), ("extreme", "Heatwave days (extreme stratum)"),
                               ("season", "Heat season")):
            L += [f"### {title}", "", "| Model | Period (query dates) | Rows | Clusters | RMSE global | C3-hop | "
                  "C3-hop per-seed | LSTM | Delta [95% CI] | Bias global | Bias C3-hop |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for name, r in models.items():
                if not r.get("rows"):
                    L.append(f"| {_label(name)} | no data | 0 | | | | | | | | |")
                    continue
                g, c, s, m = (r[k][stratum] for k in ("global", "c3hop", "c3hop_per_seed", "lstm"))
                L.append(f"| {_label(name)} | {r['first_query']} to {r['last_query']} | {g['n']} | {g['clusters']} | "
                         f"{_num(g['rmse'])} | {_num(c['rmse'])} | {_num(s['rmse'])} | {_num(m['rmse'])} | "
                         f"{_delta(r['test'][stratum])} | {_num(g['bias'], '+.2f')} | {_num(c['bias'], '+.2f')} |")
            L.append("")
        L += ["### By lead (all days, RMSE)", "", "| Model | | Day 1 | 2 | 3 | 4 | 5 |", "|---|---|---|---|---|---|---|"]
        for name, r in models.items():
            if r.get("rows"):
                for who, k in (("global", "global"), ("C3-hop", "c3hop")):
                    L.append(f"| {_label(name)} | {who} | " + " | ".join(_num(x) for x in r[k]["by_lead"]) + " |")
        L += ["", "### Yes/no heatwave warnings (in-season rows where the global model has a value)", "",
              "| Model rows | Rule | Hit rate | False-alarm ratio | ETS | Frequency bias |", "|---|---|---|---|---|---|"]
        for name, r in models.items():
            if r.get("rows"):
                w = r["warnings"]
                for rule, s in w["rules"].items():
                    L.append(f"| {_label(name)} ({w['rows']} rows, {w['events']} events) | {rule} | {_num(s['hit_rate'], '.0%')} | "
                             f"{_num(s['false_alarm_ratio'], '.0%')} | {_num(s['ets'], '.2f')} | {_num(s['frequency_bias'], '.2f')} |")
        L.append("")
    L += ["## Notes", "",
          "- Timing: lead L uses `_previous_dayL`, the value predicted L x 24 h before its valid time, so every value "
          "comes from a run started no later than 12 UTC on the day before the query date (our models' inputs end at "
          "the end of that day, IST).",
          "- Native output steps (Open-Meteo's documentation; hourly values are its interpolation, which can miss the "
          "afternoon peak, most for 6-hourly AIFS): " + "; ".join(f"{k} {v}" for k, v in res["native_steps"].items()) + ".",
          "- AIFS Single (`ecmwf_aifs025_single`, the operational AIFS) was added by the dated deviation of 2026-11-10, "
          "after the results of the other models had been seen. It is a separate model and is not joined to the "
          "experimental AIFS series (`ecmwf_aifs025`), which ends 2025-03-10.",
          "- WBGT inputs: IFS 0.25° gives only global radiation, so Open-Meteo estimates the diffuse part with a "
          "separation model; WBGT's direct / diffuse split is therefore partly modelled for IFS.",
          "- ERA5 is produced with the IFS model, so ERA5 truth may favour IFS and our ERA5-trained models over GFS "
          "and AIFS (P3-3 repeats Tmax against the Safdarjung station).",
          f"- WBGT solver failures (hours dropped): {res['wbgt_solver_failures']}.",
          f"- Grid points vs the ERA5 download: {res['grid_summary']}.",
          f"- Daily extract: `{EXTRACT.relative_to(REPO_ROOT).as_posix()}`, SHA-256 {res['extract_sha256']}.",
          f"- Timing check (IFS HRES 9 km, 12 UTC run): {res['single_status']}."]
    (OUT_DIR / "global_models.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    from training import test_lock

    test_lock.require()
    daily, wheres, failures = {}, {}, {}
    for name, (model, variables, targets, _) in MODELS.items():
        daily[name], wheres[name], failures[name] = model_daily(model, variables, targets)
    gfs_start = pd.Timestamp(om.PULLS["GFS"][2])
    daily[GFS_SINCE_2021] = {"tmax": daily["GFS"]["tmax"]}
    daily["GFS"] = {t: {L: s[s.index >= gfs_start] for L, s in by_lead.items()} for t, by_lead in daily["GFS"].items()}
    native = {k: v[3] for k, v in MODELS.items()}
    runs = om.single_run_dates()
    have = sum(om.single_path(r).exists() for r in runs)
    if have == len(runs):
        daily[SINGLE_NAME], wheres[SINGLE_NAME], failures[SINGLE_NAME], unavailable = single_daily()
        native[SINGLE_NAME] = SINGLE_STEP
        single_status = f"included ({len(unavailable)} of {len(runs)} runs not in the archive)"
    else:
        single_status = (f"not complete yet ({have} of {len(runs)} runs downloaded); "
                         "python -m followup.open_meteo --single-runs resumes it")
    EXTRACT.parent.mkdir(parents=True, exist_ok=True)
    extract_frame(daily).to_parquet(EXTRACT, index=False)
    sha = hashlib.sha256(EXTRACT.read_bytes()).hexdigest()
    families = {}
    for family, (_, target) in FAMILIES.items():
        families[family] = family_results(family, {name: d[target] for name, d in daily.items() if target in d})
    grid = {name: grid_check(w) for name, w in wheres.items()}
    summary = []
    for name, g in grid.items():
        same_grid = all(c["grid"] == c["era5_grid"] for c in g.values())
        same_height = all(c["elevation"] == c["era5_elevation"] for c in g.values())
        summary.append(f"{name}: grid points {'the same as' if same_grid else 'differ from'} ERA5's"
                       f"{'' if same_grid or name != SINGLE_NAME else ' (expected: its own 9 km grid)'}, "
                       f"elevations {'the same' if same_height else 'differ (see JSON)'}")
    res = {"families": families, "native_steps": native, "wbgt_solver_failures": failures, "grid": grid,
           "grid_summary": "; ".join(summary),
           "extract_sha256": sha, "single_status": single_status}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "global_models.json").write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    write_report(res)
    print(f"wrote {OUT_DIR / 'global_models.md'}")


if __name__ == "__main__":
    main()
