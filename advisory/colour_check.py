"""
Descriptive checks of the rule engine (decisions.md 2026-10-26 and 2026-10-27; not gates). Development years
2007-2018 only; each year's normal comes from its own fold's training years.

Part A, perfect forecast: how often each alert colour fires if the forecast equalled the observed Tmax
(forecast_rule="value"): season days by the colour of their heat-wave run, and the headline colour per advisory.
Part B, real forecasts: C3-hop's 10-seed ensemble Tmax forecasts and its development hot-day probabilities (as in
post_test/warning_rules.py), under the probability rule (the advisory's rule since 2026-10-27) and the old value
rule, compared with the perfect-forecast headline colour of the same advisory.

Run from repo root:  python -m advisory.colour_check  ->  advisory/colour_check.md
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from advisory import rules
from advisory.library import COLOURS, colour_rank
from pipeline.labels_v2 import in_season
from training.folds import FOLDS, fold_bounds, fold_daily

OUT = Path(__file__).resolve().parent / "colour_check.md"
TMAX_RUN = "A1prime_hw5_C3hop"


def observed_series() -> pd.DataFrame:
    """2007-2018 daily Tmax with each date's normal from its fold's training years, plus 7 lead-in days per block."""
    parts = []
    for fold in FOLDS:
        d, _ = fold_daily(fold, "t_max", "v2")
        _, a, b = fold_bounds(fold)
        lead_in = a - pd.Timedelta(days=rules.OBSERVED_DAYS)
        parts.append(d.loc[lead_in:b, ["t_max", "clim_mean_t_max"]].assign(block=fold))
    s = pd.concat(parts).rename(columns={"t_max": "tmax", "clim_mean_t_max": "normal"})
    s.index.name = "date"
    return s


def forecast_table() -> pd.DataFrame:
    """C3-hop's development ensemble Tmax forecasts with hot-day probabilities (0 outside the label season)."""
    from evaluation.compare_v2 import ensemble, load_run
    from post_test.warning_rules import prob_rows

    val = load_run(TMAX_RUN)
    ens = pd.concat([ensemble(val[val["fold"] == f].drop(columns="fold")).assign(fold=f) for f in FOLDS],
                    ignore_index=True)
    p = prob_rows(TMAX_RUN, "Tmax", "dev")[["query_date", "lead", "p"]]
    out = ens.merge(p, on=["query_date", "lead"], how="left", validate="one_to_one")
    if (~in_season(out["target_date"]) & out["p"].notna()).any():
        raise ValueError("a probability outside the label season")
    out["p"] = out["p"].fillna(0.0)  # outside the label season a heatwave day is impossible
    return out


def perfect_forecast_part(s: pd.DataFrame) -> tuple[Counter, pd.DataFrame]:
    run_days, rows = Counter(), []
    for fold, blk in s.groupby("block", sort=False):
        blk = blk.reset_index()
        hw, sev = rules.classify(blk["date"], blk["tmax"], blk["normal"])
        colours = rules.day_colours(hw, sev)
        _, a, b = fold_bounds(fold)
        inside = (blk["date"] >= a) & (blk["date"] <= b) & in_season(blk["date"])
        run_days.update(c for c, keep in zip(colours, inside) if keep)
        for i in range(rules.OBSERVED_DAYS, len(blk) - rules.FORECAST_DAYS + 1):
            if not inside.iloc[i]:
                continue
            obs = blk.iloc[i - rules.OBSERVED_DAYS:i][["date", "tmax", "normal"]]
            fc = blk.iloc[i:i + rules.FORECAST_DAYS][["date", "tmax", "normal"]]
            rows.append({"date": blk.at[i, "date"],
                         "headline": rules.assess(obs, fc, forecast_rule="value").headline_colour})
    return run_days, pd.DataFrame(rows)


def real_forecast_part(s: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """One row per season query date: the headline colour under the probability rule, the value rule, and the
    perfect forecast."""
    rows = []
    for fold, g in fc.groupby("fold", sort=False):
        obs_s = s[s["block"] == fold]
        for q, f in g.groupby("query_date"):
            if not in_season([q])[0]:
                continue
            f = f.sort_values("lead")
            obs_dates = pd.date_range(q - pd.Timedelta(days=rules.OBSERVED_DAYS), periods=rules.OBSERVED_DAYS)
            if not obs_dates.isin(obs_s.index).all():
                continue
            obs = pd.DataFrame({"date": obs_dates, "tmax": obs_s.loc[obs_dates, "tmax"].to_numpy(),
                                "normal": obs_s.loc[obs_dates, "normal"].to_numpy()})
            t = pd.DatetimeIndex(f["target_date"])
            normal = obs_s["normal"].reindex(t).to_numpy()
            fcast = pd.DataFrame({"date": t, "tmax": f["pred"].to_numpy(), "normal": normal, "p_hot": f["p"].to_numpy()})
            perfect = fcast.assign(tmax=f["actual"].to_numpy())
            rows.append({"date": q,
                         "probability": rules.assess(obs, fcast).headline_colour,
                         "value": rules.assess(obs, fcast, forecast_rule="value").headline_colour,
                         "perfect": rules.assess(obs, perfect, forecast_rule="value").headline_colour})
    return pd.DataFrame(rows)


def alert_scores(perfect: pd.Series, forecast: pd.Series) -> dict:
    """'Any alert' (Yellow or above): share of perfect-forecast alerts the forecast also raised, and share of forecast
    alerts the perfect forecast did not raise; exact-colour agreement."""
    pa = perfect.map(colour_rank) >= 1
    fa = forecast.map(colour_rank) >= 1
    return {"caught": float((pa & fa).sum() / pa.sum()) if pa.sum() else float("nan"),
            "false": float((fa & ~pa).sum() / fa.sum()) if fa.sum() else float("nan"),
            "same_colour": float((perfect == forecast).mean()), "n_alerts_forecast": int(fa.sum()),
            "n_alerts_perfect": int(pa.sum())}


def _crosstab(r: pd.DataFrame, col: str) -> list[str]:
    ct = pd.crosstab(r["perfect"], r[col]).reindex(index=list(COLOURS), columns=list(COLOURS), fill_value=0)
    L = ["| Perfect forecast \\ forecast | " + " | ".join(COLOURS) + " |", "|---|" + "---|" * len(COLOURS)]
    L += [f"| {c} | " + " | ".join(str(int(ct.at[c, k])) for k in COLOURS) + " |" for c in COLOURS]
    return L


def main() -> None:
    s = observed_series()
    run_days, h = perfect_forecast_part(s)
    r = real_forecast_part(s, forecast_table())
    h["year"] = h["date"].dt.year
    per_year = h.groupby(["year", "headline"]).size().unstack(fill_value=0).reindex(columns=list(COLOURS), fill_value=0)
    n_days = sum(run_days.values())
    L = ["# Rule engine: how often each colour fires on 2007-2018 (descriptive checks)", "",
         "Development years only; season Mar 15 - Jul 31; normals from each fold's training years. Written by "
         "`python -m advisory.colour_check` (decisions.md 2026-10-26 and 2026-10-27).", "",
         "## A. Perfect forecast (the observed Tmax given as the forecast)", "",
         f"Season days by the colour of their heat-wave run ({n_days} days):", "",
         "| Colour | Days | Share |", "|---|---|---|"]
    L += [f"| {c} | {run_days[c]} | {100 * run_days[c] / n_days:.1f}% |" for c in COLOURS]
    L += ["", f"Headline colour of the 5-day advisory, one per season day ({len(h)} advisories):", "",
          "| Year | " + " | ".join(COLOURS) + " |", "|---|" + "---|" * len(COLOURS)]
    L += [f"| {y} | " + " | ".join(str(int(per_year.at[y, c])) for c in COLOURS) + " |" for y in per_year.index]
    tot = per_year.sum()
    L += [f"| **all** | " + " | ".join(f"**{int(tot[c])}** ({100 * tot[c] / len(h):.0f}%)" for c in COLOURS) + " |", ""]
    L += [f"## B. Real forecasts: C3-hop, {len(r)} season advisories", "",
          f"Forecast heatwave day under the probability rule: hot-day probability >= {rules.P_HOT_LEVEL} (the "
          "advisory's rule); under the value rule: the forecast Tmax itself meets the label rule (the old rule). "
          "'Perfect' = the colour the observed weather would have given for the same 5 days.", "",
          "| Headline colour share | " + " | ".join(COLOURS) + " |", "|---|" + "---|" * len(COLOURS)]
    for col, name in (("perfect", "Perfect forecast"), ("probability", "Probability rule"), ("value", "Value rule (old)")):
        sh = r[col].value_counts(normalize=True).reindex(list(COLOURS), fill_value=0)
        L.append(f"| {name} | " + " | ".join(f"{100 * sh[c]:.0f}%" for c in COLOURS) + " |")
    L += ["", "| Rule | Perfect-forecast alerts (Yellow+) also raised | Forecast alerts not in the perfect forecast | "
              "Same colour |", "|---|---|---|---|"]
    for col, name in (("probability", "Probability rule"), ("value", "Value rule (old)")):
        a = alert_scores(r["perfect"], r[col])
        L.append(f"| {name} | {100 * a['caught']:.0f}% | {100 * a['false']:.0f}% ({a['n_alerts_forecast']} alerts vs "
                 f"{a['n_alerts_perfect']}) | {100 * a['same_colour']:.0f}% |")
    for col, name in (("probability", "Probability rule"), ("value", "Value rule (old)")):
        L += ["", f"{name}: number of advisories by colour", ""] + _crosstab(r, col)
    L.append("")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
