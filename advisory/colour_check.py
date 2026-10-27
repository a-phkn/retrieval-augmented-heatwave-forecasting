"""
Descriptive check of the rule engine (decisions.md 2026-10-26; not a gate): how often each alert colour would fire on
the OBSERVED 2007-2018 Tmax if the forecast were perfect. Development years only; each year's normal comes from its
own fold's training years. Two views:
  - season days by colour (the colour of the run each day belongs to, over the whole season);
  - the engine as it runs: for every season day, the last 7 observed days + the next 5 "forecast" (here observed)
    days -> the headline colour for those 5 days.

Run from repo root:  python -m advisory.colour_check  ->  advisory/colour_check.md
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from advisory import rules
from advisory.library import COLOURS
from pipeline.labels_v2 import in_season
from training.folds import FOLDS, fold_bounds, fold_daily

OUT = Path(__file__).resolve().parent / "colour_check.md"


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


def main() -> None:
    s = observed_series()
    rows, run_days = [], Counter()
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
            rows.append({"date": blk.at[i, "date"], "headline": rules.assess(obs, fc).headline_colour})
    h = pd.DataFrame(rows)
    h["year"] = h["date"].dt.year
    per_year = h.groupby(["year", "headline"]).size().unstack(fill_value=0).reindex(columns=list(COLOURS), fill_value=0)
    n_days = sum(run_days.values())
    L = ["# Rule engine: how often each colour fires on observed 2007-2018 Tmax (descriptive check)", "",
         "Perfect-forecast case: the engine is given the observed Tmax as its 'forecast'. Season Mar 15 - Jul 31; "
         "normals from each fold's training years. Written by `python -m advisory.colour_check` (decisions.md "
         "2026-10-26).", "",
         f"## Season days by the colour of their heat-wave run ({n_days} days)", "",
         "| Colour | Days | Share |", "|---|---|---|"]
    L += [f"| {c} | {run_days[c]} | {100 * run_days[c] / n_days:.1f}% |" for c in COLOURS]
    L += ["", f"## Headline colour of the 5-day advisory, one per season day ({len(h)} advisories)", "",
          "| Year | " + " | ".join(COLOURS) + " |", "|---|" + "---|" * len(COLOURS)]
    L += [f"| {y} | " + " | ".join(str(int(per_year.at[y, c])) for c in COLOURS) + " |" for y in per_year.index]
    tot = per_year.sum()
    L += [f"| **all** | " + " | ".join(f"**{int(tot[c])}** ({100 * tot[c] / len(h):.0f}%)" for c in COLOURS) + " |", ""]
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
