"""
Per-cell WBGT "ingredients" at each cell's own daily peak hour (physics head design D3,
decision 2026-10-07).

For each of Delhi's 9 ERA5 cells and each complete day, the hour at which that cell's hourly
Liljegren WBGT peaks, and the formula's inputs at that hour: air temperature t (C), relative
humidity rh (%), surface pressure (hPa), 10 m wind (m/s), global irradiance ghi (W/m2),
direct fraction fdir (direct / global, 0 at night), mean sunlit cosine zenith cosz, plus the
peak WBGT itself. By construction the 9-cell mean of `wbgt__c<k>` is the forecast target
wbgt_lj_max (datasets_v2/wbgt_liljegren_daily.parquet), and the exact formula applied to the
stored inputs returns `wbgt__c<k>`: the physics head can match the target exactly given
perfect ingredients (evaluation_v2/physics_head_standin.md, R4).

Output: datasets_v2/peak_ingredients_daily.parquet, index `date`, columns `<var>__c<k>`.
Like the other v2 datasets it covers the whole download; training code drops 2019+ on read.

Run from repo root:  python -m pipeline.build_peak_ingredients
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.build_wbgt_liljegren import END_DATE, OUT_DIR, cell_hourly_wbgt

OUT = OUT_DIR / "peak_ingredients_daily.parquet"
INGREDIENTS = ("t", "rh", "pressure", "wind", "ghi", "fdir", "cosz")
CELLS = range(1, 10)


def peak_rows(h: pd.DataFrame) -> pd.DataFrame:
    """One row per complete day: the inputs and WBGT at that day's WBGT peak hour."""
    h = h.assign(date=h["time"].dt.normalize(), hour=h["time"].dt.hour)
    ghi = h["ghi"].to_numpy()
    direct = np.clip(ghi - h["diffuse"].to_numpy(), 0.0, None)
    h["fdir"] = np.clip(np.divide(direct, ghi, out=np.zeros_like(ghi), where=ghi > 0), 0.0, 1.0)
    complete = h.groupby("date")["wbgt"].transform("size") == 24
    h = h[complete]
    peak = h.loc[h.groupby("date")["wbgt"].idxmax()].set_index("date")
    return peak[[*INGREDIENTS, "wbgt", "hour"]]


def build(cells=CELLS) -> pd.DataFrame:
    parts = []
    for cell in cells:
        p = peak_rows(cell_hourly_wbgt(cell))
        parts.append(p.add_suffix(f"__c{cell}"))
        print(f"  cell {cell}: {len(p)} days, mean peak hour {p['hour'].mean():.1f}", flush=True)
    out = pd.concat(parts, axis=1, join="inner")
    return out[out.index <= END_DATE].sort_index()


def main() -> None:
    out = build()
    out.index.name = "date"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT)
    print(f"Wrote {OUT.name} {out.shape}")


if __name__ == "__main__":
    main()
