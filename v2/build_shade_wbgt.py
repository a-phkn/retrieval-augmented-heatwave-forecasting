"""
Adds wbgt_shade to Delhi's hourly development file (PROTOCOL_V2 amendment A1). Writes
data/raw/era5_v2/_wbgt/delhi_hourly_dev_shade.parquet (time, cell, t, wbgt, wbgt_shade). The 2019+ file is not
touched here; it is processed only after the test-lock tag (--test, which calls v2.test_lock_v2.require()).

Run from repo root:  python -m v2.build_shade_wbgt
"""
from __future__ import annotations

import argparse

import pandas as pd

from v2.build_delhi_hourly import OUT
from v2.shade_wbgt import shade_wbgt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args(argv)
    part = "test" if a.test else "dev"
    if a.test:
        from v2.test_lock_v2 import require
        require()
    h = pd.read_parquet(OUT / f"delhi_hourly_{part}.parquet")
    if not a.test and h["time"].dt.year.max() > 2018:
        raise RuntimeError("development file contains 2019+ rows; refusing")
    h["wbgt_shade"] = shade_wbgt(h)
    d = h["wbgt"] - h["wbgt_shade"]
    print(f"sun - shade WBGT (C): mean {d.mean():.2f}, p5 {d.quantile(.05):.2f}, p95 {d.quantile(.95):.2f}, "
          f"min {d.min():.2f}; daytime (cosz>0) mean {d[h['cosz'] > 0].mean():.2f}")
    h[["time", "cell", "t", "wbgt", "wbgt_shade"]].to_parquet(OUT / f"delhi_hourly_{part}_shade.parquet", index=False)
    print(f"Wrote delhi_hourly_{part}_shade.parquet {h.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
