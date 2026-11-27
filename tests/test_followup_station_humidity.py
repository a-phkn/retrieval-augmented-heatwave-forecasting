"""followup/station_humidity.py (P3-7 step 4) on hand-made report lines and hourly series (offline)."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from followup import isd_rebuild as ir
from followup import station_humidity as sh


def isd_line(stamp: str, td: str = "+0150", dq: str = "1") -> str:
    """A synthetic ISD report line with the format document's fixed positions (stamp = YYYYMMDDHHMM, UTC)."""
    control = "0000" + "421820" + "99999" + stamp + "4" + "+28580" + "+077200" + "FM-12" + "+0216" + "VIDD " + "V020"
    mandatory = "999" + "9" + "9" + "9999" + "9" + "99999" + "9" + "9" + "9" + "999999" + "9" + "9" + "9" + "+0300" \
        + "1" + td + dq + "99999" + "9"
    assert len(control) == 60 and len(mandatory) == 45
    return control + mandatory + "ADD" + "AA1999999999" + "REMSYN004BBXX" + "\n"


def write_gz(path: Path, lines: list[str]) -> Path:
    with gzip.open(path, "wt", encoding="latin-1") as f:
        f.writelines(lines)
    return path


DAY = pd.Timestamp("2010-05-10")  # an IST day
SYNOPTIC = pd.DatetimeIndex([DAY - pd.Timedelta(hours=3)] + [DAY + pd.Timedelta(hours=h) for h in range(0, 19, 3)])


def test_the_station_reading_is_taken_at_the_8_synoptic_times(tmp_path):
    lines = [isd_line(t.strftime("%Y%m%d%H%M"), td=f"+0{100 + 10 * i:03d}") for i, t in enumerate(SYNOPTIC)]
    lines += [isd_line((DAY + pd.Timedelta(hours=1)).strftime("%Y%m%d%H%M"), td="+0999"),  # 01 UTC: not synoptic
              isd_line((DAY + pd.Timedelta(hours=6, minutes=30)).strftime("%Y%m%d%H%M"), td="+0999"),  # minute 30
              isd_line((DAY + pd.Timedelta(hours=21)).strftime("%Y%m%d%H%M"), td="+0500", dq="2")]  # suspect value
    s = sh.station_instants(ir.read_reports([write_gz(tmp_path / "x.gz", lines)]))
    assert list(s.index) == list(SYNOPTIC) and list(s) == [10.0 + i for i in range(8)]
    assert sh._year(Path("421820-99999-1985.gz")) == 1985


def test_era5_is_taken_at_half_past_the_ist_hour_and_indexed_by_utc():
    idx = pd.date_range("2010-05-10 00:00", "2010-05-11 23:00", freq="h")  # IST hours
    e = sh.era5_instants(pd.Series(np.arange(len(idx), dtype=float), index=idx))
    assert e[pd.Timestamp("2010-05-10 00:00")] == 5.5  # 00 UTC = 05:30 IST: the mean of 05:00 and 06:00 IST
    assert e[pd.Timestamp("2010-05-09 21:00")] == 2.5  # 21 UTC the day before = 02:30 IST
    assert set(e.index.hour) == set(sh.SYNOPTIC_UTC) and (e.index.minute == 0).all()
    assert pd.Timestamp("2010-05-11 18:00") not in e.index  # 23:30 IST of the last day has no next hour
    with pytest.raises(ValueError, match="hourly"):
        sh.era5_instants(pd.Series(1.0, index=idx.delete(3)))


def test_a_day_needs_7_of_its_8_instants_with_both_values():
    station = pd.Series(np.arange(8, dtype=float), index=SYNOPTIC)
    hourly = pd.Series(30.0, index=pd.date_range("2010-05-09", "2010-05-12 23:00", freq="h"))
    cells = {c: hourly + c for c in range(1, 10)}  # cell c reads 30 + c
    d = sh.daily_means(sh.instant_frame(station, cells, nearest=4))
    assert list(d.index) == [DAY] and d.loc[DAY, "instants"] == 8
    assert (d.loc[DAY, "station"], d.loc[DAY, "era5_cells"], d.loc[DAY, "era5_nearest"]) == (3.5, 35.0, 34.0)
    gap = sh.daily_means(sh.instant_frame(station.drop(SYNOPTIC[0]), cells, nearest=4))
    assert gap.loc[DAY, "instants"] == 7 and gap.loc[DAY, "station"] == 4.0  # the mean of the 7 it has
    missing = cells[2].copy()
    missing[pd.Timestamp("2010-05-10 05:00")] = np.nan  # breaks 05:30 IST (00 UTC)
    missing[pd.Timestamp("2010-05-10 08:00")] = np.nan  # breaks 08:30 IST (03 UTC)
    d6 = sh.daily_means(sh.instant_frame(station, {**cells, 2: missing}, nearest=4))
    assert DAY not in d6.index  # one cell without a value drops the instant for every ERA5 record: 6 left


def test_a_season_counts_with_70_days_and_the_change_and_trend_are_per_decade():
    rows = []
    for year in range(1990, 2012):
        days = pd.date_range(f"{year}-03-15", periods=60 if year == 1995 else 100, freq="D")
        diff = 1.5 if year >= 2002 else 1.0
        base = 20.0 + 0.03 * (year - 1990)
        rows.append(pd.DataFrame({"station": base, "era5_cells": base + diff, "era5_nearest": base + diff - 0.2,
                                  "instants": 8}, index=days))
    october = pd.DataFrame({"station": 0.0, "era5_cells": 99.0, "era5_nearest": 99.0, "instants": 8},
                           index=pd.date_range("2000-10-01", periods=5, freq="D"))  # out of season
    s = sh.season_table(pd.concat(rows + [october]))
    assert not s.loc[1995, "counts"] and s.loc[1994, "counts"] and s.loc[1994, "days"] == 100
    assert s.loc[2000, "diff_cells"] == pytest.approx(1.0) and s.loc[2000, "days"] == 100  # October left out
    assert s.loc[2005, "diff_nearest"] == pytest.approx(1.3)
    c = sh.decade_change(s, "diff_cells")
    assert (c["seasons_first"], c["seasons_second"]) == (9, 10)  # 1995 does not count; 2000-2001 are in neither
    assert c["change"] == pytest.approx(0.5) and c["ci_low"] == pytest.approx(0.5) and c["ci_high"] == pytest.approx(0.5)
    assert c["mean_first"] == pytest.approx(1.0) and sh.decade_change(s, "diff_cells") == c  # seeded
    t = sh.trend(s, "station")
    assert t["per_decade"] == pytest.approx(0.3) and (t["first"], t["last"], t["seasons"]) == (1990, 2011, 21)
    assert sh.decade_change(s[s.index >= 2002], "diff_cells") is None  # no 1990s season
    assert sh.trend(s[s.index == 2005], "station") is None


def test_the_bootstrap_interval_widens_with_the_spread_of_seasons():
    years = list(range(1990, 2000)) + list(range(2002, 2012))
    rng = np.random.default_rng(3)
    s = pd.DataFrame({"diff_cells": rng.normal(0, 0.3, len(years)) + np.where(np.array(years) >= 2002, 0.4, 0.0),
                      "counts": True}, index=pd.Index(years, name="year"))
    c = sh.decade_change(s, "diff_cells")
    assert c["ci_low"] < c["change"] < c["ci_high"] and 0.1 < c["ci_high"] - c["ci_low"] < 1.0


def test_the_nearest_cell_and_only_the_years_asked_are_read(tmp_path):
    coords = {n: c for n, c in enumerate([(a, b) for a in (28.25, 28.5, 28.75) for b in (77.0, 77.25, 77.5)], 1)}
    assert sh.nearest_cell(*sh.COORDS["safdarjung"], coords) == 5 and sh.nearest_cell(*sh.COORDS["palam"], coords) == 4
    folder = tmp_path / "cell_1"
    folder.mkdir()

    def month(name: str, times: list[str], values: list, tz: str = "Asia/Kolkata") -> None:
        (folder / f"{name}.json").write_text(json.dumps({"latitude": 28.25, "longitude": 77.0, "timezone": tz,
                                                         "utc_offset_seconds": 19800 if tz == "Asia/Kolkata" else 0,
                                                         "hourly": {"time": times, "dew_point_2m": values}}),
                                             encoding="utf-8")

    month("2018-12", ["2018-12-31T22:00", "2018-12-31T23:00"], [10.0, None])
    month("2019-01", ["2019-01-01T00:00"], [99.0])  # a test year: never read
    s = sh.load_era5_dew(1, root=tmp_path)
    assert list(s.index) == list(pd.to_datetime(["2018-12-31 22:00", "2018-12-31 23:00"]))
    assert s.iloc[0] == 10.0 and np.isnan(s.iloc[1])
    assert sh.cell_coords(tmp_path) == {1: (28.25, 77.0)}
    month("2018-11", ["2018-11-30T23:00"], [5.0], tz="GMT")
    with pytest.raises(ValueError, match="not IST"):
        sh.load_era5_dew(1, root=tmp_path)


def test_the_report_has_a_summary_and_both_stations(tmp_path, monkeypatch):
    def seasons(first: int, last: int, short: int | None = None) -> list[dict]:
        return [{"year": y, "days": 60 if y == short else 120, "station": 18.0, "era5_cells": 19.0,
                 "era5_nearest": 18.8, "diff_cells": 1.0, "diff_nearest": 0.8, "counts": y != short}
                for y in range(first, last + 1)]

    change = {"first": "1990-1999", "second": "2002-2011", "seasons_first": 10, "seasons_second": 10,
              "mean_first": 0.9, "mean_second": 1.3, "change": 0.4, "ci_low": 0.1, "ci_high": 0.7}
    t = {"per_decade": 0.2, "seasons": 39, "first": 1980, "last": 2018}
    res = {"decades": sh.DECADES, "min_instants": 7, "min_season_days": 70, "n_boot": 10_000, "seed": 0,
           "stations": {"safdarjung": {"files": 39, "nearest_cell": 5, "nearest_centre": [28.5, 77.25], "days": 13000,
                                       "in_season_days": 5000, "seasons": seasons(1980, 2018, short=1995),
                                       "change": {"diff_cells": change, "diff_nearest": change},
                                       "trend": {k: t for k in sh.RECORDS}},
                        "palam": {"files": 33, "nearest_cell": 4, "nearest_centre": [28.5, 77.0], "days": 5000,
                                  "in_season_days": 2000, "seasons": seasons(2002, 2018),
                                  "change": {"diff_cells": None, "diff_nearest": None},
                                  "trend": {k: None for k in sh.RECORDS}}}}
    monkeypatch.setattr(sh, "OUT_DIR", tmp_path)
    sh.write_report(res)
    text = (tmp_path / "station_humidity.md").read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == ["## Summary", f"## {sh.NAMES['safdarjung']}", f"## {sh.NAMES['palam']}", "## Notes"]
    assert "+0.40 °C [+0.10, +0.70] (10 and 10 seasons)" in text and "n/a (no season in one of the decades)" in text
    assert "| 1995 (not counted: fewer than 70 days) | 60 |" in text and "38 (1980-2018)" in text
    assert "nan" not in text.lower().replace("n/a", "")
