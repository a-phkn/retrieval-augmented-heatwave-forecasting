"""followup/isd_rebuild.py (P3-7): the ISD download and its manifest (offline: a fake session, no network)."""
import gzip
import json

import numpy as np
import pandas as pd
import pytest
import requests

from followup import isd_rebuild as ir


class Reply:
    def __init__(self, status: int, content: bytes = b""):
        self.status_code, self.content = status, content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class Session:
    """Replies in order per URL; records every request."""

    def __init__(self, replies: dict[str, list[Reply]]):
        self.replies, self.calls = {u: list(r) for u, r in replies.items()}, []

    def get(self, u: str, timeout: int) -> Reply:
        self.calls.append(u)
        return self.replies[u].pop(0)


GZ = gzip.compress(b"0000 a fake ISD line\n")
ST = {"x": "000000-99999"}


def test_url_and_path_follow_the_raw_archive_layout(tmp_path):
    assert ir.url("421820-99999", 1985) == "https://www.ncei.noaa.gov/pub/data/noaa/1985/421820-99999-1985.gz"
    assert ir.raw_path("421820-99999", 1985, tmp_path) == tmp_path / "421820-99999" / "421820-99999-1985.gz"
    assert set(ir.STATIONS.values()) == {"421820-99999", "421810-99999", "426470-99999", "432790-99999"}
    assert (min(ir.YEARS), max(ir.YEARS)) == (1980, 2025)


def test_download_writes_files_and_a_manifest_and_records_missing_years(tmp_path):
    s = Session({ir.url("000000-99999", 2000): [Reply(200, GZ)], ir.url("000000-99999", 2001): [Reply(404)]})
    m = ir.download(s, ST, [2000, 2001], tmp_path / "raw", tmp_path / "m.json", tmp_path / "ghcn", sleep=lambda _: None)
    assert ir.raw_path("000000-99999", 2000, tmp_path / "raw").read_bytes() == GZ
    assert m["isd_files"]["000000-99999-2000"] == {"station": "x", "url": ir.url("000000-99999", 2000),
                                                    "bytes": len(GZ), "sha256": ir.sha256(GZ)}
    assert m["isd_files"]["000000-99999-2001"]["missing"] is True
    assert json.loads((tmp_path / "m.json").read_text(encoding="utf-8")) == m
    assert m["ghcn_received"]["IN020040900"]["on_disk_matches"] is False  # not in tmp_path/ghcn


def test_files_already_on_disk_are_not_fetched_again(tmp_path):
    first = Session({ir.url("000000-99999", 2000): [Reply(200, GZ)]})
    ir.download(first, ST, [2000], tmp_path / "raw", tmp_path / "m.json", tmp_path / "ghcn", sleep=lambda _: None)
    again = Session({})  # any request would fail with KeyError
    ir.download(again, ST, [2000], tmp_path / "raw", tmp_path / "m.json", tmp_path / "ghcn", sleep=lambda _: None)
    assert again.calls == []


def test_an_html_page_is_rejected_and_server_errors_are_retried(tmp_path):
    u = ir.url("000000-99999", 2000)
    with pytest.raises(ValueError, match="not gzip"):
        ir.fetch(Session({u: [Reply(200, b"<html>not here</html>")]}), u, sleep=lambda _: None)
    waits = []
    assert ir.fetch(Session({u: [Reply(503), Reply(200, GZ)]}), u, sleep=waits.append) == GZ
    assert waits == [10]
    with pytest.raises(requests.HTTPError):
        ir.fetch(Session({u: [Reply(503)] * 4}), u, sleep=lambda _: None)


def isd_line(stamp: str, rtype: str = "FM-12", t: str = "+0300", tq: str = "1", td: str = "+0150", dq: str = "1",
             ka: str = "", rem: str = "REMSYN004BBXX") -> str:
    """A synthetic ISD report line with the format document's fixed positions (stamp = YYYYMMDDHHMM, UTC)."""
    control = "0000" + "421820" + "99999" + stamp + "4" + "+28580" + "+077200" + rtype.ljust(5) + "+0216" + "VIDD " + "V020"
    mandatory = "999" + "9" + "9" + "9999" + "9" + "99999" + "9" + "9" + "9" + "999999" + "9" + "9" + "9" + t + tq + td + dq \
        + "99999" + "9"
    assert len(control) == 60 and len(mandatory) == 45
    return control + mandatory + "ADD" + "AA1999999999" + ka + rem + "\n"


def write_gz(path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="latin-1") as f:
        f.writelines(lines)
    return path


def test_values_quality_codes_and_ka_groups_are_read_from_their_positions():
    line = isd_line("201006151200", t="+0412", tq="1", td="-0015", dq="2", ka="KA1120M+04355KA2120N+02101")
    assert ir.value(line[87:92], line[92]) == 41.2 and np.isnan(ir.value(line[93:98], line[98]))  # quality 2 dropped
    assert ir.value("+9999", "9") != ir.value("+9999", "9")  # missing
    assert [(p, c) for p, c, _, _ in ir.extremes(line)] == [(12.0, "M"), (12.0, "N")]
    assert ir.extremes(isd_line("201006151200", rem="REMKA1120M+04355")) == []  # a group-like text in remarks is ignored


def test_one_report_per_time_synop_first_then_file_order(tmp_path):
    p = write_gz(tmp_path / "x.gz", [isd_line("201006151200", "FM-15", t="+0410"),
                                     isd_line("201006151200", "FM-12", t="+0405"),
                                     isd_line("201006151200", "FM-12", t="+0400"),
                                     isd_line("201006151300", "SY-MT", t="+0399"),
                                     isd_line("201006151300", "FM-16", t="+0398")])
    rep = ir.read_reports([p])
    assert list(rep["t"]) == [40.5, 39.8]  # the first FM-12 at 12:00; FM-16 beats another type at 13:00


def test_daily_values_follow_the_ist_day_and_the_coverage_rules(tmp_path):
    day = pd.Timestamp("2010-06-15")
    hours = pd.date_range(day - pd.Timedelta(hours=5), day + pd.Timedelta(hours=18), freq="h")  # 19 UTC D-1 .. 18 UTC D
    lines = []
    for h in hours:
        ka = "KA1120M+04210" if h.hour == 12 else ("KA1120N+02800" if h.hour == 3 else "")
        lines.append(isd_line(h.strftime("%Y%m%d%H%M"), t=f"+0{300 + h.hour * 4:03d}", ka=ka))
    rep = ir.read_reports([write_gz(tmp_path / "x.gz", lines)])
    d = ir.daily(rep, day, day)
    r = d.loc[day]
    assert r["hours_t"] == 24 and r["tmax_reported"] == 42.1 and r["tmin_reported"] == 28.0
    assert r["tmax_hourly"] == (300 + 23 * 4) / 10  # 23 UTC of D-1 is the warmest synthetic hour
    assert r["tmin_hourly"] == 30.0  # 00 UTC of D in the 21-03 UTC window
    no_afternoon = rep[~rep["time"].dt.hour.isin([9])]
    assert np.isnan(ir.daily(no_afternoon, day, day).loc[day, "tmax_hourly"])  # one afternoon hour missing


def test_the_next_mornings_24_hour_maximum_is_dated_to_the_day_before(tmp_path):
    day = pd.Timestamp("2016-06-15")
    lines = [isd_line("201606150300", ka="KA1120N+02800"), isd_line("201606160300", ka="KA1240M+04300KA2120N+02900")]
    d = ir.daily(ir.read_reports([write_gz(tmp_path / "x.gz", lines)]), day, day + pd.Timedelta(days=1))
    assert d.loc[day, "tmax_24h_next_morning"] == 43.0 and np.isnan(d.loc[day + pd.Timedelta(days=1), "tmax_24h_next_morning"])


def frame(days: pd.DatetimeIndex, **cols) -> pd.DataFrame:
    base = {c: np.nan for c in ("tmax_reported", "tmax_reported_period_h", "tmax_hourly", "tmax_24h_next_morning")}
    return pd.DataFrame({**base, **cols}, index=days)


def test_deviation_a_fills_safdarjungs_gaps_from_2016_only_when_the_check_passes():
    days = pd.date_range("2014-01-01", "2016-12-31", freq="D")
    wave = pd.Series(np.sin(np.arange(len(days)) / 3) * 5 + 35, index=days)
    reported = wave.where(days < ir.FILL_FROM)  # the 12 UTC series stops at 2016
    good = frame(days, tmax_reported=reported, tmax_24h_next_morning=wave + 0.1)
    d, c = ir.deviation("safdarjung", good)
    assert c["A_next_morning_vs_12utc"]["passes"] and c["A_days_filled"] == (days >= ir.FILL_FROM).sum()
    assert (d.loc[days >= ir.FILL_FROM, "tmax_source"] == "03utc_24h_next_morning").all()
    assert (d.loc[days < ir.FILL_FROM, "tmax"] == reported[days < ir.FILL_FROM]).all()
    shifted = frame(days, tmax_reported=reported, tmax_24h_next_morning=wave.shift(1))  # a day off: no fill
    d2, c2 = ir.deviation("safdarjung", shifted)
    assert not c2["A_next_morning_vs_12utc"]["passes"] and d2.loc[days >= ir.FILL_FROM, "tmax"].isna().all()


def test_deviation_b_keeps_12_hour_groups_only(tmp_path):
    days = pd.date_range("2010-01-01", periods=4, freq="D")
    d, c = ir.deviation("palam", frame(days, tmax_reported=[40.0, 41.0, 42.0, 43.0],
                                       tmax_reported_period_h=[12.0, 24.0, np.nan, 12.0]))
    assert list(d["tmax"].fillna(-1)) == [40.0, -1, -1, 43.0] and not c["left_out"]
    assert list(d["tmax_source"]) == ["12utc_12h", "", "", "12utc_12h"]


def test_a_shifted_series_is_caught_by_the_lag_check():
    idx = pd.date_range("2000-01-01", periods=200, freq="D")
    truth = pd.Series(np.sin(np.arange(200) / 3) * 5 + 35, index=idx)
    mad = ir.lag_mad(truth.shift(1), truth)  # a file that carries the previous day's value
    assert min(mad, key=mad.get) == -1
    ok = ir.lag_mad(truth, truth)
    assert min(ok, key=ok.get) == 0


def test_the_manifest_is_written_even_when_the_run_stops(tmp_path):
    s = Session({ir.url("000000-99999", 2000): [Reply(200, GZ)], ir.url("000000-99999", 2001): [Reply(403)]})
    with pytest.raises(requests.HTTPError):
        ir.download(s, ST, [2000, 2001], tmp_path / "raw", tmp_path / "m.json", tmp_path / "ghcn", sleep=lambda _: None)
    m = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert list(m["isd_files"]) == ["000000-99999-2000"]  # the finished file is kept for the re-run
