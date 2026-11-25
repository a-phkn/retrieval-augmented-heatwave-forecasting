"""followup/isd_rebuild.py (P3-7): the ISD download and its manifest (offline: a fake session, no network)."""
import gzip
import json

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


def test_the_manifest_is_written_even_when_the_run_stops(tmp_path):
    s = Session({ir.url("000000-99999", 2000): [Reply(200, GZ)], ir.url("000000-99999", 2001): [Reply(403)]})
    with pytest.raises(requests.HTTPError):
        ir.download(s, ST, [2000, 2001], tmp_path / "raw", tmp_path / "m.json", tmp_path / "ghcn", sleep=lambda _: None)
    m = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert list(m["isd_files"]) == ["000000-99999-2000"]  # the finished file is kept for the re-run
