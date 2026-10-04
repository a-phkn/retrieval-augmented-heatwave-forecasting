"""pipeline/download_era5_v2.py -- no network: requests are mocked."""
import ast
import json
from datetime import date

import pytest

import pipeline.download_era5_v2 as dl
from pipeline.download_era5_v2 import CELLS, END_DATE, months, out_path, params_for


def test_cells_match_v1_downloader():
    tree = ast.parse((dl.REPO_ROOT / "download_era5.py").read_text(encoding="utf-8"))
    v1 = next(ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "CELLS")
    assert CELLS == v1


def test_months_cover_fixed_range_and_clamp_last_month():
    ms = months()
    assert ms[0] == (date(1980, 1, 1), date(1980, 1, 31))
    assert ms[-1] == (date(2026, 9, 1), END_DATE) and END_DATE == date(2026, 9, 6)
    assert len(ms) == 46 * 12 + 9
    assert ms[1] == (date(1980, 2, 1), date(1980, 2, 29))  # leap year


def test_params_use_era5_local_time_and_at_most_10_variables():
    p = params_for(5, date(2019, 6, 1), date(2019, 6, 30))
    assert p["models"] == "era5" and p["timezone"] == "Asia/Kolkata" and p["wind_speed_unit"] == "ms"
    assert len(p["hourly"].split(",")) <= 10
    assert (p["latitude"], p["longitude"]) == CELLS[4]


def _payload(first, last, null_at=None, shift_days=0):
    import pandas as pd

    n = ((last - first).days + 1) * 24
    start = pd.Timestamp(first) + pd.Timedelta(days=shift_days)
    hourly = {"time": pd.date_range(start, periods=n, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()}
    for var in dl.HOURLY_VARIABLES:
        hourly[var] = [1.0] * n
    if null_at is not None:
        hourly["dew_point_2m"][null_at] = None
    return {"hourly": hourly}


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise dl.requests.HTTPError(str(self.status_code))

    def json(self):
        return self._p


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, url, params, timeout):
        self.calls += 1
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)


def test_download_writes_complete_file_atomically(tmp_path):
    first, last = date(2019, 6, 1), date(2019, 6, 30)
    s = _Session([_Resp(_payload(first, last))])
    assert dl.download_one(s, 1, first, last, root=tmp_path)
    path = out_path(1, first, tmp_path)
    assert dl.is_complete(path, first, last)
    assert not list(tmp_path.rglob("*.tmp"))


def test_resume_skips_complete_files(tmp_path):
    first, last = date(2019, 6, 1), date(2019, 6, 30)
    dl.write_atomic(out_path(1, first, tmp_path), _payload(first, last))
    s = _Session([])
    assert dl.download_one(s, 1, first, last, root=tmp_path) and s.calls == 0


def test_incomplete_or_null_files_are_redownloaded(tmp_path):
    first, last = date(2019, 6, 1), date(2019, 6, 30)
    path = out_path(1, first, tmp_path)
    dl.write_atomic(path, _payload(first, last, null_at=5))
    assert not dl.is_complete(path, first, last)
    path.write_text("{truncated", encoding="utf-8")
    assert not dl.is_complete(path, first, last)
    s = _Session([_Resp(_payload(first, last))])
    assert dl.download_one(s, 1, first, last, root=tmp_path) and s.calls == 1


def test_rate_limit_then_success(tmp_path):
    first, last = date(2019, 6, 1), date(2019, 6, 30)
    s = _Session([_Resp({}, status=429), _Resp(_payload(first, last))])
    assert dl.download_one(s, 1, first, last, root=tmp_path) and s.calls == 2


def test_gives_up_after_retries_and_leaves_no_file(tmp_path):
    first, last = date(2019, 6, 1), date(2019, 6, 30)
    s = _Session([_Resp({}, status=500) for _ in range(4)])
    assert not dl.download_one(s, 1, first, last, root=tmp_path, retries=4)
    assert not out_path(1, first, tmp_path).exists()


def test_rejects_response_with_nulls_and_writes_nothing(tmp_path):
    first, last = date(2019, 6, 1), date(2019, 6, 30)
    s = _Session([_Resp(_payload(first, last, null_at=3)) for _ in range(2)])
    assert not dl.download_one(s, 1, first, last, root=tmp_path, retries=2)
    assert not out_path(1, first, tmp_path).exists()  # validated in memory, never written


def test_rejects_wrong_time_range_even_with_right_count(tmp_path):
    first, last = date(2019, 6, 1), date(2019, 6, 30)
    assert not dl.payload_complete(_payload(first, last, shift_days=1), first, last)
    assert dl.payload_complete(_payload(first, last), first, last)


def test_rate_limit_waits_do_not_use_up_retries(tmp_path):
    first, last = date(2019, 6, 1), date(2019, 6, 30)
    s = _Session([_Resp({}, status=429) for _ in range(5)] + [_Resp(_payload(first, last))])
    assert dl.download_one(s, 1, first, last, root=tmp_path, retries=1) and s.calls == 6


def test_persistent_rate_limit_stops_the_whole_run(tmp_path, monkeypatch):
    months2 = [(date(2019, 6, 1), date(2019, 6, 30)), (date(2019, 7, 1), date(2019, 7, 31))]
    monkeypatch.setattr(dl, "months", lambda *a, **k: months2)

    class S(_Session):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    session = S([_Resp({}, status=429) for _ in range(50)])
    monkeypatch.setattr(dl.requests, "Session", lambda: session)
    assert dl.run([1], sleep_s=0, root=tmp_path) == 2
    assert session.calls == 11  # 10 waits + the 11th 429, then stop -- July is never requested


def test_run_skips_a_failing_month_and_continues(tmp_path, monkeypatch):
    months2 = [(date(2019, 6, 1), date(2019, 6, 30)), (date(2019, 7, 1), date(2019, 7, 31))]
    monkeypatch.setattr(dl, "months", lambda *a, **k: months2)
    bad = [_Resp({}, status=500) for _ in range(4)]
    good = [_Resp(_payload(*months2[1]))]

    class S(_Session):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(dl.requests, "Session", lambda: S(bad + good))
    assert dl.run([1], sleep_s=0, root=tmp_path) == 1  # reports the failure
    assert dl.is_complete(out_path(1, months2[1][0], tmp_path), *months2[1])  # but finished July
