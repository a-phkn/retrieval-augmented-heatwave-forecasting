"""pipeline/download_era5_upstream.py -- no network: requests are mocked."""
from datetime import date

import pandas as pd
import pytest

import pipeline.download_era5_upstream as up
from pipeline.download_era5_upstream import DAILY_VARIABLES, NODES, END_DATE, node_id, out_path, params_for, years

NODE = (26, 72)
Y2019 = (date(2019, 1, 1), date(2019, 12, 31))


def test_node_set_is_the_fixed_27_point_lattice():
    lattice = {(a, o) for a in range(24, 33, 2) for o in range(68, 79, 2)}
    dropped = {(32, 68), (30, 70), (32, 78)}  # > 1,000 m (elevation probe 2026-10-05)
    assert len(NODES) == 27 and len(set(NODES)) == 27
    assert set(NODES) == lattice - dropped
    assert node_id(26, 72) == "n26e072" and node_id(32, 70) == "n32e070"


def test_years_cover_fixed_range_and_clamp_last_year():
    ys = years()
    assert ys[0] == (date(1980, 1, 1), date(1980, 12, 31))
    assert ys[-1] == (date(2026, 1, 1), END_DATE) and END_DATE == date(2026, 9, 6)
    assert len(ys) == 47


def test_params_daily_local_time_era5_and_at_most_10_variables():
    p = params_for(NODE, *Y2019)
    assert p["models"] == "era5" and p["timezone"] == "Asia/Kolkata" and p["wind_speed_unit"] == "ms"
    assert p["daily"].split(",") == DAILY_VARIABLES and len(DAILY_VARIABLES) <= 10
    assert (p["latitude"], p["longitude"]) == NODE and "hourly" not in p


def _payload(first, last, null_at=None, shift_days=0, lat=26.0, lon=72.0, drop=None):
    n = (last - first).days + 1
    daily = {"time": pd.date_range(pd.Timestamp(first) + pd.Timedelta(days=shift_days), periods=n, freq="D")
             .strftime("%Y-%m-%d").tolist()}
    for var in DAILY_VARIABLES:
        daily[var] = [1.0] * n
    if null_at is not None:
        daily["temperature_2m_max"][null_at] = None
    if drop:
        del daily[drop]
    return {"latitude": lat, "longitude": lon, "elevation": 133.0, "daily": daily}


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise up.requests.HTTPError(str(self.status_code))

    def json(self):
        return self._p


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, url, params, timeout):
        self.calls += 1
        return self.responses.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(up.time, "sleep", lambda s: None)


def test_payload_validation():
    assert up.payload_complete(_payload(*Y2019), NODE, *Y2019)
    assert not up.payload_complete(_payload(*Y2019, null_at=10), NODE, *Y2019)
    assert not up.payload_complete(_payload(*Y2019, shift_days=1), NODE, *Y2019)  # right count, wrong range
    assert not up.payload_complete(_payload(*Y2019, drop="soil_moisture_0_to_7cm_mean"), NODE, *Y2019)
    assert not up.payload_complete(_payload(*Y2019, lat=26.5), NODE, *Y2019)  # snapped to the wrong grid point
    assert up.payload_complete(_payload(*Y2019, lat=26.1, lon=71.9), NODE, *Y2019)  # within the grid half-cell
    assert not up.payload_complete({"daily": _payload(*Y2019)["daily"]}, NODE, *Y2019)  # no location
    assert not up.payload_complete([], NODE, *Y2019)


def test_download_writes_complete_file_atomically(tmp_path):
    s = _Session([_Resp(_payload(*Y2019))])
    assert up.download_one(s, NODE, *Y2019, root=tmp_path)
    path = out_path(NODE, Y2019[0], tmp_path)
    assert path == tmp_path / "n26e072" / "2019.json" and up.is_complete(path, NODE, *Y2019)
    assert not list(tmp_path.rglob("*.tmp"))


def test_resume_skips_complete_and_refetches_incomplete_files(tmp_path):
    path = out_path(NODE, Y2019[0], tmp_path)
    up.write_atomic(path, _payload(*Y2019))
    s = _Session([])
    assert up.download_one(s, NODE, *Y2019, root=tmp_path) and s.calls == 0
    path.write_text("{truncated", encoding="utf-8")
    s = _Session([_Resp(_payload(*Y2019))])
    assert up.download_one(s, NODE, *Y2019, root=tmp_path) and s.calls == 1


def test_bad_responses_are_never_written(tmp_path):
    s = _Session([_Resp(_payload(*Y2019, null_at=3)) for _ in range(2)])
    assert not up.download_one(s, NODE, *Y2019, root=tmp_path, retries=2)
    s = _Session([_Resp({}, status=500) for _ in range(2)])
    assert not up.download_one(s, NODE, *Y2019, root=tmp_path, retries=2)
    assert not out_path(NODE, Y2019[0], tmp_path).exists()


def test_rate_limit_waits_do_not_use_up_retries(tmp_path):
    s = _Session([_Resp({}, status=429) for _ in range(5)] + [_Resp(_payload(*Y2019))])
    assert up.download_one(s, NODE, *Y2019, root=tmp_path, retries=1) and s.calls == 6


def test_persistent_rate_limit_stops_the_whole_run(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "plan", lambda *a, **k: [(NODE, *Y2019), ((28, 74), *Y2019)])
    session = _Session([_Resp({}, status=429) for _ in range(50)])
    monkeypatch.setattr(up.requests, "Session", lambda: session)
    assert up.run(sleep_s=0, root=tmp_path) == 2
    assert session.calls == 11  # 10 waits + the 11th 429, then stop: the second node is never requested


def test_run_skips_a_failing_year_and_continues(tmp_path, monkeypatch):
    other = (28, 74)
    monkeypatch.setattr(up, "plan", lambda *a, **k: [(NODE, *Y2019), (other, *Y2019)])
    good = _payload(*Y2019, lat=28.0, lon=74.0)
    monkeypatch.setattr(up.requests, "Session", lambda: _Session([_Resp({}, status=500)] * 4 + [_Resp(good)]))
    assert up.run(sleep_s=0, root=tmp_path) == 1  # reports the failure
    assert up.is_complete(out_path(other, Y2019[0], tmp_path), other, *Y2019)  # but finished the next node
    assert up.verify(root=tmp_path) == 1


def test_call_estimate_matches_open_meteo_weighting():
    # one full year ~ 365/14 calls; a short request still costs at least one call
    assert up.estimated_calls([(NODE, *Y2019)]) == pytest.approx(365 / 14)
    assert up.estimated_calls([(NODE, date(2026, 9, 1), date(2026, 9, 6))]) == 1.0
    full = up.estimated_calls(up.plan())
    assert 30_000 < full < 36_000  # ~3-4 days of the free 10,000/day quota
