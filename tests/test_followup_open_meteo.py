"""followup/open_meteo.py: request plans, reply checks, retries and the cache, all offline (no network)."""
import json
from datetime import date

import pytest

from followup import open_meteo as om


def hourly_reply(first: str, days: int, cols: dict, spot=(28.25, 77.0, 274.0)) -> dict:
    t0 = date.fromisoformat(first)
    times = [f"{date.fromordinal(t0.toordinal() + d).isoformat()}T{h:02d}:00" for d in range(days) for h in range(24)]
    return {"latitude": spot[0], "longitude": spot[1], "elevation": spot[2],
            "hourly": {"time": times, **{c: [f(i) for i in range(len(times))] for c, f in cols.items()}}}


class FakeResponse:
    def __init__(self, status: int, payload):
        self.status_code, self._payload = status, payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise om.requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return self.responses.pop(0)


def test_every_variable_and_lead_is_requested_at_most_10_columns_at_a_time():
    groups = om.column_groups(om.WBGT_VARS)
    assert [len(g) for g in groups] == [10, 10, 10]
    assert sorted(c for g in groups for c in g) == sorted(f"{v}_previous_day{L}" for v in om.WBGT_VARS for L in range(1, 6))
    assert om.column_groups(("temperature_2m",)) == [[f"temperature_2m_previous_day{L}" for L in range(1, 6)]]


def test_plans_cover_every_cell_month_and_run():
    pulls = {"x": ("m", ("temperature_2m",), date(2024, 1, 1), date(2024, 2, 10))}
    plan = om.previous_plan(pulls)
    assert len(plan) == 2 * 9 and plan[-1][3] == date(2024, 2, 10)  # 2 months x 9 cells, last month clamped
    runs = om.single_run_dates()
    assert runs[0] == date(2024, 3, 14) and runs[-1] == date(2026, 9, 1)  # q = r + 1 <= 2026-09-02, so q + 4 <= END
    p = om.single_params(runs[0])
    assert p["run"] == "2024-03-14T12:00" and p["latitude"].count(",") == 8 and p["models"] == "ecmwf_ifs"


def test_reply_checks_allow_archive_gaps_but_not_missing_hours_or_columns():
    cols = om.column_groups(("temperature_2m",))[0]
    ok = hourly_reply("2024-03-01", 2, {c: (lambda i: None if i == 5 else 30.0) for c in cols})
    assert om.previous_reply_ok(ok, date(2024, 3, 1), date(2024, 3, 2), cols)  # a null (archive gap) is fine
    assert not om.previous_reply_ok(ok, date(2024, 3, 1), date(2024, 3, 3), cols)  # hours missing
    assert not om.previous_reply_ok(ok, date(2024, 3, 1), date(2024, 3, 2), cols + ["relative_humidity_2m_previous_day1"])


def test_rate_limits_wait_the_daily_limit_stops_and_refusals_raise():
    slept = []
    s = FakeSession([FakeResponse(429, {"reason": "Minutely API request limit exceeded."}), FakeResponse(500, {}),
                     FakeResponse(200, {"ok": 1})])
    assert om.get_json(s, "u", {}, sleep=slept.append) == {"ok": 1} and slept == [60, 15]
    with pytest.raises(om.DailyLimit):
        om.get_json(FakeSession([FakeResponse(429, {"reason": "Daily API request limit exceeded."})]), "u", {},
                    sleep=slept.append)
    with pytest.raises(ValueError, match="not available"):
        om.get_json(FakeSession([FakeResponse(400, {"reason": "The requested model run is not available."})]), "u", {},
                    sleep=slept.append)


def test_a_cached_reply_is_never_fetched_again(tmp_path, monkeypatch):
    cols = om.column_groups(("temperature_2m",))[0]
    req = ("ecmwf_ifs025", 1, date(2024, 3, 1), date(2024, 3, 2), cols)
    monkeypatch.setattr(om, "previous_plan", lambda pulls=None: [req])
    reply = hourly_reply("2024-03-01", 2, {c: (lambda i: 30.0) for c in cols})
    s = FakeSession([FakeResponse(200, reply)])
    assert om.download_previous(s, root=tmp_path, pause=0, sleep=lambda x: None) == 0 and len(s.calls) == 1
    assert s.calls[0]["cell_selection"] == "nearest" and s.calls[0]["wind_speed_unit"] == "ms"
    assert om.download_previous(FakeSession([]), root=tmp_path, pause=0, sleep=lambda x: None) == 0  # no call made


def test_cached_groups_are_joined_per_hour_and_split_by_lead(tmp_path):
    g1, g2 = om.column_groups(("temperature_2m", "relative_humidity_2m", "surface_pressure"))[:2]
    d = tmp_path / "m" / "cell_1"
    d.mkdir(parents=True)
    (d / "2024-03_a.json").write_text(json.dumps(hourly_reply("2024-03-01", 1, {c: (lambda i, c=c: float(c[-1])) for c in g1})))
    (d / "2024-03_b.json").write_text(json.dumps(hourly_reply("2024-03-01", 1, {c: (lambda i: 950.0) for c in g2})))
    by_lead, where = om.load_previous("m", ("temperature_2m", "relative_humidity_2m", "surface_pressure"), tmp_path)
    assert sorted(by_lead) == [1, 2, 3, 4, 5] and where == {1: {"latitude": 28.25, "longitude": 77.0, "elevation": 274.0}}
    f = by_lead[3]
    assert len(f) == 24 and (f["temperature_2m"] == 3.0).all() and (f["surface_pressure"] == 950.0).all()
    (d / "2024-04_a.json").write_text(json.dumps(hourly_reply("2024-04-01", 1, {c: (lambda i: 1.0) for c in g1},
                                                              spot=(28.5, 77.0, 274.0))))
    with pytest.raises(ValueError, match="grid point"):
        om.load_previous("m", ("temperature_2m",), tmp_path)


def test_single_runs_and_unavailable_runs_are_read_back(tmp_path):
    d = tmp_path / om.SINGLE_DIR
    d.mkdir()
    loc = hourly_reply("2024-03-15", 1, {v: (lambda i: 1.0) for v in om.WBGT_VARS})
    (d / "2024-03-14.json").write_text(json.dumps([loc] * 9))
    (d / "2024-03-15.json").write_text(json.dumps({"unavailable": "not in the archive"}))
    df, where, unavailable = om.load_single(tmp_path)
    assert len(df) == 9 * 24 and sorted(where) == list(range(1, 10)) and unavailable == ["2024-03-15"]
    assert om.single_reply_ok({"unavailable": "x"}, date(2024, 3, 15))
