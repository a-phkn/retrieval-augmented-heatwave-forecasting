"""v2/exposure.py, v2/insurance_triggers.py (H-E) and v2/risk_control.py (E5)."""
import numpy as np
import pandas as pd

from v2.exposure import WORK_HOURS, daily_exposure, longest_run
from v2.insurance_triggers import Trigger, audit, mark_trigger
from v2.risk_control import crc_threshold, evaluate


def _hourly(day: str, wbgt_by_hour: dict, t=35.0, cell=1):
    times = pd.date_range(day, periods=24, freq="h")
    w = [wbgt_by_hour.get(h, 20.0) for h in range(24)]
    return pd.DataFrame({"time": times, "t": t, "wbgt": w, "cell": cell})


def test_unsafe_day_needs_two_consecutive_working_hours():
    two = _hourly("2010-05-01", {13: 29.0, 14: 29.0})
    apart = _hourly("2010-05-02", {11: 29.0, 13: 29.0})
    night = _hourly("2010-05-03", {2: 35.0, 3: 35.0})          # outside 06-20
    edge = _hourly("2010-05-04", {20: 29.0, 21: 29.0})         # 21 is outside the working hours
    d = daily_exposure(pd.concat([two, apart, night, edge]), limit=28.0).set_index("date")
    assert d["unsafe_day"].tolist() == [True, False, False, False]
    assert d["unsafe_hours"].tolist() == [2, 2, 0, 1]
    assert longest_run(np.array([1, 1, 0, 1, 1, 1], bool)) == 3 and len(WORK_HOURS) == 15


def _daily(tmax, start="2010-04-01", unsafe=None, cell=1):
    dates = pd.date_range(start, periods=len(tmax), freq="D")
    u = unsafe if unsafe is not None else [False] * len(tmax)
    return pd.DataFrame({"cell": cell, "date": dates, "tmax": tmax, "unsafe_day": u,
                         "unsafe_hours": [3 if x else 0 for x in u]})


def test_consecutive_rule_and_strict_threshold():
    trig = Trigger("t", "x", 43.6, "consecutive", 2, "04-01", "07-31", "test")
    d = mark_trigger(_daily([44, 44, 43.6, 44, 43, 44, 44, 44]), trig)
    assert d["counts"].tolist() == [True, True, False, False, False, True, True, True]
    assert d["season_paid"].all()


def test_season_count_rule_pays_only_at_n_days():
    trig = Trigger("t", "x", 43.2, "season_count", 3, "04-01", "07-31", "test")
    assert not mark_trigger(_daily([44, 40, 44, 40]), trig)["counts"].any()
    d = mark_trigger(_daily([44, 40, 44, 40, 44]), trig)
    assert d["counts"].tolist() == [True, False, True, False, True]


def test_risk_period_and_calendar_gaps():
    trig = Trigger("t", "x", 40.0, "consecutive", 2, "04-01", "07-31", "test")
    d = _daily([45, 45, 45], start="2010-03-31")                 # first day outside the period
    assert mark_trigger(d, trig)["counts"].tolist() == [False, True, True]
    gap = pd.concat([_daily([45], start="2010-05-01"), _daily([45], start="2010-05-03")])
    assert not mark_trigger(gap, trig)["counts"].any()           # not consecutive in the calendar


def test_audit_counts_missed_unsafe_days():
    trig = Trigger("t", "x", 43.6, "consecutive", 2, "04-01", "07-31", "test")
    d = _daily([44, 44, 40, 41, 42, 39], unsafe=[True, False, True, True, False, False])
    a = audit(d, trig)
    assert a["unsafe_days"] == 3 and abs(a["share_unsafe_days_not_counted"] - 2 / 3) < 1e-12
    assert a["unsafe_hours_on_missed_days"] == 6 and abs(a["share_counted_days_not_unsafe"] - 0.5) < 1e-12


def test_crc_threshold_bound_and_coverage():
    rng = np.random.default_rng(0)
    alpha, misses = 0.2, []
    for _ in range(400):
        y = rng.random(600) < 0.3
        s = rng.normal(np.where(y, 1.0, 0.0), 1.0)
        lam = crc_threshold(s[:300], y[:300], alpha)
        misses.append(evaluate(s[300:], y[300:], lam)["miss_rate"])
        k = int((s[:300][y[:300]] < lam).sum())
        assert (k + 1) / (y[:300].sum() + 1) <= alpha + 1e-12
    assert np.mean(misses) <= alpha + 0.01                        # guarantee holds on average


def test_crc_edge_cases():
    assert crc_threshold(np.array([1.0, 2.0]), np.array([True, True]), 0.2) == -np.inf   # 1/3 > 0.2
    assert crc_threshold(np.array([1.0]), np.array([False]), 0.2) == -np.inf             # no unsafe days


def test_vectorised_runs_match_the_loop():
    rng = np.random.default_rng(1)
    parts = [_hourly(f"2011-05-{d:02d}", {h: float(rng.choice([20, 30])) for h in range(24)}, cell=c)
             for d in range(1, 21) for c in (1, 2)]
    hourly = pd.concat(parts, ignore_index=True)
    d = daily_exposure(hourly, limit=28.0)
    for _, row in d.iterrows():
        x = hourly[(hourly["cell"] == row["cell"]) & (hourly["time"].dt.normalize() == row["date"])]
        x = x[x["time"].dt.hour.isin(WORK_HOURS)].sort_values("time")
        assert row["longest_unsafe_run"] == longest_run((x["wbgt"] >= 28.0).to_numpy())
