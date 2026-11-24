"""v2/window_test.py (H-A)."""
import numpy as np
import pandas as pd

from v2.window_test import CANDIDATES, best_window, covers, coverage, h_a, jackknife, s2_alert_days


def test_window_convention_is_hour_ending():
    hrs = np.arange(24)
    assert hrs[covers((12, 16), hrs)].tolist() == [13, 14, 15, 16]
    assert CANDIDATES[0] == (5, 9) and CANDIDATES[-1] == (16, 20)


def test_alert_days_exclude_missing_and_flagged():
    st = pd.DataFrame({"date": ["1990-05-01", "1990-05-02", "1990-05-03", "1990-03-10"],
                       "tmax_c": [41.0, np.nan, 42.0, 41.0], "tmax_qflag": [None, None, "O", None]})
    days, missing = s2_alert_days(st)
    assert list(days.strftime("%Y-%m-%d")) == ["1990-05-01"]          # Mar 10 is out of season
    assert missing.loc[1990, "unclassified_days"] == 200 - 1           # 200 season days, 1 usable


def _synthetic(peak_hours, years=range(1980, 2019)):
    """Every May day is an alert day; unsafe hours at peak_hours."""
    times, w, st = [], [], []
    for y in years:
        for d in pd.date_range(f"{y}-05-01", f"{y}-05-31"):
            st.append((d, 41.0))
            for h in range(24):
                times.append(d + pd.Timedelta(hours=h))
                w.append(30.0 if h in peak_hours else 20.0)
    return (pd.DataFrame({"time": times, "wbgt": w}),
            pd.DataFrame({"date": [s[0] for s in st], "tmax_c": [s[1] for s in st], "tmax_qflag": None}))


def test_morning_peak_is_found_and_supported():
    hourly, station = _synthetic({10, 11, 12, 13})
    r = h_a(hourly, station, limit=28.0)
    assert r["best_past_window"] == "9-13" and r["supported"]
    assert abs(r["share_outside W1 12-4 pm (HAP p.155)"]["estimate"] - 0.75) < 1e-12


def test_no_difference_when_peak_is_inside_w1():
    hourly, station = _synthetic({13, 14, 15, 16})
    r = h_a(hourly, station, limit=28.0)
    assert r["best_past_window"] == "12-16" and not r["supported"]


def test_jackknife_matches_hand_formula():
    rng = np.random.default_rng(0)
    dates = pd.to_datetime([f"{y}-05-{d:02d}" for y in range(2000, 2010) for d in range(1, 11)])
    u = pd.DataFrame({"date": dates, "hour": rng.integers(8, 18, len(dates))})
    r = jackknife(u, lambda t: coverage(t, (12, 16)))
    reps = [coverage(u[u["date"].dt.year != y], (12, 16)) for y in range(2000, 2010)]
    se = np.sqrt(9 / 10 * np.sum((np.array(reps) - np.mean(reps)) ** 2))
    assert abs(r.se - se) < 1e-12 and r.n_clusters == 10


def _station(values: dict[str, float]):
    d = pd.date_range("1981-01-01", "2010-12-31")
    st = pd.DataFrame({"date": d, "tmax_c": 30.0, "tmax_qflag": None})
    for k, v in values.items():
        st.loc[st["date"] == pd.Timestamp(k), "tmax_c"] = v
    return st


def test_s1_needs_two_consecutive_days_and_departure():
    from v2.window_test import s1_alert_days
    # flat normal 30 C (a few hot days barely move the 30-year mean)
    st = _station({"1995-05-10": 41.0, "1995-05-11": 41.0, "1995-06-01": 46.0, "1995-06-20": 41.0})
    days, _ = s1_alert_days(st)
    assert [d.strftime("%m-%d") for d in days if d.year == 1995] == ["05-11"]  # declared on 2nd day only


def test_s1_unknown_previous_day_is_unclassified():
    from v2.window_test import s1_alert_days
    st = _station({"1995-05-10": np.nan, "1995-05-11": 41.0})
    days, missing = s1_alert_days(st)
    assert pd.Timestamp("1995-05-11") not in days and missing.loc[1995, "unclassified_days"] == 1  # May 10 itself is "no": May 9 is known non-criterion


def test_s3_departure_and_era5_proxy_season():
    from v2.window_test import era5_proxy_alert_days, s3_alert_days
    st = _station({"1995-05-10": 34.5, "1995-05-11": 33.0})
    days, _ = s3_alert_days(st)
    assert [d.strftime("%m-%d") for d in days if d.year == 1995] == ["05-10"]
    t = pd.Series(30.0, index=pd.date_range("1980-01-01", "2018-12-31"))
    t[pd.Timestamp("2000-05-10")] = 41.0
    t[pd.Timestamp("2000-08-10")] = 46.0          # outside the proxy's Mar 15-Jul 31 season
    days, missing = era5_proxy_alert_days(t)
    assert list(days.strftime("%Y-%m-%d")) == ["2000-05-10"] and missing["unclassified_days"].sum() == 0
