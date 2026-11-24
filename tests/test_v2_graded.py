"""v2/work_rest.py, v2/shade_wbgt.py, v2/graded.py (PROTOCOL_V2 amendment A1)."""
import numpy as np
import pandas as pd

from pipeline.wbgt_liljegren import globe_temperature, natural_wet_bulb, wind_2m
from v2.graded import audit_graded, h_a1, rest_hour_table, wcoverage
from v2.insurance_triggers import Trigger
from v2.shade_wbgt import shade_wbgt
from v2.work_rest import rest_share


def test_rest_share_matches_acgih_rows():
    # moderate TLV: 28.0 / 29.0 / 30.0 / 31.5 ("at or below" allows the row)
    assert rest_share([28.0, 28.01, 29.0, 30.0, 31.5, 31.51]).tolist() == [0, .25, .25, .5, .75, 1.0]
    # heavy TLV has no continuous-work row: rest >= 25 % even when cool
    assert rest_share([20.0], "heavy").tolist() == [0.25]
    # very heavy Action Limit: 24.5 / 27.0
    assert rest_share([24.5, 26.0, 27.5], "very_heavy", "AL").tolist() == [0.5, 0.75, 1.0]


def _met(n=4, ghi=800.0, dif=200.0, cosz=0.8):
    return pd.DataFrame({"t": [38.0] * n, "rh": [30.0] * n, "pressure": [980.0] * n, "wind": [2.5] * n,
                         "ghi": [ghi] * n, "diffuse": [dif] * n, "cosz": [cosz] * n})


def test_shade_below_sun_by_day_and_uses_frozen_components():
    from pipeline.wbgt_liljegren import wbgt
    d = _met()
    sun = wbgt(d["t"], d["rh"], d["pressure"], d["wind"], d["ghi"], d["ghi"] - d["diffuse"], d["cosz"])["wbgt"]
    shade = shade_wbgt(d)
    assert np.all(shade < sun) and np.all(sun - shade < 6.0)
    # at night (no sun) shade = 0.7 Tnwb + 0.3 Tg from the frozen components
    n = _met(ghi=0.0, dif=0.0, cosz=0.0)
    tk, rh = n["t"].to_numpy() + 273.15, n["rh"].to_numpy() / 100
    u2 = wind_2m(n["wind"].to_numpy(), 10.0, 0.0, 0.0)
    ref = 0.7 * (natural_wet_bulb(tk, rh, 980.0, u2, 0.0, 0.0, 0.0) - 273.15) + 0.3 * (globe_temperature(tk, rh, 980.0, u2, 0.0, 0.0, 0.0) - 273.15)
    assert np.allclose(shade_wbgt(n), ref)


def test_weighted_coverage_and_h_a1_on_synthetic_morning_peak():
    rows = []
    for y in range(1980, 2019):
        for d in pd.date_range(f"{y}-05-01", f"{y}-05-31"):
            for h in range(24):
                rows.append((d + pd.Timedelta(hours=h), 31.0 if h in (10, 11, 12, 13) else (28.5 if h in (14, 15) else 20.0)))
    hm = pd.DataFrame(rows, columns=["time", "wbgt"])
    alert = pd.date_range("1980-05-01", "2018-05-31")
    alert = alert[alert.month == 5]
    u = rest_hour_table(hm, alert, "moderate", "TLV")
    # per day: 4 h x 0.75 + 2 h x 0.25 = 3.5 rest-hours; 12-4 pm covers 13 (0.75) + 14, 15 (0.25 each) = 1.25
    assert abs(wcoverage(u, (12, 16)) - 1.25 / 3.5) < 1e-12
    r = h_a1(hm, alert, "moderate", "TLV")
    assert r["best_past_window"] == "9-13" and abs(r["rest_hours_per_alert_day"] - 3.5) < 1e-9
    assert r["diff_best_minus_W1"]["ci_low"] > 0


def test_audit_graded_counts_rest_on_unpaid_days():
    dates = pd.date_range("2000-05-01", "2000-05-04")
    dr = pd.DataFrame({"cell": 1, "date": dates, "tmax": [46.0, 40.0, 40.0, 40.0], "rest_hours": [2.0, 3.0, 3.0, 2.0]})
    trig = Trigger("x", "delhi", 45.27, "consecutive", 1, "05-01", "07-31", "test")
    r = audit_graded(dr, trig)
    assert abs(r["share_rest_hours_on_days_not_counted"] - 8 / 10) < 1e-12 and r["share_seasons_paid"] == 1.0
