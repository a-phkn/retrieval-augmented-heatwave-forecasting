"""v2/calibration_v2.py (H-C, E5, Holm)."""
import numpy as np
import pandas as pd

from v2.calibration_v2 import conformal_offsets, e5, h_c, holm


def _tables(seed_count=2, noise_hot=1.0, years_val=range(2007, 2019), years_inner=range(2003, 2007)):
    rng = np.random.default_rng(0)
    def mk(years):
        rows = []
        for s in range(seed_count):
            for y in years:
                for d in pd.date_range(f"{y}-03-15", f"{y}-09-30"):
                    truth = 28 + 4 * np.sin((d.dayofyear - 80) / 365 * 2 * np.pi) + rng.normal(0, 1)
                    pred = truth + rng.normal(0, 1.0)
                    rows.append((s, 1, d, pred, truth))
        return pd.DataFrame(rows, columns=["seed", "lead", "target_date", "pred", "actual"])
    return mk(years_inner), mk(years_val)


def test_conformal_offsets_finite_sample():
    r = np.arange(1, 20, dtype=float)   # n = 19
    lo, hi = conformal_offsets(r, 0.8)   # ceil(20*0.9)=18th, floor(20*0.1)=2nd
    assert (lo, hi) == (2.0, 18.0)


def test_h_c_near_nominal_when_exchangeable():
    inner, val = _tables()
    r = h_c([(inner, val)])
    assert 0.74 < r["all_days"]["estimate"] < 0.86 and r["all_days"]["n_clusters"] >= 12
    assert 0 <= r["p_tost"] <= 1


def test_e5_miss_rate_controlled_and_holm():
    inner, val = _tables()
    days = pd.date_range("2003-01-01", "2018-12-31")
    # event: truth (seed 0 actual) above 30
    truth = pd.concat([inner, val]).query("seed == 0").set_index("target_date")["actual"]
    event = (truth > 30).reindex(days, fill_value=False)
    r = e5([(inner, val)], event)
    assert r["miss_rate"]["estimate"] <= 0.30 and r["event_days"] > 100
    assert 0 <= r["false_alarm_rate"] <= 1 and abs(r["peirce_skill"] - (1 - r["miss_rate"]["estimate"] - r["false_alarm_rate"])) < 1e-12
    assert 0 < r["base_rate"] < 1 and r["lift"] > 1.0 and abs(r["precision"] - r["lift"] * r["base_rate"]) < 1e-12
    sev = (truth > 31).reindex(days, fill_value=False)
    r2 = e5([(inner, val)], event, severe=sev)
    assert 0 < r2["severe_event_days"] < r2["event_days"]
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"]["p_holm"] == 0.03 and h["c"]["p_holm"] == 0.06 and h["b"]["p_holm"] == 0.06
    assert h["a"]["reject"] and not h["b"]["reject"]


def test_e5_baselines_scores():
    from v2.e5_baselines import scores
    idx = pd.date_range("1980-01-01", "2018-12-31")
    rest = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    p = scores(rest, "persistence", 2)
    assert p.iloc[5] == rest.iloc[3] and np.isnan(p.iloc[0])
    c = scores(rest, "calendar", 1)
    assert c.notna().all() and c.index.equals(rest.index)
