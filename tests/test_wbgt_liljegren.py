"""pipeline/wbgt_liljegren.py: port of Liljegren's WBGT (Argonne C code), adapted to
reanalysis following Kong & Huber (2022).

Three independent checks:
1. Our bracketed solver against a literal transcription of Liljegren's original damped
   fixed-point iteration (wbgt.c.original, Argonne licence -- see the module docstring).
2. Reference outputs from Kong & Huber's PyWBGT implementation (commit b40942b) on real
   Delhi hours (inputs from our ERA5 data; only the resulting numbers are stored here).
3. Solar geometry and basic physics.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import wbgt_liljegren as W

DELHI = (28.5, 77.25)


# ---------------------------------------------------------------- 1. original algorithm

def _tg_original(tk, rh, p, u, solar, fdir, cza):
    """Tglobe() from wbgt.c.original, line by line (damped iteration, 0.02 K convergence)."""
    tsfc = prev = tk
    for _ in range(50):
        tref = 0.5 * (prev + tk)
        h = W.h_sphere(tref, p, u)
        new = (0.5 * (W.emis_atm(tk, rh) * tk**4 + W.EMIS_SFC * tsfc**4) - h / (W.STEFANB * W.EMIS_GLOBE) * (prev - tk)
               + solar / (2 * W.STEFANB * W.EMIS_GLOBE) * (1 - W.ALB_GLOBE) * (fdir * (1 / (2 * cza) - 1) + 1 + W.ALB_SFC)) ** 0.25
        if abs(new - prev) < 0.02:
            return new
        prev = 0.9 * prev + 0.1 * new
    return np.nan


def _twb_original(tk, rh, p, u, solar, fdir, cza, rad=1):
    """Twb() from wbgt.c.original, line by line."""
    tsfc, sza = tk, np.arccos(cza)
    eair = rh * W.esat(tk)
    prev = W.dew_point(eair)
    for _ in range(50):
        tref = 0.5 * (prev + tk)
        h = W.h_cylinder(tref, p, u)
        fatm = (W.STEFANB * W.EMIS_WICK * (0.5 * (W.emis_atm(tk, rh) * tk**4 + W.EMIS_SFC * tsfc**4) - prev**4)
                + (1 - W.ALB_WICK) * solar * ((1 - fdir) * (1 + 0.25 * W.D_WICK / W.L_WICK)
                                             + fdir * (np.tan(sza) / np.pi + 0.25 * W.D_WICK / W.L_WICK) + W.ALB_SFC))
        ewick = W.esat(prev)
        density = p * 100 / (W.R_AIR * tref)
        sc = W.viscosity(tref) / (density * W.diffusivity(tref, p))
        new = tk - W.evap(tref) / W.RATIO * (ewick - eair) / (p - ewick) * (W.PR / sc) ** 0.56 + fatm / h * rad
        if abs(new - prev) < 0.02:
            return new
        prev = 0.9 * prev + 0.1 * new
    return np.nan


@pytest.fixture(scope="module")
def random_inputs():
    rng = np.random.default_rng(1)
    n = 300
    cz = rng.uniform(0.05, 1, n)
    return dict(tk=rng.uniform(288, 320, n), rh=rng.uniform(0.08, 0.95, n), p=rng.uniform(975, 1000, n),
                u=rng.uniform(0.2, 6, n), solar=rng.uniform(0, 1000, n) * cz, fdir=rng.uniform(0, 0.9, n), cz=cz)


def test_solver_matches_the_original_liljegren_iteration(random_inputs):
    a = random_inputs
    args = (a["tk"], a["rh"], a["p"], a["u"], a["solar"], a["fdir"], a["cz"])
    tg = W.globe_temperature(*args)
    tw = W.natural_wet_bulb(*args)
    tg_o = np.array([_tg_original(*x) for x in zip(*args)])
    tw_o = np.array([_twb_original(*x) for x in zip(*args)])
    assert np.isfinite(tg).all() and np.isfinite(tw).all()
    # agreement within the original's own convergence tolerance (0.02 K)
    assert np.nanmax(np.abs(tg - tg_o)) < 0.025
    assert np.nanmax(np.abs(tw - tw_o)) < 0.025


# ---------------------------------------------------------------- 2. Kong & Huber reference values

# (case, T C, RH %, p hPa, 10 m wind m/s, GHI W/m2, direct horizontal W/m2, sunlit-mean cos zenith,
#  WBGT C from Kong & Huber's PyWBGT equations with the same inputs). Real hours, Delhi cell 5.
KONG_HUBER_REFERENCE = [
    ("hot_dry_noon", 42.0, 18, 972.9, 4.34, 843.0, 652.0, 0.9449, 31.319),  # 2003-05-19 14:00
    ("humid_monsoon_day", 26.3, 95, 966.3, 5.32, 105.0, 9.0, 0.9800, 26.343),  # 1997-08-02 13:00
    ("night", 31.6, 19, 970.0, 3.14, 0.0, 0.0, 0.0000, 20.356),  # 1995-06-06 02:00
    ("winter_morning", 11.4, 93, 989.4, 1.36, 248.0, 166.0, 0.2447, 14.640),  # 2023-01-22 09:00
    ("sunset_hour", 25.5, 28, 982.7, 2.39, 1.0, 0.0, 0.0754, 17.176),  # 1993-04-06 19:00
]


@pytest.mark.parametrize("case, t, rh, p, w10, ghi, direct, cosz, expected", KONG_HUBER_REFERENCE)
def test_matches_kong_huber_reference_outputs(case, t, rh, p, w10, ghi, direct, cosz, expected):
    out = W.wbgt([t], [rh], [p], [w10], [ghi], [direct], [cosz])
    # Kong & Huber use a pressure-dependent saturation-vapour enhancement and slightly
    # different fits for some air properties; on 3,000 real hours the max difference was 0.007 C.
    assert out["wbgt"][0] == pytest.approx(expected, abs=0.02), case


# ---------------------------------------------------------------- 3. solar geometry and physics

@pytest.mark.parametrize("day, dec", [("2010-06-21", 23.44), ("2010-12-21", -23.44)])
def test_noon_sun_height_at_solstices(day, dec):
    t = pd.date_range(f"{day} 05:30", f"{day} 08:00", freq="1min").to_numpy()  # UTC, around Delhi noon
    alt = W.solar_altitude_deg(t, *DELHI).max()
    assert alt == pytest.approx(90 - abs(DELHI[0] - dec), abs=0.3)


def test_sunlit_mean_cosz_is_zero_at_night_and_partial_at_sunrise():
    end_utc = np.array(["2010-06-21T20:00", "2010-06-21T01:00", "2010-06-21T07:00"], dtype="datetime64[ns]")
    cz = W.mean_sunlit_cosz(end_utc, *DELHI)  # 01:30 IST night; 05:30-06:30 IST sunrise; around noon
    assert cz[0] == 0.0
    assert 0 < cz[1] < 0.2 and cz[2] > 0.95


def test_wind_is_reduced_to_2m_and_floored():
    u2 = W.wind_2m(np.array([5.0, 5.0, 0.01]), 10.0, np.array([0.8, 0.0, 0.8]), np.array([800.0, 0.0, 800.0]))
    assert (u2[:2] < 5.0).all() and u2[2] == W.MIN_SPEED


def test_physics_sanity(random_inputs):
    a = random_inputs
    args = (a["tk"], a["rh"], a["p"], a["u"], a["solar"], a["fdir"], a["cz"])
    tw = W.natural_wet_bulb(*args)
    tpsy = W.natural_wet_bulb(*args, radiative=False)
    tg = W.globe_temperature(*args)
    assert (tpsy <= a["tk"] + 1e-3).all()  # no sun: wet bulb never above air temperature
    sunny = a["solar"] > 200
    assert (tw[sunny] >= tpsy[sunny] - 1e-3).all()  # sunshine warms the wick
    assert (tg[a["solar"] > 300] > a["tk"][a["solar"] > 300]).all()  # globe hotter than air in sun
    # more sun -> higher WBGT; more wind in sun -> lower WBGT
    base = dict(t_c=[38.0], rh_pct=[30.0], p_hpa=[975.0], cosz=[0.9])
    lo = W.wbgt(**base, wind10m=[2.0], ghi=[200.0], direct_horizontal=[100.0])["wbgt"][0]
    hi = W.wbgt(**base, wind10m=[2.0], ghi=[900.0], direct_horizontal=[700.0])["wbgt"][0]
    windy = W.wbgt(**base, wind10m=[8.0], ghi=[900.0], direct_horizontal=[700.0])["wbgt"][0]
    assert lo < hi and windy < hi
