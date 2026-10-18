"""pipeline/download_era5_z500.py: time axis, chunk plan and box selection (no network)."""
import numpy as np
import pandas as pd

import pipeline.download_era5_z500 as dz
from training.folds import TEST_START


def test_time_index_points_at_12_utc_on_the_6_hourly_axis():
    idx = dz.time_index(pd.DatetimeIndex(["1959-01-01", "1980-01-01"]))
    assert idx.tolist() == [2, 7670 * 4 + 2]  # 7670 days from 1959-01-01 to 1980-01-01


def test_chunk_plan_groups_dates_by_chunk_with_output_rows():
    plan = dz.chunk_plan(np.array([38, 42, 79, 80]), t_chunk=40)
    assert set(plan) == {0, 1, 2}
    assert plan[0][0].tolist() == [38] and plan[1][0].tolist() == [2, 39] and plan[1][1].tolist() == [1, 2]
    assert plan[2][0].tolist() == [0] and plan[2][1].tolist() == [3]


def test_box_on_the_real_2p8_degree_grid_and_test_period_never_read():
    lat = -88.59375 + 2.8125 * np.arange(64)
    lon = 2.8125 * np.arange(128)
    ilat, ilon = dz.box_indices(lat, lon)
    assert len(ilat) == 12 and len(ilon) == 21
    assert lat[ilat][0] >= 10 and lat[ilat][-1] <= 45 and lon[ilon][0] >= 40 and lon[ilon][-1] <= 100
    assert dz.END < TEST_START
    assert len(dz.columns(lat[ilat], lon[ilon])) == 252
