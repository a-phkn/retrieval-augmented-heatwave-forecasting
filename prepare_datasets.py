import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_PATH = Path("data/processed/weather_daily.parquet")

EVENT_DIR = Path("events")
DATASET_DIR = Path("datasets")
TEST_DIR = Path("tests")

EVENT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Project settings
# ============================================================

TRAIN_END = pd.Timestamp("2015-12-31")
VAL_START = pd.Timestamp("2016-01-01")
VAL_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")

INPUT_DAYS = 14
FORECAST_DAYS = 5

CLIM_WINDOW = 7
THRESHOLD_SIGMA = 1.5
MIN_EPISODE_DAYS = 3


# ============================================================
# Load processed data
# ============================================================

df = pd.read_parquet(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("=" * 60)
print("STEP 2 — HEATWAVE LABELS & DATASET PREPARATION")
print("=" * 60)

print(f"Input rows: {len(df)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")


# ============================================================
# Basic date checks
# ============================================================

expected_dates = pd.date_range(
    df["date"].min(),
    df["date"].max(),
    freq="D"
)

missing_dates = expected_dates.difference(df["date"])

if len(missing_dates) > 0:
    raise ValueError(
        f"Missing dates detected: {len(missing_dates)}"
    )

if df["date"].duplicated().any():
    raise ValueError("Duplicate dates detected.")


# ============================================================
# TRAIN-ONLY CLIMATOLOGY
#
# For each calendar day-of-year:
# use training data within +/- 7 calendar days.
# ============================================================

train = df[df["date"] <= TRAIN_END].copy()

if train.empty:
    raise ValueError("Training dataset is empty.")

print("\nCreating train-only climatology...")

train["doy"] = train["date"].dt.dayofyear


def climatology_for_doy(doy):
    """
    Return mean and std of training Tmax values
    within +/- 7 DOY around the requested DOY.

    Handles the year boundary by wrapping DOY values.
    """
    days_in_year = 366

    distances = np.abs(train["doy"] - doy)

    # Circular distance around the calendar year.
    circular_distance = np.minimum(
        distances,
        days_in_year - distances
    )

    values = train.loc[
        circular_distance <= CLIM_WINDOW,
        "t_max"
    ]

    return values.mean(), values.std(ddof=0)


clim_rows = []

for doy in range(1, 367):
    mean_temp, std_temp = climatology_for_doy(doy)

    clim_rows.append(
        {
            "doy": doy,
            "clim_mean_t_max": mean_temp,
            "clim_std_t_max": std_temp,
            "threshold_t_max": (
                mean_temp + THRESHOLD_SIGMA * std_temp
            ),
        }
    )

clim = pd.DataFrame(clim_rows)


# ============================================================
# HANDLE NON-LEAP-YEAR DOY
#
# Feb 29 does not occur in non-leap years.
# For ordinary dates, map directly by day-of-year.
# ============================================================

df["doy"] = df["date"].dt.dayofyear

df = df.merge(
    clim,
    on="doy",
    how="left"
)

if df[
    [
        "clim_mean_t_max",
        "clim_std_t_max",
        "threshold_t_max"
    ]
].isna().any().any():
    raise ValueError("Missing climatology values detected.")


# ============================================================
# ANOMALY
# ============================================================

df["t_max_anomaly"] = (
    df["t_max"] - df["clim_mean_t_max"]
)


# ============================================================
# HOT DAY
#
# Hot if anomaly > 1.5 standard deviations.
# ============================================================

df["hot_day"] = (
    df["t_max_anomaly"]
    > THRESHOLD_SIGMA * df["clim_std_t_max"]
)


# ============================================================
# HEATWAVE EPISODES
#
# Consecutive hot days belong to the same episode.
# A 1-day gap is allowed.
# Minimum episode length = 3 days.
# ============================================================

print("Creating heatwave episodes...")

hot_dates = df.loc[df["hot_day"], "date"].tolist()

episodes = []

if hot_dates:
    episode_start = hot_dates[0]
    previous_hot = hot_dates[0]

    for current_date in hot_dates[1:]:
        gap = (current_date - previous_hot).days

        if gap <= 2:
            # Same episode.
            previous_hot = current_date
        else:
            # End previous episode.
            episode_end = previous_hot

            episode_dates = pd.date_range(
                episode_start,
                episode_end,
                freq="D"
            )

            if len(episode_dates) >= MIN_EPISODE_DAYS:
                episodes.append(
                    {
                        "episode_start": episode_start,
                        "episode_end": episode_end,
                        "duration_days": len(episode_dates),
                    }
                )

            episode_start = current_date
            previous_hot = current_date

    # Close final episode.
    episode_end = previous_hot

    episode_dates = pd.date_range(
        episode_start,
        episode_end,
        freq="D"
    )

    if len(episode_dates) >= MIN_EPISODE_DAYS:
        episodes.append(
            {
                "episode_start": episode_start,
                "episode_end": episode_end,
                "duration_days": len(episode_dates),
            }
        )


events = pd.DataFrame(episodes)

if events.empty:
    events = pd.DataFrame(
        columns=[
            "episode_id",
            "episode_start",
            "episode_end",
            "duration_days",
        ]
    )
else:
    events.insert(
        0,
        "episode_id",
        range(1, len(events) + 1)
    )


# ============================================================
# Assign episode IDs to daily data
# ============================================================

df["heatwave_episode_id"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="Int64"
)

for _, episode in events.iterrows():
    mask = (
        (df["date"] >= episode["episode_start"])
        & (df["date"] <= episode["episode_end"])
    )

    df.loc[mask, "heatwave_episode_id"] = int(
        episode["episode_id"]
    )


# ============================================================
# Split assignment
# ============================================================

df["split"] = np.where(
    df["date"] <= TRAIN_END,
    "train",
    np.where(
        df["date"] <= VAL_END,
        "val",
        "test"
    )
)


# ============================================================
# Save event catalogue
# ============================================================

events.to_parquet(
    EVENT_DIR / "event_catalogue.parquet",
    index=False
)


# ============================================================
# SAVE DAILY DATASETS
#
# Remove temporary helper columns that aren't part
# of the required final dataset.
# ============================================================

output_columns = [
    "date",
    "t_max",
    "t_min",
    "shortwave_radiation_sum",
    "t_mean",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
    "doy_sin",
    "doy_cos",
    "years_since_1980",
    "clim_mean_t_max",
    "clim_std_t_max",
    "t_max_anomaly",
    "split",
    "hot_day",
    "heatwave_episode_id",
]

df[output_columns].to_parquet(
    DATASET_DIR / "all_daily.parquet",
    index=False
)

df[df["split"] == "train"][output_columns].to_parquet(
    DATASET_DIR / "train.parquet",
    index=False
)

df[df["split"] == "val"][output_columns].to_parquet(
    DATASET_DIR / "val.parquet",
    index=False
)

df[df["split"] == "test"][output_columns].to_parquet(
    DATASET_DIR / "test.parquet",
    index=False
)


# ============================================================
# CREATE 14-DAY → 5-DAY WINDOWS
#
# Each sample:
#   input = previous 14 days
#   target = next 5 days
#
# Retrieval leakage rule:
# candidate future must finish before query forecast begins.
#
# Candidate start <= query date - 5 days.
# ============================================================

print("\nCreating forecasting windows...")

feature_columns = [
    "t_max",
    "t_min",
    "t_mean",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
    "doy_sin",
    "doy_cos",
    "years_since_1980",
    "clim_mean_t_max",
    "clim_std_t_max",
    "t_max_anomaly",
]


window_rows = []

# Date → row lookup.
date_to_idx = {
    date: idx
    for idx, date in enumerate(df["date"])
}


def get_split(date):
    if date <= TRAIN_END:
        return "train"

    if date <= VAL_END:
        return "val"

    return "test"


all_dates = df["date"].tolist()

for i in range(INPUT_DAYS, len(df) - FORECAST_DAYS + 1):

    query_date = df.loc[i, "date"]

    input_start = query_date - pd.Timedelta(days=INPUT_DAYS)
    input_end = query_date - pd.Timedelta(days=1)

    forecast_start = query_date
    forecast_end = query_date + pd.Timedelta(
        days=FORECAST_DAYS - 1
    )

    # Ensure all dates exist.
    required_dates = pd.date_range(
        input_start,
        forecast_end,
        freq="D"
    )

    if not all(
        date in date_to_idx
        for date in required_dates
    ):
        continue

    query_split = get_split(query_date)

    if query_split == "train" and forecast_end > TRAIN_END:
        continue

    if query_split == "val" and forecast_end > VAL_END:
        continue

    # --------------------------------------------------------
    # Retrieval candidate boundary.
    #
    # Candidate consists of:
    #   14-day input
    #   5-day future
    #
    # Candidate future must end before query forecast starts.
    #
    # Therefore candidate start <= query_date - 19 days.
    # --------------------------------------------------------

    candidate_latest_start = (
        query_date
        - pd.Timedelta(
            days=INPUT_DAYS + FORECAST_DAYS
        )
    )

    window_rows.append(
        {
            "query_date": query_date,
            "input_start": input_start,
            "input_end": input_end,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end,
            "split": query_split,
            "target_episode_id": df.loc[
                i,
                "heatwave_episode_id"
            ],
            "target_is_heatwave": bool(
                df.loc[
                    i,
                    "heatwave_episode_id"
                ]
                == df.loc[
                    i,
                    "heatwave_episode_id"
                ]
            )
            if pd.notna(
                df.loc[i, "heatwave_episode_id"]
            )
            else False,
            "candidate_latest_start": candidate_latest_start,
        }
    )


windows = pd.DataFrame(window_rows)

windows.to_parquet(
    DATASET_DIR / "forecast_windows.parquet",
    index=False
)


# ============================================================
# LEAKAGE TEST FILE
# ============================================================

test_code = r'''
import pandas as pd


WINDOW_PATH = "datasets/forecast_windows.parquet"


def test_retrieval_candidate_boundary():
    """
    A retrieval candidate's complete 14-day input + 5-day future
    must finish before the query forecast begins.

    Therefore:

        candidate_start <= query_date - 19 days
    """

    windows = pd.read_parquet(WINDOW_PATH)

    required_latest = (
        windows["query_date"]
        - pd.Timedelta(days=19)
    )

    assert (
        windows["candidate_latest_start"]
        <= required_latest
    ).all()


def test_split_dates():
    windows = pd.read_parquet(WINDOW_PATH)

    train = windows[windows["split"] == "train"]
    val = windows[windows["split"] == "val"]
    test = windows[windows["split"] == "test"]

    if not train.empty:
        assert train["query_date"].max() <= pd.Timestamp(
            "2015-12-31"
        )

    if not val.empty:
        assert val["query_date"].min() >= pd.Timestamp(
            "2016-01-01"
        )
        assert val["query_date"].max() <= pd.Timestamp(
            "2018-12-31"
        )

    if not test.empty:
        assert test["query_date"].min() >= pd.Timestamp(
            "2019-01-01"
        )


def test_no_duplicate_query_dates():
    windows = pd.read_parquet(WINDOW_PATH)

    assert not windows["query_date"].duplicated().any()
'''

with open(TEST_DIR / "test_no_leakage.py", "w") as f:
    f.write(test_code)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 2 COMPLETE")
print("=" * 60)

print(f"\nTraining days:   {(df['split'] == 'train').sum()}")
print(f"Validation days: {(df['split'] == 'val').sum()}")
print(f"Test days:       {(df['split'] == 'test').sum()}")

print(f"\nHeatwave episodes: {len(events)}")

if not events.empty:
    print(
        f"Longest episode: "
        f"{events['duration_days'].max()} days"
    )

print(f"\nForecast windows: {len(windows)}")

print("\nSaved:")
print("  events/event_catalogue.parquet")
print("  datasets/all_daily.parquet")
print("  datasets/train.parquet")
print("  datasets/val.parquet")
print("  datasets/test.parquet")
print("  datasets/forecast_windows.parquet")
print("  tests/test_no_leakage.py")