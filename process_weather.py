import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

RAW_DIR = Path("data/raw/era5_monthly")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# The 9 ERA5 grid cells
CELLS = [
    (28.295254, 76.93878),
    (28.365553, 77.23042),
    (28.295254, 77.44898),
    (28.576448, 76.98177),
    (28.576448, 77.18678),
    (28.576448, 77.4943),
    (28.857643, 76.9222),
    (28.857643, 77.231125),
    (28.857643, 77.43707),
]


WEATHER_COLUMNS = [
    "t_max",
    "t_min",
    "shortwave_radiation_sum",
    "t_mean",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
]


# ============================================================
# READ ONE MONTHLY JSON FILE
# ============================================================

def read_monthly_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    # ----------------------------
    # Daily variables
    # ----------------------------

    daily = pd.DataFrame(data["daily"])

    daily["date"] = pd.to_datetime(daily["time"])

    daily = daily[
        [
            "date",
            "temperature_2m_max",
            "temperature_2m_min",
            "shortwave_radiation_sum",
        ]
    ].rename(
        columns={
            "temperature_2m_max": "t_max",
            "temperature_2m_min": "t_min",
        }
    )

    # ----------------------------
    # Hourly variables
    # ----------------------------

    hourly = pd.DataFrame(data["hourly"])

    hourly["date"] = pd.to_datetime(
        hourly["time"]
    ).dt.date

    hourly["date"] = pd.to_datetime(
        hourly["date"]
    )

    # Convert hourly data to daily means
    hourly_daily = (
        hourly.groupby("date")
        .agg(
            t_mean=("temperature_2m", "mean"),
            relative_humidity_mean=(
                "relative_humidity_2m",
                "mean",
            ),
            wind_speed_mean=(
                "wind_speed_10m",
                "mean",
            ),
            surface_pressure_mean=(
                "surface_pressure",
                "mean",
            ),
        )
        .reset_index()
    )

    # Combine daily and hourly-derived variables
    result = daily.merge(
        hourly_daily,
        on="date",
        how="left",
    )

    return result


# ============================================================
# READ ALL MONTHS FOR ONE CELL
# ============================================================

def read_cell(cell_number):
    cell_dir = RAW_DIR / f"cell_{cell_number}"

    monthly_files = sorted(
        cell_dir.glob("*.json")
    )

    print(
        f"Cell {cell_number}: "
        f"{len(monthly_files)} monthly files"
    )

    all_months = []

    for file_path in monthly_files:
        monthly_data = read_monthly_file(file_path)
        all_months.append(monthly_data)

    cell_data = pd.concat(
        all_months,
        ignore_index=True,
    )

    cell_data = (
        cell_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    return cell_data


# ============================================================
# MAIN PROCESSING
# ============================================================

print("Starting weather data processing...\n")


# ------------------------------------------------------------
# 1. Read all 9 cells
# ------------------------------------------------------------

all_cells = []

for cell_number in range(1, 10):
    cell_data = read_cell(cell_number)

    cell_data["cell_number"] = cell_number

    all_cells.append(cell_data)


# ------------------------------------------------------------
# 2. Combine all cells
# ------------------------------------------------------------

all_cells_data = pd.concat(
    all_cells,
    ignore_index=True,
)

print(
    f"\nCombined rows: "
    f"{len(all_cells_data)}"
)


# ------------------------------------------------------------
# 3. Calculate 9-cell area average
# ------------------------------------------------------------

weather_daily = (
    all_cells_data
    .groupby("date")[WEATHER_COLUMNS]
    .mean()
    .reset_index()
)

print(
    f"Daily rows after 9-cell average: "
    f"{len(weather_daily)}"
)


# ------------------------------------------------------------
# 4. Remove incomplete current tail
# ------------------------------------------------------------

# Find the latest date that is available in all 9 cells.
# Find the latest date with complete weather data across all 9 cells.
common_dates = None

for cell_id in range(1, 10):
    cell_df = read_cell(cell_id)

    # Keep only dates where all required weather variables are present.
    required_cols = [
        "t_max",
        "t_min",
        "shortwave_radiation_sum",
        "t_mean",
        "relative_humidity_mean",
        "wind_speed_mean",
        "surface_pressure_mean",
    ]

    cell_df = cell_df.dropna(subset=required_cols)
    cell_dates = set(cell_df["date"])

    if common_dates is None:
        common_dates = cell_dates
    else:
        common_dates &= cell_dates

if not common_dates:
    raise ValueError("No common complete dates found across all 9 cells.")

last_complete_date = pd.Timestamp(max(common_dates))

print(
    f"Latest date with complete weather data across all 9 cells: "
    f"{last_complete_date.date()}"
)

print(
    f"Latest date complete across all 9 cells: "
    f"{last_complete_date.date()}"
)

print(
    f"Latest date complete across all 9 cells: "
    f"{last_complete_date.date()}"
)

weather_daily = weather_daily[
    weather_daily["date"] <= last_complete_date
].copy()

print(
    f"Final date range: "
    f"{weather_daily['date'].min().date()} "
    f"to "
    f"{weather_daily['date'].max().date()}"
)


# ------------------------------------------------------------
# 5. Add calendar features
# ------------------------------------------------------------

weather_daily["day_of_year"] = (
    weather_daily["date"].dt.dayofyear
)

weather_daily["doy_sin"] = np.sin(
    2 * np.pi
    * weather_daily["day_of_year"]
    / 365.25
)

weather_daily["doy_cos"] = np.cos(
    2 * np.pi
    * weather_daily["day_of_year"]
    / 365.25
)

weather_daily["years_since_1980"] = (
    weather_daily["date"].dt.year - 1980
)

# Remove temporary column
weather_daily = weather_daily.drop(
    columns=["day_of_year"]
)


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

print("\n==============================")
print("DATA QUALITY CHECKS")
print("==============================")


# ------------------------------------------------------------
# 6. Date checks
# ------------------------------------------------------------

duplicate_dates = (
    weather_daily["date"]
    .duplicated()
    .sum()
)

expected_dates = pd.date_range(
    start=weather_daily["date"].min(),
    end=weather_daily["date"].max(),
    freq="D",
)

missing_dates = expected_dates.difference(
    weather_daily["date"]
)

print("\nDate checks:")
print("Duplicate dates:", duplicate_dates)
print("Missing dates:", len(missing_dates))


# ------------------------------------------------------------
# 7. Missing value check
# ------------------------------------------------------------

print("\nMissing values:")

missing_values = weather_daily.isna().sum()

print(missing_values)


# ------------------------------------------------------------
# 8. Temperature checks
# ------------------------------------------------------------

print("\nTemperature checks:")

print(
    "Days where t_max < t_min:",
    (
        weather_daily["t_max"]
        < weather_daily["t_min"]
    ).sum()
)

print(
    "Minimum t_min:",
    weather_daily["t_min"].min()
)

print(
    "Maximum t_max:",
    weather_daily["t_max"].max()
)


# ------------------------------------------------------------
# 9. Humidity checks
# ------------------------------------------------------------

print("\nHumidity checks:")

print(
    "RH below 0%:",
    (
        weather_daily[
            "relative_humidity_mean"
        ] < 0
    ).sum()
)

print(
    "RH above 100%:",
    (
        weather_daily[
            "relative_humidity_mean"
        ] > 100
    ).sum()
)

print(
    "Minimum RH:",
    weather_daily[
        "relative_humidity_mean"
    ].min()
)

print(
    "Maximum RH:",
    weather_daily[
        "relative_humidity_mean"
    ].max()
)


# ------------------------------------------------------------
# 10. Wind speed check
# ------------------------------------------------------------

print("\nWind speed check:")

print(
    "Negative wind speed:",
    (
        weather_daily["wind_speed_mean"]
        < 0
    ).sum()
)


# ------------------------------------------------------------
# 11. Surface pressure check
# ------------------------------------------------------------

print("\nSurface pressure check:")

print(
    "Non-positive surface pressure:",
    (
        weather_daily[
            "surface_pressure_mean"
        ] <= 0
    ).sum()
)


# ------------------------------------------------------------
# 12. Radiation check
# ------------------------------------------------------------

print("\nShortwave radiation check:")

print(
    "Negative shortwave radiation:",
    (
        weather_daily[
            "shortwave_radiation_sum"
        ] < 0
    ).sum()
)


# ============================================================
# SAVE FINAL PROCESSED DATASET
# ============================================================

output_file = (
    PROCESSED_DIR
    / "weather_daily.parquet"
)

weather_daily.to_parquet(
    output_file,
    index=False,
)


print("\n==============================")
print("PROCESSING COMPLETE")
print("==============================")

print(
    f"Saved: {output_file}"
)

print(
    f"Final rows: {len(weather_daily)}"
)

print(
    f"Final columns: "
    f"{weather_daily.columns.tolist()}"
)