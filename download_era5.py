import json
import time
from calendar import monthrange
from datetime import date
from pathlib import Path

import requests

URL = "https://archive-api.open-meteo.com/v1/archive"

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

START_YEAR = 1980
TODAY = date.today()

RAW_DIR = Path("data/raw/era5_monthly")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def file_exists(cell_number, year, month):
    return (
        RAW_DIR
        / f"cell_{cell_number}"
        / f"{year}-{month:02d}.json"
    ).exists()


def download_month(cell_number, lat, lon, year, month):

    cell_dir = RAW_DIR / f"cell_{cell_number}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    output_file = cell_dir / f"{year}-{month:02d}.json"

    if output_file.exists():
        print(f"  Cell {cell_number}: already exists")
        return True

    start_date = date(year, month, 1)
    end_date = date(
        year,
        month,
        monthrange(year, month)[1]
    )

    if end_date > TODAY:
        end_date = TODAY

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "surface_pressure"
        ),
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "shortwave_radiation_sum"
        ),
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "ms",
        "cell_selection": "nearest",
        "models": "era5",
    }

    for attempt in range(1, 4):

        try:
            response = requests.get(
                URL,
                params=params,
                timeout=120
            )

            if response.status_code == 429:
                wait = 30 * attempt
                print(
                    f"  Cell {cell_number}: "
                    f"rate limited, waiting {wait}s"
                )
                time.sleep(wait)
                continue

            response.raise_for_status()

            data = response.json()

            with open(output_file, "w") as f:
                json.dump(data, f)

            print(f"  Cell {cell_number}: saved")
            return True

        except requests.RequestException as error:

            print(
                f"  Cell {cell_number}: "
                f"attempt {attempt} failed"
            )

            if attempt < 3:
                time.sleep(30 * attempt)

    print(f"  Cell {cell_number}: FAILED")
    return False


# ---------------------------------------------------------
# Find where the previous run stopped
# ---------------------------------------------------------

resume_year = START_YEAR
resume_month = 1
resume_cell = 1

found_resume_point = False

for year in range(START_YEAR, TODAY.year + 1):

    last_month = 12

    if year == TODAY.year:
        last_month = TODAY.month

    for month in range(1, last_month + 1):

        for cell_number in range(1, 10):

            if not file_exists(cell_number, year, month):

                resume_year = year
                resume_month = month
                resume_cell = cell_number
                found_resume_point = True
                break

        if found_resume_point:
            break

    if found_resume_point:
        break


print(
    f"\nResuming from "
    f"{resume_year}-{resume_month:02d}, "
    f"Cell {resume_cell}"
)


# ---------------------------------------------------------
# Continue downloading
# ---------------------------------------------------------

for year in range(resume_year, TODAY.year + 1):

    first_month = resume_month if year == resume_year else 1

    last_month = 12

    if year == TODAY.year:
        last_month = TODAY.month

    for month in range(first_month, last_month + 1):

        print(f"\n========== {year}-{month:02d} ==========")

        first_cell = (
            resume_cell
            if year == resume_year
            and month == resume_month
            else 1
        )

        for cell_number in range(first_cell, 10):

            lat, lon = CELLS[cell_number - 1]

            success = download_month(
                cell_number,
                lat,
                lon,
                year,
                month
            )

            if not success:
                print("\nDownload stopped safely.")
                print("Run the script again later to resume.")
                raise SystemExit

            time.sleep(2)


print("\nAll ERA5 monthly downloads completed.")