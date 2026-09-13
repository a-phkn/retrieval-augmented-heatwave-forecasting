import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/processed/weather_daily.parquet")
LOG_PATH = Path("data/processing_log.json")
PLOT_PATH = Path("data/processed/heatwave_sanity_2019_2022.png")

# -----------------------------
# Load processed data
# -----------------------------
df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

print("=" * 50)
print("2019 / 2022 SANITY CHECK")
print("=" * 50)

# -----------------------------
# 2019 check
# -----------------------------
for year in [2019, 2022]:
    year_df = df[df["date"].dt.year == year].copy()

    hottest = year_df.loc[year_df["t_max"].idxmax()]

    print(f"\n{year}:")
    print(f"  Maximum t_max: {hottest['t_max']:.2f} °C")
    print(f"  Date: {hottest['date'].date()}")

# -----------------------------
# Plot 2019 and 2022
# -----------------------------
plot_df = df[df["date"].dt.year.isin([2019, 2022])]

plt.figure(figsize=(12, 5))

for year in [2019, 2022]:
    year_df = plot_df[plot_df["date"].dt.year == year]
    plt.plot(
        year_df["date"],
        year_df["t_max"],
        label=str(year)
    )

plt.xlabel("Date")
plt.ylabel("Daily Maximum Temperature (°C)")
plt.title("Delhi/NCR Daily Maximum Temperature — 2019 and 2022")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
plt.close()

print(f"\nSanity-check plot saved: {PLOT_PATH}")

# -----------------------------
# Processing log
# -----------------------------
log = {
    "source": "Open-Meteo Historical Weather API",
    "dataset": "ERA5",
    "region": "Delhi_NCR",
    "spatial_cells": 9,
    "aggregation": "Arithmetic mean across 9 ERA5 grid cells",
    "frequency": "daily",
    "start_date": str(df["date"].min().date()),
    "end_date": str(df["date"].max().date()),
    "rows": int(len(df)),
    "columns": list(df.columns),
    "quality_checks": {
        "duplicate_dates": int(df["date"].duplicated().sum()),
        "missing_dates": int(
            len(
                pd.date_range(
                    df["date"].min(),
                    df["date"].max(),
                    freq="D"
                )
                .difference(df["date"])
            )
        ),
        "tmax_less_than_tmin": int((df["t_max"] < df["t_min"]).sum()),
        "rh_below_zero": int((df["relative_humidity_mean"] < 0).sum()),
        "rh_above_100": int((df["relative_humidity_mean"] > 100).sum()),
        "negative_wind": int((df["wind_speed_mean"] < 0).sum()),
        "non_positive_pressure": int(
            (df["surface_pressure_mean"] <= 0).sum()
        ),
        "negative_shortwave": int(
            (df["shortwave_radiation_sum"] < 0).sum()
        )
    }
}

with open(LOG_PATH, "w") as f:
    json.dump(log, f, indent=2)

print(f"Processing log saved: {LOG_PATH}")

print("\n" + "=" * 50)
print("SANITY CHECK COMPLETE")
print("=" * 50)