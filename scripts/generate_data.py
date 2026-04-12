"""
Generate mock IMD (India Meteorological Department) weather data
for Punjab rice season (June-October) 2020-2025.
Run once: python scripts/generate_data.py
"""

import numpy as np
import pandas as pd
import os

def generate_punjab_weather(seed=42):
    np.random.seed(seed)
    records = []

    years = range(2021, 2027)
    # Rice kharif season: Jun(6) to Oct(10)
    season_months = {
        6:  {"rain_mean": 4.0,  "rain_std": 6.0,  "temp_mean": 33, "temp_std": 2.0},
        7:  {"rain_mean": 9.0,  "rain_std": 10.0, "temp_mean": 31, "temp_std": 1.5},
        8:  {"rain_mean": 8.0,  "rain_std": 9.0,  "temp_mean": 30, "temp_std": 1.5},
        9:  {"rain_mean": 5.0,  "rain_std": 7.0,  "temp_mean": 29, "temp_std": 2.0},
        10: {"rain_mean": 1.5,  "rain_std": 3.0,  "temp_mean": 26, "temp_std": 2.5},
    }

    for year in years:
        for month, params in season_months.items():
            days_in_month = 30 if month in [6, 9] else 31 if month in [7, 8, 10] else 28
            for day in range(1, days_in_month + 1):
                # Rainfall: Poisson-like (gamma clipped), mm/day
                rain = max(0.0, np.random.gamma(1.2, params["rain_mean"] / 1.2)
                           if np.random.random() < 0.45 else 0.0)
                rain = round(min(rain, 80.0), 2)   # cap at 80mm/day

                # Temperature
                temp = round(np.random.normal(params["temp_mean"], params["temp_std"]), 1)
                temp = float(np.clip(temp, 22, 42))

                # Solar radiation proxy (MJ/m2/day): inversely related to rain
                solar = round(max(5.0, np.random.normal(18 - rain * 0.2, 2.0)), 1)

                # Power cut flag
                power_cut = int(np.random.random() < 0.10)

                records.append({
                    "date": f"{year}-{month:02d}-{day:02d}",
                    "year": year,
                    "month": month,
                    "day_of_month": day,
                    "rainfall_mm": rain,
                    "temp_celsius": temp,
                    "solar_mj": solar,
                    "power_cut": power_cut,
                })

    df = pd.DataFrame(records)
    # Add a sequential season_day column (1..N per year)
    df["season_day"] = df.groupby("year").cumcount() + 1
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_punjab_weather()
    out_path = os.path.join("data", "rainfall.csv")
    df.to_csv(out_path, index=False)
    print(f"✅  Generated {len(df)} rows → {out_path}")
    print(df.head(10).to_string())
    print(f"\nStats:\n{df[['rainfall_mm','temp_celsius','solar_mj']].describe().round(2)}")
