# data/imd_mock.py
# -----------------------------------------------------------------------------
# Mock IMD (India Meteorological Department) weather data for Punjab.
# Generates realistic daily weather based on historical statistics.
#
# Sources modelled after IMD seasonal reports for Ludhiana/Amritsar district:
#   - Kharif (rice) season: June-October
#   - Rabi season: November-April (wheat/mustard)
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

# -- Punjab historical monthly climate statistics ------------------------------
PUNJAB_MONTHLY = {
    1:  (0.5,  15.0, 1.0, 3.0, 0.05),   # January
    2:  (0.8,  18.0, 1.5, 3.5, 0.06),   # February
    3:  (1.2,  24.0, 2.5, 3.0, 0.07),   # March
    4:  (0.5,  30.5, 1.5, 3.0, 0.18),   # April
    5:  (0.3,  35.5, 1.2, 2.5, 0.25),   # May
    6:  (3.5,  34.0, 5.5, 2.5, 0.20),   # June
    7:  (9.0,  31.5, 8.5, 2.0, 0.12),   # July
    8:  (8.0,  31.0, 7.5, 1.8, 0.14),   # August
    9:  (4.5,  29.5, 5.5, 2.0, 0.10),   # September
    10: (0.5,  26.5, 1.5, 2.5, 0.07),   # October
    11: (0.2,  21.5, 0.8, 3.5, 0.06),   # November
    12: (0.3,  15.5, 1.0, 3.5, 0.05),   # December
}

def _month_to_season(month: int) -> str:
    if month in (6,):
        return "pre_monsoon"
    elif month in (7, 8, 9):
        return "monsoon"
    elif month in (10, 11):
        return "post_monsoon"
    elif month in (4, 5):
        return "summer"
    else:
        return "rabi"

@dataclass
class DailyWeather:
    day: int
    month: int
    rain_mm: float
    temp_celsius: float
    humidity_pct: float
    power_cut: bool
    power_cut_duration_h: int
    event: str
    season: str

class IMDWeatherGenerator:
    def __init__(self, start_month: int = 7, seed: int = 42, scenario: str = "monsoon"):
        self._rng = random.Random(seed)
        self._start_month = start_month
        self._scenario = scenario
        self._drought_modifier = {
            "monsoon":     1.0,
            "pre_monsoon": 0.7,
            "drought":     0.25,
        }.get(scenario, 1.0)

    def generate_sequence(self, n_days: int) -> List[DailyWeather]:
        sequence: List[DailyWeather] = []
        for day_idx in range(n_days):
            month = ((self._start_month - 1 + day_idx // 30) % 12) + 1
            weather = self._generate_day(day_idx + 1, month)
            sequence.append(weather)
        return sequence

    def get_forecast(self, sequence: List[DailyWeather], current_day: int, n_ahead: int = 3) -> List[DailyWeather]:
        forecasts = []
        for offset in range(1, n_ahead + 1):
            idx = current_day - 1 + offset
            if idx < len(sequence):
                base = sequence[idx]
                noise_scale = 0.1 * offset
                noisy = DailyWeather(
                    day=base.day,
                    month=base.month,
                    rain_mm=max(0.0, base.rain_mm + self._rng.gauss(0, noise_scale * base.rain_mm + 0.5)),
                    temp_celsius=base.temp_celsius + self._rng.gauss(0, noise_scale * 2),
                    humidity_pct=min(100.0, max(20.0, base.humidity_pct + self._rng.gauss(0, 3.0))),
                    power_cut=base.power_cut,
                    power_cut_duration_h=base.power_cut_duration_h,
                    event=base.event,
                    season=base.season,
                )
                forecasts.append(noisy)
        return forecasts

    def _generate_day(self, day: int, month: int) -> DailyWeather:
        avg_rain, avg_temp, rain_std, temp_std, pc_prob = PUNJAB_MONTHLY[month]
        avg_rain = avg_rain * self._drought_modifier
        if avg_rain > 0:
            rain = max(0.0, self._rng.gauss(avg_rain, rain_std))
            if month in (7, 8) and self._rng.random() < 0.15:
                rain *= self._rng.uniform(2.5, 5.0)
        else:
            rain = 0.0
        temp = self._rng.gauss(avg_temp, temp_std)
        base_hum = 50 + (rain / (avg_rain + 1)) * 20
        humidity = min(100.0, max(20.0, base_hum + self._rng.gauss(0, 8)))
        scenario_pc_multiplier = 1.5 if self._scenario == "drought" else 1.0
        power_cut = self._rng.random() < (pc_prob * scenario_pc_multiplier)
        cut_hours = 0
        if power_cut:
            if month in (4, 5, 6):
                cut_hours = int(self._rng.uniform(6, 14))
            elif month in (7, 8):
                cut_hours = int(self._rng.uniform(2, 8))
            else:
                cut_hours = int(self._rng.uniform(2, 6))
        event = "normal"
        if rain > 30:
            event = "flood"
        elif temp > 42:
            event = "heatwave"
        season = _month_to_season(month)
        return DailyWeather(day, month, round(rain, 2), round(temp, 1), round(humidity, 1), power_cut, cut_hours, event, season)

TASK_SCENARIOS = {
    "easy": {"start_month": 7, "total_days": 30, "scenario": "monsoon", "seed": 42, "water_reserve": 5000.0, "description": "30-day kharif season"},
    "medium": {"start_month": 6, "total_days": 60, "scenario": "pre_monsoon", "seed": 7, "water_reserve": 3500.0, "description": "60-day season"},
    "hard": {"start_month": 4, "total_days": 90, "scenario": "drought", "seed": 13, "water_reserve": 2000.0, "description": "90-day dry season"},
}
