# server/graders.py
from typing import Any
def _clamp01(v: float) -> float:
    c = round(max(0.0, min(1.0, v)), 4)
    return 0.001 if c <= 0 else (0.999 if c >= 1 else c)
def _base_grade(final_obs, rewards, steps, target_growth, total_days, budget) -> float:
    cg = float(final_obs.get("crop_growth", 0.0))
    weff = float(final_obs.get("water_efficiency_index", 0.0))
    stress = int(final_obs.get("stress_days", steps))
    water = float(final_obs.get("cumulative_water_used", budget))
    total = 0.6 * _clamp01((cg/max(0.01, target_growth))**2) + 0.2 * _clamp01(weff/2.0) + 0.1 * _clamp01(1.0 - stress/max(1, steps)) + 0.1 * _clamp01(1.0 - min(1.0, water/max(1, budget)) + 0.3)
    return _clamp01(total)
def grade_task(name, final_obs, rewards, steps) -> float:
    diff = "hard" if "hard" in name else ("medium" if "medium" in name else "easy")
    targets = {"easy": 0.70, "medium": 0.72, "hard": 0.65}
    budgets = {"easy": 5000.0, "medium": 3500.0, "hard": 2000.0}
    days = {"easy": 30, "medium": 60, "hard": 90}
    return _base_grade(final_obs, rewards, steps, targets[diff], days[diff], budgets[diff])
