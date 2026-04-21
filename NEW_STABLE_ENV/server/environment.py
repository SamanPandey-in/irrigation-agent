# server/environment.py
# -----------------------------------------------------------------------------
# PunjabPrecisionIrrigationEnvironment
#
# Simulates a Punjab rice paddy (kharif/basmati).
# -----------------------------------------------------------------------------

from __future__ import annotations
import math
import uuid
from typing import List, Optional, Tuple
from openenv.core.env_server import Environment
from models import (
    IrrigationAction,
    IrrigationObservation,
    IrrigationState,
    DayForecast,
)
from data.imd_mock import IMDWeatherGenerator, TASK_SCENARIOS, DailyWeather

PUMP_MAX_L = 500.0
ACTION_VOLUMES = {0: 0.0, 1: 50.0, 2: 150.0, 3: 250.0, 4: 0.0}
ACTION_NAMES = {0: "no_water", 1: "low_irrigation_50L", 2: "medium_irrigation_150L", 3: "high_irrigation_250L", 4: "drain_excess"}
WATER_COST_PER_L = 0.1
YIELD_BONUS_SCALE = 1000.0
TERMINAL_BONUS_SCALE = 1000.0
STRESS_PENALTY = 50.0
OPTIMAL_LOW, OPTIMAL_HIGH = 0.45, 0.70
STRESS_DRY, STRESS_WET = 0.25, 0.85
BASELINE_DAILY_L = 150.0

class PunjabPrecisionIrrigationEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False
    def __init__(self):
        self._cfg, self._weather_seq = {}, []
        self._day, self._soil_moisture, self._crop_growth = 0, 0.5, 0.0
        self._water_reserve, self._power_status, self._cum_water = 5000.0, 1, 0.0
        self._stress_days, self._cum_reward = 0, 0.0
        self._last_result, self._done = "Episode not started.", False
        self._state = IrrigationState()

    def reset(self, task_name: str = "easy", seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> IrrigationObservation:
        cfg = TASK_SCENARIOS.get(task_name, TASK_SCENARIOS["easy"])
        effective_seed = seed if seed is not None else cfg["seed"]
        self._cfg, self._day, self._soil_moisture, self._crop_growth = cfg, 0, 0.50, 0.02
        self._water_reserve, self._power_status, self._cum_water = cfg["water_reserve"], 1, 0.0
        self._stress_days, self._cum_reward, self._done = 0, 0.0, False
        self._last_result = "Season begins. Monitor soil and weather."
        self._weather_gen = IMDWeatherGenerator(start_month=cfg["start_month"], seed=effective_seed, scenario=cfg["scenario"])
        self._weather_seq = self._weather_gen.generate_sequence(cfg["total_days"])
        eid = episode_id or str(uuid.uuid4())
        self._state = IrrigationState(episode_id=eid, task_name=task_name, day=0, total_days=cfg["total_days"], seed=effective_seed, scenario=cfg["scenario"], cumulative_reward=0.0, crop_growth=self._crop_growth, soil_moisture=self._soil_moisture, water_reserve=self._water_reserve, cumulative_water_used=0.0)
        return self._build_obs(done=False, reward=None, episode_id=eid)

    def step(self, action: IrrigationAction, timeout_s: Optional[float] = None, **kwargs) -> IrrigationObservation:
        if self._done: return self._build_obs(done=True, reward=0.0, episode_id=self._state.episode_id)
        act = int(action.action)
        if act not in ACTION_VOLUMES: act = 0
        today = self._weather_seq[self._day]
        self._power_status = 0 if today.power_cut else 1
        liters_used, action_msg = self._apply_action(act, today)
        self._cum_water += liters_used
        self._soil_moisture = self._update_soil_moisture(today, liters_used, act)
        growth_delta = self._update_crop_growth(today)
        crop_stress = self._soil_moisture < STRESS_DRY or self._soil_moisture > STRESS_WET
        if crop_stress: self._stress_days += 1
        reward = self._compute_reward(growth_delta, liters_used, crop_stress)
        self._cum_reward += reward
        self._day += 1
        done = self._day >= self._cfg["total_days"]
        if done:
            terminal = self._crop_growth * TERMINAL_BONUS_SCALE
            reward += terminal
            self._cum_reward += terminal
        self._last_result = f"{action_msg} | Rain: {today.rain_mm}mm | Reward: {reward:+.2f}"
        self._state.day, self._state.cumulative_reward = self._day, round(self._cum_reward, 2)
        self._state.crop_growth, self._state.soil_moisture = round(self._crop_growth, 4), round(self._soil_moisture, 4)
        self._state.water_reserve, self._state.cumulative_water_used = round(self._water_reserve, 2), round(self._cum_water, 2)
        self._done = done
        return self._build_obs(done=done, reward=round(reward, 4), episode_id=self._state.episode_id)

    @property
    def state(self) -> IrrigationState: return self._state

    def _apply_action(self, act: int, today: DailyWeather) -> Tuple[float, str]:
        if act == 4:
            if self._soil_moisture > OPTIMAL_HIGH:
                drained = (self._soil_moisture - OPTIMAL_HIGH) * 0.6
                self._soil_moisture -= drained
                return 0.0, f"Drained {drained*100:.1f}% excess."
            return 0.0, "Drain skip."
        volume = ACTION_VOLUMES[act]
        if self._power_status == 0 and volume > 0: return 0.0, f"Power Cut: {today.power_cut_duration_h}h."
        if volume > self._water_reserve: volume = self._water_reserve
        self._water_reserve = max(0.0, self._water_reserve - volume)
        return volume, f"Applied {volume}L."

    def _update_soil_moisture(self, today: DailyWeather, liters_used: float, act: int) -> float:
        et = (0.02 + 0.003 * max(0.0, today.temp_celsius - 20.0)) * (1.05 + 0.25 * self._crop_growth)
        sm = self._soil_moisture - et + min(0.20, today.rain_mm / 250.0)
        self._water_reserve = min(self._cfg["water_reserve"], self._water_reserve + today.rain_mm * 5.0)
        sm += liters_used / 2000.0
        if sm > 0.92: sm -= (sm - 0.92) * 0.5
        return round(min(1.0, max(0.0, sm)), 4)

    def _update_crop_growth(self, today: DailyWeather) -> float:
        sm = self._soil_moisture
        m_f = 1.0 if OPTIMAL_LOW <= sm <= OPTIMAL_HIGH else (max(0.0, sm/OPTIMAL_LOW)**1.5 if sm < OPTIMAL_LOW else max(0.0, 1.0-(sm-OPTIMAL_HIGH)/0.3))
        t = today.temp_celsius
        t_f = 1.0 if 26<=t<=34 else (max(0.0, (t-10)/16) if t<26 else max(0.0, 1.0-(t-34)/10))
        prog = self._day / self._cfg["total_days"]
        base = 0.025 if prog < 0.25 else (0.045 if prog < 0.65 else 0.02)
        delta = base * m_f * t_f
        self._crop_growth = min(1.0, self._crop_growth + delta)
        return round(delta, 5)

    def _compute_reward(self, delta: float, used: float, stress: bool) -> float:
        return round(delta * YIELD_BONUS_SCALE - used * WATER_COST_PER_L - (STRESS_PENALTY if stress else 0.0), 4)

    def _build_obs(self, done: bool, reward: Optional[float], episode_id: str) -> IrrigationObservation:
        day_idx = min(self._day, len(self._weather_seq) - 1)
        today = self._weather_seq[day_idx]
        forecasts = [DayForecast(day_offset=i+1, rain_mm=round(fw.rain_mm,2), temp_celsius=round(fw.temp_celsius,1), humidity_pct=round(fw.humidity_pct,1), power_cut_risk=round(0.12 if fw.power_cut else 0.05, 2)) for i, fw in enumerate(self._weather_gen.get_forecast(self._weather_seq, day_idx, 3))]
        weff = round((self._crop_growth / 0.55) / (max(1.0, self._cum_water) / (BASELINE_DAILY_L * self._cfg["total_days"])), 3) if self._cfg else 1.0
        return IrrigationObservation(episode_id=episode_id, done=done, reward=reward, day=self._day, total_days=self._cfg.get("total_days", 30), soil_moisture=round(self._soil_moisture, 4), crop_growth=round(self._crop_growth, 4), water_reserve=round(self._water_reserve, 2), power_status=self._power_status, current_rain_mm=today.rain_mm, current_temp_celsius=today.temp_celsius, weather_forecast=forecasts, season=today.season, cumulative_water_used=round(self._cum_water, 2), stress_days=self._stress_days, last_action_result=self._last_result, water_efficiency_index=weff, echoed_message=f"Day {self._day} | Growth {self._crop_growth:.1%} | SM {self._soil_moisture:.1%}")
