"""
envs/precision_irrigation_v3.py  — Phase 3-ready calibrated environment
========================================================================
Physics fixes vs v2:
  • Moisture per pump action halved (divisor TANK_CAPACITY × 4 → × 8)
    So medium action (3kL) adds ~0.032 moisture instead of ~0.128
  • ET scaled more gently
  • Canal refill capped at 10 % (not 20 %) so tank matters more
  • Flash flood threshold raised to 35 mm

Reward fixes:
  • W_WATER reduced 15× (irrigation is not ruinously expensive)
  • W_STRESS / W_OVERWET reduced 6× (stress is informative, not dominant)
  • W_GROWTH increased (agent learns to grow the crop)
  • TERMINAL bonus kept high → long-run growth is the objective
  • Dense shaping: +small bonus when moisture inside optimal band
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, Optional

import numpy as np
import pandas as pd

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
RAINFALL_CSV = os.path.join(DATA_DIR, "rainfall.csv")
IMD_API_URL  = "http://localhost:8765/forecast"

MAX_STEPS     = 90
TANK_CAPACITY = 10_000.0
PUMP_EFF      = 0.85
GRID_SIZE     = 4

DRY_THR = 0.22
WET_THR = 0.78
OPT_LO  = 0.40   # optimal moisture band
OPT_HI  = 0.72

ET_COEFF   = 0.004   # gentler evapotranspiration
RAIN_SCALE = 0.0008  # mm → moisture fraction on 1 ha
FORECAST_NOISE_STD = 2.0

ACTION_VOLUMES = [0.0, 0.10, 0.30, 0.50, -0.10]

STAGES = [
    (0.00, "seedling",     0.014),
    (0.25, "vegetative",   0.022),
    (0.55, "reproductive", 0.018),
    (0.80, "maturity",     0.010),
]

# ── Reward calibration ────────────────────────────────────────────────────────
W_GROWTH   =  4.0     # growth delta reward (main signal)
W_OPTIMAL  =  0.08    # small bonus per step when in optimal moisture band
W_WATER    = -0.00001 # per litre pumped   (medium = 3k L → -0.03)
W_STRESS   = -0.30    # dry stress
W_OVERWET  = -0.20    # waterlogging
W_TERMINAL = 15.0     # crop_growth at episode end

STRESS_GROWTH_PENALTY = 0.010

REWARD_EPS = 1e-6


def _to_open_unit_interval(value: float) -> float:
    # Stable sigmoid mapping from unbounded reward to strict (0, 1).
    z = float(np.clip(value, -60.0, 60.0))
    y = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(y, REWARD_EPS, 1.0 - REWARD_EPS))


class PrecisionIrrigationEnvV3:
    """
    Calibrated RL environment for precision irrigation.
    Compatible with any PPO / DQN agent via obs dict + discrete action.
    """

    def __init__(self, render_mode=None, max_steps=MAX_STEPS,
                 use_api=True, seed=None):
        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.use_api     = use_api
        self._rng        = np.random.default_rng(seed)

        self._weather_df = self._load_weather()
        self._years      = sorted(self._weather_df["year"].unique())

        self._soil_moisture = 0.5
        self._crop_growth   = 0.1
        self._water_tank    = TANK_CAPACITY * 0.6
        self._day           = 0
        self._power_status  = 1
        self._grid_moisture = np.full((GRID_SIZE, GRID_SIZE), 0.5)
        self._current_year  = self._years[0]
        self._season_data   = pd.DataFrame()
        self._cum_water     = 0.0
        self._stress_days   = 0
        self._history: dict[str, list] = {}

    def _load_weather(self):
        if os.path.exists(RAINFALL_CSV):
            return pd.read_csv(RAINFALL_CSV)
        rng = np.random.default_rng(0)
        rows = [{"year": y, "season_day": d,
                 "rainfall_mm": float(max(0, rng.gamma(1.2, 4.5)) if rng.random() < .45 else 0),
                 "temp_celsius": float(np.clip(rng.normal(30, 2.5), 22, 42)),
                 "solar_mj": float(max(5, rng.normal(18, 2))),
                 "power_cut": int(rng.random() < .10)}
                for y in range(2020, 2026) for d in range(1, 91)]
        return pd.DataFrame(rows)

    def _get_row(self, day):
        idx = min(day, len(self._season_data) - 1)
        return self._season_data.iloc[idx]

    def _fetch_api_forecast(self, day: int) -> Optional[np.ndarray]:
        """Try the mock IMD API; return None on failure."""
        if not self.use_api:
            return None
        try:
            api_day = max(1, int(day) + 1)  # API expects season_day in [1..90]
            url = f"{IMD_API_URL}?year={self._current_year}&day={api_day}&days=3"
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                data = json.loads(resp.read())
            forecast = np.array(
                [[float(item["rain_mm"]), float(item["temp_c"])] for item in data],
                dtype=np.float32,
            )
            if forecast.shape != (3, 2):
                return None
            return forecast
        except Exception:
            return None

    def _get_forecast(self, from_day):
        api_fc = self._fetch_api_forecast(from_day)
        if api_fc is not None:
            noise = self._rng.normal(0, FORECAST_NOISE_STD, (3, 2))
            noise[:, 1] *= 0.3
            return np.clip(api_fc + noise, 0, None).astype(np.float32)

        fc = []
        for d in range(from_day, from_day + 3):
            row = self._get_row(d)
            fc.append([float(row["rainfall_mm"]) + self._rng.normal(0, FORECAST_NOISE_STD),
                       float(row["temp_celsius"]) + self._rng.normal(0, 0.4)])
        return np.clip(fc, 0, None).astype(np.float32)

    def _crop_stage(self):
        p = self._day / self.max_steps
        for i in range(len(STAGES)-1, -1, -1):
            if p >= STAGES[i][0]:
                return i, STAGES[i][2]
        return 0, STAGES[0][2]

    def _build_obs(self):
        noise = float(self._rng.normal(0, 0.02))
        si, _ = self._crop_stage()
        return {
            "soil_moisture_obs": np.array([np.clip(self._soil_moisture + noise, 0, 1)], dtype=np.float32),
            "crop_growth":       np.array([self._crop_growth],   dtype=np.float32),
            "crop_stage":        np.array([si],                  dtype=np.int32),
            "weather_forecast":  self._get_forecast(self._day),
            "water_tank":        np.array([self._water_tank / TANK_CAPACITY], dtype=np.float32),
            "power_status":      np.array([self._power_status],  dtype=np.int8),
            "day_of_season":     np.array([self._day],           dtype=np.int32),
            "grid_moisture":     self._grid_moisture.astype(np.float32),
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current_year = int(self._rng.choice(self._years))
        self._season_data  = self._weather_df[
            self._weather_df["year"] == self._current_year].reset_index(drop=True)
        self._soil_moisture = float(self._rng.uniform(0.40, 0.60))
        self._crop_growth   = float(self._rng.uniform(0.05, 0.12))
        self._water_tank    = float(self._rng.uniform(0.5, 0.8)) * TANK_CAPACITY
        self._day           = 0
        self._power_status  = 1
        self._grid_moisture = np.full((GRID_SIZE, GRID_SIZE), self._soil_moisture) + \
                              self._rng.uniform(-0.03, 0.03, (GRID_SIZE, GRID_SIZE))
        self._grid_moisture = np.clip(self._grid_moisture, 0, 1)
        self._cum_water     = 0.0
        self._stress_days   = 0
        self._history       = {k: [] for k in
                                ["moisture","growth","reward","action",
                                 "rain","temp","tank","stage"]}
        return self._build_obs(), {"year": self._current_year}

    def step(self, action: int):
        assert 0 <= action <= 4

        row        = self._get_row(self._day)
        rain_mm    = float(row["rainfall_mm"])
        temp_c     = float(row["temp_celsius"])
        solar_mj   = float(row["solar_mj"])
        power_cut  = bool(row["power_cut"])
        self._power_status = 0 if power_cut else 1

        flash = rain_mm > 35.0

        # ── Irrigation ────────────────────────────────────────────────────────
        vol = ACTION_VOLUMES[action]
        litres = 0.0
        if vol > 0 and self._power_status == 1:
            litres = min(vol * TANK_CAPACITY, self._water_tank)
            self._water_tank -= litres
            # FIXED: divisor × 8 → medium action (3kL) adds ~0.032 moisture
            moisture_delta = (litres * PUMP_EFF) / (TANK_CAPACITY * 8)
        elif vol < 0:
            moisture_delta = vol * 0.10   # drain: small negative
        else:
            moisture_delta = 0.0

        self._cum_water += litres

        rain_add = rain_mm * RAIN_SCALE
        if flash:
            rain_add += 0.08

        et = ET_COEFF * max(0, temp_c - 22) * self._soil_moisture

        prev_m = self._soil_moisture
        self._soil_moisture = float(np.clip(
            self._soil_moisture + moisture_delta + rain_add - et, 0, 1))

        # Grid
        noise = self._rng.normal(0, 0.008, (GRID_SIZE, GRID_SIZE))
        self._grid_moisture = np.clip(
            self._grid_moisture + moisture_delta + rain_add - et + noise, 0, 1)

        # Tank refill (canal): 10 % per day (not 20 % — makes tank matter)
        self._water_tank = min(self._water_tank + 0.10 * TANK_CAPACITY, TANK_CAPACITY)

        # ── Growth ────────────────────────────────────────────────────────────
        si, stage_rate = self._crop_stage()
        optimal = float(np.clip(1.0 - 2.5 * abs(self._soil_moisture - 0.56), 0, 1))
        sun     = solar_mj / 20.0
        prev_g  = self._crop_growth
        self._crop_growth = float(np.clip(
            self._crop_growth + stage_rate * optimal * sun, 0, 1))

        # Stress
        dry  = float(max(0, DRY_THR - self._soil_moisture) / DRY_THR)
        wet  = float(max(0, self._soil_moisture - WET_THR) / (1 - WET_THR))
        if flash: wet = min(1.0, wet + 0.25)

        self._crop_growth = float(np.clip(
            self._crop_growth - (dry + wet) * STRESS_GROWTH_PENALTY, 0, 1))
        if dry > 0.3 or wet > 0.3:
            self._stress_days += 1

        # ── Reward ────────────────────────────────────────────────────────────
        dg      = self._crop_growth - prev_g
        in_opt  = float(OPT_LO <= self._soil_moisture <= OPT_HI)
        reward_raw  = (W_GROWTH  * dg
                 + W_OPTIMAL * in_opt
                 + W_WATER   * litres
                 + W_STRESS  * dry
                 + W_OVERWET * wet)

        self._day  += 1
        terminated  = self._crop_growth >= 0.95
        truncated   = self._day >= self.max_steps
        final_yield = self._crop_growth * 8.0

        if terminated or truncated:
            reward_raw += W_TERMINAL * self._crop_growth

        reward = _to_open_unit_interval(reward_raw)

        for k, v in [("moisture", self._soil_moisture), ("growth", self._crop_growth),
                     ("reward", reward), ("action", action), ("rain", rain_mm),
                     ("temp", temp_c), ("tank", self._water_tank / TANK_CAPACITY),
                     ("stage", si)]:
            self._history[k].append(v)

        info = {
            "day": self._day, "rain_mm": rain_mm, "temp_c": temp_c,
            "flash_flood": flash, "litres_pumped": litres,
            "cum_water_kl": self._cum_water / 1000,
            "power_status": self._power_status, "dry_stress": dry, "wet_stress": wet,
            "stage": STAGES[si][1], "stress_days": self._stress_days,
            "yield_t_ha": final_yield,
            "water_eff_kg_kl": (final_yield * 1000) / max(self._cum_water / 1000, 0.001),
        }

        if self.render_mode == "human":
            self.render()

        return self._build_obs(), float(reward), terminated, truncated, info

    # ── Render (same rich plot as v2) ─────────────────────────────────────────
    def render(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
            from matplotlib.colors import LinearSegmentedColormap
            from matplotlib.patches import Patch
        except ImportError:
            return None

        days = list(range(len(self._history["moisture"])))
        if not days: return None

        si, _ = self._crop_stage()
        fig = plt.figure(figsize=(16, 9), facecolor="#0d1117")
        fig.suptitle(
            f"🌾 Precision Irrigation Agent v3  Year {self._current_year}  "
            f"Day {self._day}/{self.max_steps}  Stage: {STAGES[si][1].capitalize()}  "
            f"Yield: {self._crop_growth*8:.2f} t/ha  Water: {self._cum_water/1000:.1f} kL",
            color="white", fontsize=12, fontweight="bold")

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

        def sax(ax, title):
            ax.set_facecolor("#161b22"); ax.set_title(title, color="#8b949e", fontsize=8.5)
            ax.tick_params(colors="#8b949e", labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
            return ax

        AC = ["#444","#74b9ff","#0984e3","#6c5ce7","#fd79a8"]
        SC = ["#f9ca24","#6ab04c","#e55039","#d35400"]

        ax1 = sax(fig.add_subplot(gs[0,0]), "Soil Moisture")
        ax1.plot(days, self._history["moisture"], color="#58a6ff", lw=1.5)
        ax1.axhline(DRY_THR, color="#f85149", ls="--", lw=0.8, label="Dry")
        ax1.axhline(WET_THR, color="#fd79a8", ls="--", lw=0.8, label="Wet")
        ax1.axhspan(OPT_LO, OPT_HI, color="#3fb950", alpha=0.07, label="Optimal")
        ax1.set_ylim(0,1); ax1.legend(fontsize=6, labelcolor="white",
                                       facecolor="#21262d", edgecolor="#30363d")

        ax2 = sax(fig.add_subplot(gs[0,1]), "Crop Growth by Stage")
        stgs = self._history["stage"]; prev_s = stgs[0] if stgs else 0; ss = 0
        for i, s in enumerate(stgs):
            if s != prev_s or i == len(stgs)-1:
                end = i+1
                ax2.plot(days[ss:end], self._history["growth"][ss:end],
                         color=SC[prev_s], lw=1.8)
                ss = i; prev_s = s
        ax2.set_ylim(0,1)
        ax2.legend(handles=[Patch(facecolor=SC[i], label=STAGES[i][1].capitalize())
                             for i in range(4)],
                   fontsize=6, labelcolor="white", facecolor="#21262d", edgecolor="#30363d")

        cmap = LinearSegmentedColormap.from_list("dtw",["#c0392b","#f39c12","#27ae60","#2980b9"])
        ax3 = sax(fig.add_subplot(gs[0,2]), f"Farm Grid ({GRID_SIZE}×{GRID_SIZE})")
        im = ax3.imshow(self._grid_moisture, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        cb = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6, colors="white")
        ax3.set_xticks([]); ax3.set_yticks([])

        ax4 = sax(fig.add_subplot(gs[1,0]), "Water Tank Level")
        ax4.fill_between(days, self._history["tank"], color="#0984e3", alpha=0.5)
        ax4.plot(days, self._history["tank"], color="#74b9ff", lw=1.2)
        ax4.set_ylim(0,1)

        ax5 = sax(fig.add_subplot(gs[1,1]), "Rainfall (mm) + Flash Floods")
        ax5.bar(days, self._history["rain"], color="#74b9ff", alpha=0.7, width=1)
        for i, r in enumerate(self._history["rain"]):
            if r > 35: ax5.axvspan(i, i+1, color="#e17055", alpha=0.5)
        ax5.set_ylabel("mm", color="#8b949e", fontsize=7)

        ax6 = sax(fig.add_subplot(gs[1,2]), "Cumulative Reward")
        cum = np.cumsum(self._history["reward"])
        ax6.plot(days, cum, color="#e3b341", lw=1.5)
        ax6.axhline(0, color="#8b949e", lw=0.5)
        ax6.fill_between(days, cum, 0, where=cum>=0, color="#3fb950", alpha=0.2)
        ax6.fill_between(days, cum, 0, where=cum<0,  color="#f85149", alpha=0.2)

        ax7 = sax(fig.add_subplot(gs[2,:2]), "Daily Irrigation Actions")
        colors_b = [AC[a] for a in self._history["action"]]
        ax7.bar(days, [1]*len(days), color=colors_b, width=1.0, align="edge")
        ax7.set_xlim(0, self.max_steps); ax7.set_yticks([])
        ax7.legend(handles=[Patch(facecolor=c,label=l) for c,l in
                             zip(AC,["No water","Low","Medium","High","Drain"])],
                   fontsize=6, labelcolor="white", facecolor="#21262d",
                   edgecolor="#30363d", ncol=5, loc="upper right")

        ax8 = sax(fig.add_subplot(gs[2,2]), "Episode Stats")
        ax8.axis("off")
        for i,(k,v) in enumerate([
            ("Yield",       f"{self._crop_growth*8:.2f} t/ha"),
            ("Water used",  f"{self._cum_water/1000:.1f} kL"),
            ("Stress days", f"{self._stress_days}"),
            ("Tank",        f"{self._water_tank/TANK_CAPACITY*100:.0f}%"),
            ("Stage",       STAGES[si][1].capitalize()),
            ("Total rwd",   f"{sum(self._history['reward']):.1f}"),
        ]):
            ax8.text(0.05, 0.90-i*0.15, k, transform=ax8.transAxes, color="#8b949e", fontsize=8)
            ax8.text(0.55, 0.90-i*0.15, v, transform=ax8.transAxes, color="white",
                     fontsize=8, fontweight="bold")

        import io, matplotlib.image as mpimg
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor(), dpi=120)
        plt.close(fig); buf.seek(0)
        return (mpimg.imread(buf)*255).astype(np.uint8)

    def close(self): pass
