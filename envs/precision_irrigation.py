"""
PrecisionIrrigationEnv — Gymnasium-compatible RL environment
Simulates a 1-ha Punjab rice farm. An agent decides daily irrigation
to maximise yield while minimising groundwater use.

Usage:
    import gymnasium as gym
    import envs  # triggers registration
    env = gym.make("PrecisionIrrigation-v0")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(2)
"""

from __future__ import annotations

import os
import random
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional Gymnasium import — falls back to a minimal shim so the module can
# be imported even before gymnasium is installed (e.g. during unit tests that
# mock the class).
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:                      # pragma: no cover
    GYM_AVAILABLE = False
    # Minimal shim so the class definition still works
    class _FakeGym:
        class Env:
            metadata: dict = {}
            spec = None
            def __init_subclass__(cls, **kwargs): ...
            def reset(self, *, seed=None, options=None): ...
            def step(self, action): ...
    class _FakeSpaces:
        class Dict:
            def __init__(self, *a, **kw): ...
            def contains(self, x): return True
        class Box:
            def __init__(self, *a, **kw): ...
            def contains(self, x): return True
        class Discrete:
            def __init__(self, n): self.n = n
            def contains(self, x): return 0 <= int(x) < self.n
    gym = _FakeGym()       # type: ignore[assignment]
    spaces = _FakeSpaces() # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Constants / tuneable parameters
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAINFALL_CSV = os.path.join(DATA_DIR, "rainfall.csv")

MAX_STEPS        = 90           # days per episode (rice kharif season)
FARM_AREA_HA     = 1.0          # hectares
TANK_CAPACITY    = 10_000.0     # litres max
TANK_DAILY_REFILL_FRAC = 0.20  # canal tops up 20 % of capacity daily
PUMP_EFFICIENCY  = 0.85         # fraction actually reaching roots

# Irrigation volumes per action (fraction of tank capacity)
ACTION_VOLUMES = [0.0, 0.10, 0.30, 0.50, -0.15]   # negative = drain

# Evapotranspiration coefficient per °C above 20 °C (fraction of moisture/day)
ET_COEFF = 0.007

# Stress thresholds
DRY_THRESHOLD  = 0.20   # below → drought stress
WET_THRESHOLD  = 0.80   # above → waterlogging stress

# Growth physics
GROWTH_RATE_BASE = 0.018        # max daily growth fraction under ideal moisture
STRESS_GROWTH_PENALTY = 0.015  # growth loss per stress unit

# Reward weights
W_GROWTH    =  2.0
W_WATER     = -0.0002   # per litre used (pump energy + scarcity cost)
W_STRESS    = -1.5
W_OVERWET   = -0.8
TERMINAL_YIELD_BONUS = 10.0    # final multiplier on crop_growth at episode end

REWARD_EPS = 1e-6


def _to_open_unit_interval(value: float) -> float:
    # Stable sigmoid mapping from unbounded reward to strict (0, 1).
    z = float(np.clip(value, -60.0, 60.0))
    y = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(y, REWARD_EPS, 1.0 - REWARD_EPS))


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
class PrecisionIrrigationEnv(gym.Env):
    """
    Observation (Dict):
        soil_moisture   float32  [0, 1]         fraction of field capacity
        crop_growth     float32  [0, 1]         fraction of max yield achieved
        weather_forecast float32 [3 days × 2]  [[rain_mm, temp_C], ...]
        water_tank      float32  [0, 1]         normalised tank level
        power_status    int8     {0, 1}         1 = power available
        day_of_season   int16    [0, MAX_STEPS] current step

    Action (Discrete 5):
        0  — no irrigation
        1  — low   (10 % tank ≈  1 000 L)
        2  — medium (30 % tank ≈  3 000 L)
        3  — high   (50 % tank ≈  5 000 L)
        4  — drain  (remove 15 % excess to prevent waterlogging)

    Reward (dense + sparse terminal):
        dense  = W_GROWTH * Δgrowth + W_WATER * litres_used
                 + W_STRESS * dry_stress + W_OVERWET * wet_stress
        terminal (done) += TERMINAL_YIELD_BONUS * crop_growth
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = MAX_STEPS,
        farm_size_ha: float = FARM_AREA_HA,
        data_path: str = RAINFALL_CSV,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.render_mode   = render_mode
        self.max_steps     = max_steps
        self.farm_size_ha  = farm_size_ha
        self.data_path     = data_path

        # ── Action & Observation spaces ─────────────────────────────────────
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Dict({
            "soil_moisture":     spaces.Box(0.0, 1.0,  shape=(1,), dtype=np.float32),
            "crop_growth":       spaces.Box(0.0, 1.0,  shape=(1,), dtype=np.float32),
            "weather_forecast":  spaces.Box(-np.inf, np.inf, shape=(3, 2), dtype=np.float32),
            "water_tank":        spaces.Box(0.0, 1.0,  shape=(1,), dtype=np.float32),
            "power_status":      spaces.Box(0,   1,    shape=(1,), dtype=np.int8),
            "day_of_season":     spaces.Box(0,   max_steps, shape=(1,), dtype=np.int32),
        })

        # ── Load weather data ───────────────────────────────────────────────
        self._weather_df = self._load_weather(data_path)
        self._years = sorted(self._weather_df["year"].unique())

        # ── Internal state (populated in reset) ────────────────────────────
        self._soil_moisture: float = 0.4
        self._crop_growth:   float = 0.1
        self._water_tank:    float = 0.5 * TANK_CAPACITY
        self._day:           int   = 0
        self._current_year:  int   = self._years[0]
        self._season_data:   pd.DataFrame = self._weather_df[
            self._weather_df["year"] == self._current_year
        ].reset_index(drop=True)

        # ── History for rendering ───────────────────────────────────────────
        self._history: dict[str, list] = {"moisture": [], "growth": [],
                                           "reward": [], "action": []}
        self._rng = np.random.default_rng(seed)

    # ─────────────────────────── helpers ────────────────────────────────────
    @staticmethod
    def _load_weather(path: str) -> pd.DataFrame:
        if os.path.exists(path):
            return pd.read_csv(path)
        # Fallback: generate synthetic data inline so env works without CSV
        import warnings
        warnings.warn(f"Weather CSV not found at {path}. Using synthetic fallback.", stacklevel=2)
        rng = np.random.default_rng(0)
        records = []
        for year in range(2020, 2026):
            for day in range(1, 91):
                month = 6 + (day - 1) // 30
                rain  = float(max(0, rng.gamma(1.2, 4.5)) if rng.random() < 0.45 else 0)
                temp  = float(np.clip(rng.normal(30, 2.5), 22, 42))
                solar = float(max(5, rng.normal(18 - rain * 0.2, 2)))
                records.append({"year": year, "season_day": day,
                                 "rainfall_mm": round(rain, 2),
                                 "temp_celsius": round(temp, 1),
                                 "solar_mj": round(solar, 1),
                                 "power_cut": int(rng.random() < 0.10)})
        return pd.DataFrame(records)

    def _get_weather_row(self, day: int) -> pd.Series:
        """Return the weather row for `day` (0-indexed) in the current season."""
        idx = min(day, len(self._season_data) - 1)
        return self._season_data.iloc[idx]

    def _get_forecast(self, from_day: int) -> np.ndarray:
        """Return 3-day forecast array [[rain, temp], ...] from `from_day`."""
        forecast = []
        for d in range(from_day, from_day + 3):
            row = self._get_weather_row(d)
            forecast.append([row["rainfall_mm"], row["temp_celsius"]])
        return np.array(forecast, dtype=np.float32)

    def _build_obs(self) -> dict[str, np.ndarray]:
        return {
            "soil_moisture":    np.array([self._soil_moisture],  dtype=np.float32),
            "crop_growth":      np.array([self._crop_growth],    dtype=np.float32),
            "weather_forecast": self._get_forecast(self._day),
            "water_tank":       np.array([self._water_tank / TANK_CAPACITY], dtype=np.float32),
            "power_status":     np.array([self._power_status],   dtype=np.int8),
            "day_of_season":    np.array([self._day],            dtype=np.int32),
        }

    # ─────────────────────────── Gym API ────────────────────────────────────
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Pick a random historical year for variety
        self._current_year = int(self._rng.choice(self._years))
        self._season_data  = self._weather_df[
            self._weather_df["year"] == self._current_year
        ].reset_index(drop=True)

        # Randomise start conditions slightly
        self._soil_moisture = float(self._rng.uniform(0.35, 0.55))
        self._crop_growth   = float(self._rng.uniform(0.05, 0.15))
        self._water_tank    = float(self._rng.uniform(0.4, 0.7)) * TANK_CAPACITY
        self._day           = 0
        self._power_status  = 1   # starts with power

        self._history = {"moisture": [], "growth": [], "reward": [], "action": []}

        obs  = self._build_obs()
        info = {"year": self._current_year, "initial_moisture": self._soil_moisture}
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        weather = self._get_weather_row(self._day)
        rain_mm   = float(weather["rainfall_mm"])
        temp_c    = float(weather["temp_celsius"])
        solar_mj  = float(weather["solar_mj"])
        power_cut = bool(weather["power_cut"])
        self._power_status = 0 if power_cut else 1

        # ── Irrigation ──────────────────────────────────────────────────────
        vol_fraction = ACTION_VOLUMES[action]
        litres_pumped = 0.0
        if vol_fraction > 0 and self._power_status == 1:
            litres_pumped = min(vol_fraction * TANK_CAPACITY, self._water_tank)
            self._water_tank   -= litres_pumped
            moisture_added      = (litres_pumped * PUMP_EFFICIENCY) / (TANK_CAPACITY * 2)
        elif vol_fraction < 0:
            # drain action — removes excess moisture
            litres_pumped    = 0.0
            moisture_added   = vol_fraction * 0.15   # negative Δ
        else:
            moisture_added   = 0.0

        # Rain contribution (1 mm/day ≈ 0.001 normalised moisture on 1 ha)
        rain_moisture = rain_mm * 0.001

        # Evapotranspiration (higher temp → more loss)
        et_loss = ET_COEFF * max(0, temp_c - 20) * self._soil_moisture

        prev_moisture = self._soil_moisture
        self._soil_moisture = float(np.clip(
            self._soil_moisture + moisture_added + rain_moisture - et_loss,
            0.0, 1.0
        ))

        # ── Tank refill (canal) ──────────────────────────────────────────────
        daily_refill = TANK_DAILY_REFILL_FRAC * TANK_CAPACITY
        self._water_tank = min(self._water_tank + daily_refill, TANK_CAPACITY)

        # ── Crop growth ──────────────────────────────────────────────────────
        # Optimal moisture window: 0.4–0.75
        optimal = 1.0 - 2.0 * abs(self._soil_moisture - 0.575)
        optimal = float(np.clip(optimal, 0.0, 1.0))
        sunlight_factor = solar_mj / 20.0          # normalised

        prev_growth = self._crop_growth
        delta_growth = GROWTH_RATE_BASE * optimal * sunlight_factor
        self._crop_growth = float(np.clip(self._crop_growth + delta_growth, 0.0, 1.0))

        # ── Stress signals ───────────────────────────────────────────────────
        dry_stress = float(max(0.0, DRY_THRESHOLD - self._soil_moisture) / DRY_THRESHOLD)
        wet_stress = float(max(0.0, self._soil_moisture - WET_THRESHOLD) / (1 - WET_THRESHOLD))

        # Stress sets back growth
        stress_penalty = (dry_stress + wet_stress) * STRESS_GROWTH_PENALTY
        self._crop_growth = float(np.clip(self._crop_growth - stress_penalty, 0.0, 1.0))

        # ── Reward ───────────────────────────────────────────────────────────
        actual_delta_growth = self._crop_growth - prev_growth
        reward_raw = (
            W_GROWTH  * actual_delta_growth
            + W_WATER   * litres_pumped
            + W_STRESS  * dry_stress
            + W_OVERWET * wet_stress
        )

        # ── Termination ──────────────────────────────────────────────────────
        self._day += 1
        terminated = self._crop_growth >= 0.95
        truncated  = self._day >= self.max_steps

        if terminated or truncated:
            # Sparse terminal bonus: reward proportional to final yield
            # Map growth fraction → tonnes/ha (rice max ~8 t/ha)
            simulated_yield_t = self._crop_growth * 8.0
            reward_raw += TERMINAL_YIELD_BONUS * self._crop_growth
            info_yield = simulated_yield_t
        else:
            info_yield = self._crop_growth * 8.0

        reward = _to_open_unit_interval(reward_raw)

        # ── Logging history ──────────────────────────────────────────────────
        self._history["moisture"].append(self._soil_moisture)
        self._history["growth"].append(self._crop_growth)
        self._history["reward"].append(reward)
        self._history["action"].append(action)

        obs  = self._build_obs()
        info = {
            "day":            self._day,
            "rain_mm":        rain_mm,
            "temp_c":         temp_c,
            "litres_pumped":  litres_pumped,
            "power_status":   self._power_status,
            "dry_stress":     dry_stress,
            "wet_stress":     wet_stress,
            "yield_t_ha":     info_yield,
            "tank_level":     self._water_tank,
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    # ─────────────────────────── Rendering ──────────────────────────────────
    def render(self) -> Optional[np.ndarray]:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("[render] matplotlib not installed — skipping visual render.")
            return None

        days = list(range(len(self._history["moisture"])))
        if not days:
            return None

        fig = plt.figure(figsize=(12, 7), facecolor="#0d1117")
        fig.suptitle(
            f"Precision Irrigation Agent — Year {self._current_year}  "
            f"Day {self._day}/{self.max_steps}",
            color="white", fontsize=13, fontweight="bold"
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ACTION_LABELS = ["No water", "Low (10%)", "Med (30%)", "High (50%)", "Drain"]
        ACTION_COLORS = ["#444", "#74b9ff", "#0984e3", "#6c5ce7", "#fd79a8"]

        def style_ax(ax, title):
            ax.set_facecolor("#161b22")
            ax.set_title(title, color="#8b949e", fontsize=9)
            ax.tick_params(colors="#8b949e", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        # 1. Soil moisture
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(days, self._history["moisture"], color="#58a6ff", lw=1.5)
        ax1.axhline(DRY_THRESHOLD, color="#f85149", ls="--", lw=0.8, label="Dry limit")
        ax1.axhline(WET_THRESHOLD, color="#3fb950", ls="--", lw=0.8, label="Wet limit")
        ax1.set_ylim(0, 1); ax1.legend(fontsize=6, labelcolor="white",
                                        facecolor="#21262d", edgecolor="#30363d")
        style_ax(ax1, "Soil Moisture")

        # 2. Crop growth
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(days, self._history["growth"], color="#3fb950", lw=1.5)
        ax2.set_ylim(0, 1)
        style_ax(ax2, f"Crop Growth (yield ≈ {self._crop_growth*8:.1f} t/ha)")

        # 3. Cumulative reward
        ax3 = fig.add_subplot(gs[1, 0])
        cum_reward = np.cumsum(self._history["reward"])
        ax3.plot(days, cum_reward, color="#e3b341", lw=1.5)
        ax3.axhline(0, color="#8b949e", lw=0.5)
        style_ax(ax3, "Cumulative Reward")

        # 4. Actions bar
        ax4 = fig.add_subplot(gs[1, 1])
        colors = [ACTION_COLORS[a] for a in self._history["action"]]
        ax4.bar(days, [1] * len(days), color=colors, width=1.0, align="edge")
        # Legend patches
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=l)
                           for c, l in zip(ACTION_COLORS, ACTION_LABELS)]
        ax4.legend(handles=legend_elements, fontsize=5, labelcolor="white",
                   facecolor="#21262d", edgecolor="#30363d",
                   loc="upper left", ncol=2)
        ax4.set_yticks([]); ax4.set_xlim(0, self.max_steps)
        style_ax(ax4, "Actions per Day")

        if self.render_mode == "human":
            plt.tight_layout()
            plt.pause(0.01)
            plt.show(block=False)
            return None
        else:
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            buf.seek(0)
            import matplotlib.image as mpimg
            img = mpimg.imread(buf)
            return (img * 255).astype(np.uint8)

    def close(self):
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except ImportError:
            pass
