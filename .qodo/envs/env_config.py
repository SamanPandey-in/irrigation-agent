"""
evaluator/env_config.py
========================
Dataclass holding every tunable parameter of PrecisionIrrigationEnvV3.
Passed to ConfigurableEnv to override module-level constants at runtime.

Groups:
  PhysicsConfig   — moisture physics (ET, rain scale, pump efficiency)
  RewardConfig    — reward weights
  MoistureConfig  — optimal band + stress thresholds
  ActionConfig    — volumes for each of the 5 actions
  EpisodeConfig   — season length, tank capacity, grid size
  ScenarioConfig  — built-in presets (drought / flood / heat-wave / custom)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


_PRESET_ALIASES = {
    "normal": "🌦️ Normal",
    "drought": "🏜️ Drought",
    "flood": "🌊 Flood",
    "heat": "🌡️ Heat Wave",
    "heat_wave": "🌡️ Heat Wave",
    "power_crisis": "⚡ Power Crisis",
    "power_outage": "⚡ Power Crisis",
    "water_conservation": "💧 Water Conservation",
    "water_scarce": "💧 Water Conservation",
    "high_yield_focus": "🌾 High Yield Focus",
    "high_yield_challenge": "🌾 High Yield Focus",
    "hard": "🎯 Hard Mode",
    "hard_mode": "🎯 Hard Mode",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-configs
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsConfig:
    """Evapotranspiration, rainfall conversion, pump efficiency."""
    et_coeff:          float = 0.004    # ET per degree above 22 °C
    rain_scale:        float = 0.0008   # mm → moisture fraction
    pump_efficiency:   float = 0.85     # fraction of pumped water reaching soil
    flash_flood_thr:   float = 35.0     # mm/day that triggers a flash flood
    flash_flood_bonus: float = 0.08     # extra moisture added on flood
    tank_refill_rate:  float = 0.10     # fraction of capacity refilled per day
    power_cut_prob:    float = 0.05     # daily probability of pump power cut

    DISPLAY_NAMES = {
        "et_coeff":          ("ET Coefficient",           0.001, 0.015, 0.001),
        "rain_scale":        ("Rain→Moisture Scale",      0.0002, 0.002, 0.0001),
        "pump_efficiency":   ("Pump Efficiency",          0.5,   1.0,   0.05),
        "flash_flood_thr":   ("Flash Flood Threshold mm", 10.0,  80.0,  5.0),
        "flash_flood_bonus": ("Flash Flood Bonus",        0.0,   0.20,  0.01),
        "tank_refill_rate":  ("Tank Refill Rate/day",     0.0,   0.30,  0.01),
        "power_cut_prob":    ("Power Cut Probability",    0.0,   0.30,  0.01),
    }


@dataclass
class RewardConfig:
    """Reward signal weights."""
    w_growth:   float =  4.0       # growth delta multiplier
    w_optimal:  float =  0.08      # bonus per step in optimal moisture band
    w_water:    float = -0.00001   # per litre pumped (negative = cost)
    w_stress:   float = -0.30      # dry stress penalty
    w_overwet:  float = -0.20      # waterlogging penalty
    w_terminal: float =  15.0      # final crop_growth multiplier at episode end
    stress_growth_penalty: float = 0.010  # growth deducted per stress unit

    DISPLAY_NAMES = {
        "w_growth":              ("Growth Reward Weight",     0.0,  10.0, 0.5),
        "w_optimal":             ("Optimal Band Bonus",       0.0,  0.5,  0.01),
        "w_water":               ("Water Cost (×10⁻⁵)",       -0.0001, 0.0, 0.000005),
        "w_stress":              ("Dry Stress Penalty",       -2.0, 0.0,  0.05),
        "w_overwet":             ("Overwet Penalty",          -2.0, 0.0,  0.05),
        "w_terminal":            ("Terminal Bonus Weight",    0.0,  30.0, 1.0),
        "stress_growth_penalty": ("Stress Growth Deduction",  0.0,  0.05, 0.002),
    }


@dataclass
class MoistureConfig:
    """Soil moisture thresholds that define stress and optimality."""
    opt_lo:  float = 0.40   # lower bound of optimal band
    opt_hi:  float = 0.72   # upper bound of optimal band
    dry_thr: float = 0.22   # moisture below this = severe drought stress
    wet_thr: float = 0.78   # moisture above this = waterlogging

    DISPLAY_NAMES = {
        "opt_lo":  ("Optimal Band Lower",  0.10, 0.60, 0.01),
        "opt_hi":  ("Optimal Band Upper",  0.40, 0.95, 0.01),
        "dry_thr": ("Dry Stress Threshold",0.05, 0.40, 0.01),
        "wet_thr": ("Wet Stress Threshold",0.55, 0.98, 0.01),
    }


@dataclass
class ActionConfig:
    """Volume fractions for each of the 5 actions (fraction of TANK_CAPACITY)."""
    # Original defaults: [0.0, 0.10, 0.30, 0.50, -0.10]
    vol_none:   float =  0.00   # action 0: no water
    vol_low:    float =  0.10   # action 1: low  (→ ~1 000 L)
    vol_medium: float =  0.30   # action 2: medium (→ ~3 000 L)
    vol_high:   float =  0.50   # action 3: high  (→ ~5 000 L)
    vol_drain:  float = -0.10   # action 4: drain (negative = removal)

    DISPLAY_NAMES = {
        "vol_none":   ("Action 0 – No Water (vol)", -0.20, 0.20, 0.01),
        "vol_low":    ("Action 1 – Low Vol",         0.01, 0.30, 0.01),
        "vol_medium": ("Action 2 – Medium Vol",      0.10, 0.60, 0.01),
        "vol_high":   ("Action 3 – High Vol",        0.20, 0.90, 0.01),
        "vol_drain":  ("Action 4 – Drain Vol",      -0.50, 0.0,  0.01),
    }

    def to_list(self) -> List[float]:
        return [self.vol_none, self.vol_low, self.vol_medium, self.vol_high, self.vol_drain]


@dataclass
class EpisodeConfig:
    """Season and environment structure."""
    max_steps:     int   = 90       # season length in days
    tank_capacity: float = 10_000.0 # litres
    grid_size:     int   = 4        # N×N sub-field grid
    seed:          Optional[int] = 42

    DISPLAY_NAMES = {
        "max_steps":     ("Season Length (days)", 30, 120, 1),
        "tank_capacity": ("Tank Capacity (L)",  1000, 50000, 500),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario presets
# ═══════════════════════════════════════════════════════════════════════════════

_PRESETS: dict[str, dict] = {
    "🌦️ Normal": {
        # Easy: rain is generous, ET is low, power is reliable, wide optimal band
        "physics": {"rain_scale": 0.0014, "et_coeff": 0.002, "power_cut_prob": 0.02, "tank_refill_rate": 0.15},
        "moisture": {"opt_lo": 0.38, "opt_hi": 0.74},
    },

    "🏜️ Drought": {
        # Medium: mild adversity that a smart agent can partially overcome.
        # Rain is close to normal, ET is still elevated, and occasional power cuts
        # force strategic timing without making the task physically impossible.
        "physics": {
            "rain_scale": 0.0009,
            "et_coeff": 0.0035,
            "power_cut_prob": 0.12,
            "tank_refill_rate": 0.08,
        },
        "moisture": {
            "dry_thr": 0.26,
        },
        "reward": {
            "w_stress": -0.40,
        },
    },

    "🌊 Flood": {
        # Hard: compounded adversity; physically difficult to sustain optimal moisture
        "physics": {"rain_scale": 0.00008, "et_coeff": 0.012, "power_cut_prob": 0.30,
                    "tank_refill_rate": 0.02, "flash_flood_thr": 18.0, "flash_flood_bonus": 0.12},
        "moisture": {"dry_thr": 0.32, "opt_lo": 0.46, "opt_hi": 0.64, "wet_thr": 0.74},
        "reward":  {"w_stress": -0.80, "w_overwet": -0.70, "w_water": -0.00008,
                    "w_terminal": 10.0, "stress_growth_penalty": 0.025},
    },

    "🌡️ Heat Wave": {
        "physics": {"et_coeff": 0.010},
        "reward":  {"w_stress": -0.40},
    },

    "⚡ Power Crisis": {
        "physics": {"power_cut_prob": 0.25, "tank_refill_rate": 0.04},
    },

    "🌾 High Yield Focus": {
        "reward": {"w_growth": 7.0, "w_terminal": 25.0, "w_water": -0.000005},
    },

    "💧 Water Conservation": {
        "reward": {"w_water": -0.0001, "w_growth": 2.0},
        "physics": {"pump_efficiency": 0.70},
    },

    "🎯 Hard Mode": {
        "physics": {"et_coeff": 0.008, "rain_scale": 0.0003, "power_cut_prob": 0.15},
        "moisture": {"dry_thr": 0.30, "opt_lo": 0.45, "opt_hi": 0.68},
        "reward":  {"w_stress": -0.60, "w_overwet": -0.50},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Master EnvConfig
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnvConfig:
    """
    Master configuration object.
    Pass to ConfigurableEnv() to override module-level constants.
    """
    physics:  PhysicsConfig  = field(default_factory=PhysicsConfig)
    reward:   RewardConfig   = field(default_factory=RewardConfig)
    moisture: MoistureConfig = field(default_factory=MoistureConfig)
    actions:  ActionConfig   = field(default_factory=ActionConfig)
    episode:  EpisodeConfig  = field(default_factory=EpisodeConfig)
    preset:   str            = "🌦️ Normal"
    sensor_noise_std: float  = 0.008

    @property
    def opt_lo(self) -> float:
        return self.moisture.opt_lo

    @property
    def opt_hi(self) -> float:
        return self.moisture.opt_hi

    @property
    def max_steps(self) -> int:
        return self.episode.max_steps

    @property
    def tank_capacity(self) -> float:
        return self.episode.tank_capacity

    @staticmethod
    def from_preset(name: str) -> "EnvConfig":
        """Build config from a named preset."""
        resolved = _PRESET_ALIASES.get(str(name).strip().lower(), name)
        cfg = EnvConfig(preset=resolved)
        overrides = _PRESETS.get(resolved, {})
        for group, vals in overrides.items():
            sub = getattr(cfg, group)
            for k, v in vals.items():
                setattr(sub, k, v)
        return cfg

    @staticmethod
    def preset_names() -> List[str]:
        return list(_PRESETS.keys())

    def to_dict(self) -> dict:
        return {
            "preset":   self.preset,
            "sensor_noise_std": self.sensor_noise_std,
            "physics":  asdict(self.physics),
            "reward":   asdict(self.reward),
            "moisture": asdict(self.moisture),
            "actions":  asdict(self.actions),
            "episode":  asdict(self.episode),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_dict(d: dict) -> "EnvConfig":
        cfg = EnvConfig()
        cfg.preset = d.get("preset", "Custom")
        cfg.sensor_noise_std = float(d.get("sensor_noise_std", cfg.sensor_noise_std))
        for k, v in d.get("physics",  {}).items(): setattr(cfg.physics,  k, v)
        for k, v in d.get("reward",   {}).items(): setattr(cfg.reward,   k, v)
        for k, v in d.get("moisture", {}).items(): setattr(cfg.moisture, k, v)
        for k, v in d.get("actions",  {}).items(): setattr(cfg.actions,  k, v)
        for k, v in d.get("episode",  {}).items(): setattr(cfg.episode,  k, v)

        # Backward-compatible top-level overrides used by CLI flags.
        if "et_coeff" in d:
            cfg.physics.et_coeff = float(d["et_coeff"])
        if "max_steps" in d:
            cfg.episode.max_steps = int(d["max_steps"])
        if "tank_capacity" in d:
            cfg.episode.tank_capacity = float(d["tank_capacity"])
        if "power_cut_prob" in d:
            cfg.physics.power_cut_prob = float(d["power_cut_prob"])
        if "rain_multiplier" in d:
            cfg.physics.rain_scale = float(cfg.physics.rain_scale) * float(d["rain_multiplier"])

        return cfg

    @staticmethod
    def from_json(path: str) -> "EnvConfig":
        with open(path, "rb") as f:
            raw = f.read()

        # PowerShell redirection can emit UTF-16; try common encodings safely.
        for enc in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
            try:
                return EnvConfig.from_dict(json.loads(raw.decode(enc)))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        raise ValueError(f"Could not parse JSON config file: {path}")

    def summary(self) -> str:
        return (
            f"{self.preset} | steps={self.episode.max_steps} | "
            f"band=[{self.moisture.opt_lo:.2f},{self.moisture.opt_hi:.2f}]"
        )

    def validate(self) -> List[str]:
        errs: List[str] = []
        if not (0.0 <= self.moisture.opt_lo <= self.moisture.opt_hi <= 1.0):
            errs.append("moisture band must satisfy 0 <= opt_lo <= opt_hi <= 1")
        if not (0.0 <= self.moisture.dry_thr <= 1.0):
            errs.append("dry_thr must be in [0,1]")
        if not (0.0 <= self.moisture.wet_thr <= 1.0):
            errs.append("wet_thr must be in [0,1]")
        if self.episode.max_steps <= 0:
            errs.append("max_steps must be > 0")
        if self.episode.tank_capacity <= 0:
            errs.append("tank_capacity must be > 0")
        if not (0.0 <= self.physics.power_cut_prob <= 1.0):
            errs.append("power_cut_prob must be in [0,1]")
        if self.sensor_noise_std < 0:
            errs.append("sensor_noise_std must be >= 0")
        return errs

    def diff_from_defaults(self) -> List[str]:
        """Return list of parameters that differ from defaults."""
        defaults = EnvConfig()
        changed = []
        for group in ("physics", "reward", "moisture", "actions", "episode"):
            sub = getattr(self, group)
            def_sub = getattr(defaults, group)
            for k in sub.__dataclass_fields__:
                if getattr(sub, k) != getattr(def_sub, k):
                    changed.append(f"{group}.{k}: {getattr(def_sub, k)} → {getattr(sub, k)}")
        return changed

    def summary_text(self) -> str:
        lines = [f"Preset: {self.preset}"]
        diffs = self.diff_from_defaults()
        if diffs:
            lines.append("Custom overrides:")
            lines += [f"  • {d}" for d in diffs]
        else:
            lines.append("All parameters at defaults.")
        m = self.moisture
        lines += [
            f"Optimal Band: [{m.opt_lo:.2f}, {m.opt_hi:.2f}]",
            f"Stress Thresholds: Dry<{m.dry_thr:.2f}  Wet>{m.wet_thr:.2f}",
            f"Season: {self.episode.max_steps} days",
        ]
        return "\n".join(lines)
