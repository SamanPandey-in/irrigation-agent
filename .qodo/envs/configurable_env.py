"""
envs/configurable_env.py
=========================
Wraps PrecisionIrrigationEnvV3 with runtime parameter injection from EnvConfig.
Monkey-patches step() so every physics constant, reward weight and threshold
comes from the config object — no source editing required.

Usage:
    from envs.configurable_env import make_env
    from envs.env_config import EnvConfig
    env = make_env(EnvConfig.from_preset("🏜️ Drought"), seed=42)
"""
from __future__ import annotations
import os, sys, types
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.env_config import EnvConfig


def make_env(cfg: EnvConfig | None = None, seed: int | None = None):
    if cfg is None:
        cfg = EnvConfig()

    from envs.precision_irrigation_v3 import PrecisionIrrigationEnvV3

    eff_seed = seed if seed is not None else (cfg.episode.seed or 42)
    env = PrecisionIrrigationEnvV3(
        max_steps=cfg.episode.max_steps,
        seed=eff_seed,
    )
    # Resize tank to configured capacity
    env._water_tank = cfg.episode.tank_capacity * 0.6

    # Attach dynamic config flags (used by patched step)
    env._flash_flood_thr   = cfg.physics.flash_flood_thr
    env._flash_flood_bonus = cfg.physics.flash_flood_bonus
    env._tank_refill_rate  = cfg.physics.tank_refill_rate
    env._power_cut_prob    = cfg.physics.power_cut_prob
    env._cfg               = cfg

    _patch_step(env, cfg)
    return env


def _patch_step(env, cfg: EnvConfig):
    p   = cfg.physics
    r   = cfg.reward
    m   = cfg.moisture
    avs = cfg.actions.to_list()
    tc  = cfg.episode.tank_capacity
    gs  = cfg.episode.grid_size

    def patched_step(self_env, action: int):
        assert 0 <= action <= 4

        row       = self_env._get_row(self_env._day)
        rain_mm   = float(row["rainfall_mm"])
        temp_c    = float(row["temp_celsius"])
        solar_mj  = float(row["solar_mj"])
        power_cut = bool(row.get("power_cut", False))
        if not power_cut:
            power_cut = (self_env._rng.random() < p.power_cut_prob)
        self_env._power_status = 0 if power_cut else 1

        flash = rain_mm > p.flash_flood_thr

        # ── Irrigation ────────────────────────────────────────────────────────
        vol    = avs[action]
        litres = 0.0
        if vol > 0 and self_env._power_status == 1:
            litres = min(vol * tc, self_env._water_tank)
            self_env._water_tank -= litres
            moisture_delta = (litres * p.pump_efficiency) / (tc * 8)
        elif vol < 0:
            moisture_delta = vol * 0.10
        else:
            moisture_delta = 0.0

        self_env._cum_water += litres

        rain_add = rain_mm * p.rain_scale
        if flash:
            rain_add += p.flash_flood_bonus

        et = p.et_coeff * max(0, temp_c - 22) * self_env._soil_moisture

        self_env._soil_moisture = float(np.clip(
            self_env._soil_moisture + moisture_delta + rain_add - et, 0, 1))

        noise = self_env._rng.normal(0, 0.008, (gs, gs))
        self_env._grid_moisture = np.clip(
            self_env._grid_moisture + moisture_delta + rain_add - et + noise, 0, 1)

        self_env._water_tank = min(
            self_env._water_tank + p.tank_refill_rate * tc, tc)

        # ── Growth ────────────────────────────────────────────────────────────
        si, stage_rate = self_env._crop_stage()
        optimal = float(np.clip(1.0 - 2.5 * abs(self_env._soil_moisture - 0.56), 0, 1))
        sun     = solar_mj / 20.0
        prev_g  = self_env._crop_growth
        self_env._crop_growth = float(np.clip(
            self_env._crop_growth + stage_rate * optimal * sun, 0, 1))

        dry_range = max(m.dry_thr, 1e-9)
        wet_range = max(1 - m.wet_thr, 1e-9)
        dry = float(max(0, m.dry_thr - self_env._soil_moisture) / dry_range)
        wet = float(max(0, self_env._soil_moisture - m.wet_thr) / wet_range)
        if flash:
            wet = min(1.0, wet + 0.25)

        self_env._crop_growth = float(np.clip(
            self_env._crop_growth - (dry + wet) * r.stress_growth_penalty, 0, 1))
        if dry > 0.3 or wet > 0.3:
            self_env._stress_days += 1

        # ── Reward ────────────────────────────────────────────────────────────
        dg     = self_env._crop_growth - prev_g
        in_opt = float(m.opt_lo <= self_env._soil_moisture <= m.opt_hi)
        reward = (r.w_growth  * dg
                + r.w_optimal * in_opt
                + r.w_water   * litres
                + r.w_stress  * dry
                + r.w_overwet * wet)

        self_env._day += 1
        done = self_env._day >= self_env.max_steps

        if done:
            reward += r.w_terminal * self_env._crop_growth

        obs  = self_env._build_obs()
        info = {
            "rain_mm": rain_mm, "et": et, "litres": litres,
            "dry": dry, "wet": wet, "flash": flash,
            "cum_water": self_env._cum_water,
        }
        return obs, reward, done, False, info

    env.step = types.MethodType(patched_step, env)
