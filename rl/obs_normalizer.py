"""
rl/obs_normalizer.py — Running mean/std normalizer for dict observations
"""

from __future__ import annotations
import numpy as np
from rl.ppo import flatten_obs


class RunningNormalizer:
    """Welford online mean/var + clip to [-5, 5] after normalising."""

    def __init__(self, dim: int, clip: float = 5.0, eps: float = 1e-8):
        self.dim   = dim
        self.clip  = clip
        self.eps   = eps
        self.mean  = np.zeros(dim, dtype=np.float64)
        self.var   = np.ones(dim,  dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray):
        self.count += 1
        delta      = x - self.mean
        self.mean += delta / self.count
        delta2     = x - self.mean
        self.var   = (self.var * (self.count - 1) + delta * delta2) / max(self.count, 1)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + self.eps)
        return np.clip((x - self.mean) / std, -self.clip, self.clip).astype(np.float32)

    def update_and_normalize(self, x: np.ndarray) -> np.ndarray:
        self.update(x)
        return self.normalize(x)


def manual_normalize(obs: dict) -> np.ndarray:
    """
    Fast hand-coded normalisation (no running stats needed).
    Maps each obs key to ~[-1, 1] range analytically.
    """
    parts = []
    order = sorted(obs.keys())
    for key in order:
        v = obs[key].astype(np.float32).ravel()
        if key == "soil_moisture_obs":
            parts.append(v * 2 - 1)                          # [0,1] → [-1,1]
        elif key == "crop_growth":
            parts.append(v * 2 - 1)
        elif key == "crop_stage":
            parts.append(v.astype(np.float32) / 3.0)         # [0,3] → [0,1]
        elif key == "weather_forecast":
            # rain [0,40] → [0,1], temp [20,42] → [-1,1]
            rain = np.clip(v[0::2], 0, 40) / 40.0
            temp = (np.clip(v[1::2], 20, 42) - 31) / 11.0
            interleaved = np.empty(len(rain) + len(temp), dtype=np.float32)
            interleaved[0::2] = rain
            interleaved[1::2] = temp
            parts.append(interleaved)
        elif key == "water_tank":
            parts.append(v * 2 - 1)
        elif key == "power_status":
            parts.append(v.astype(np.float32))
        elif key == "day_of_season":
            parts.append(v.astype(np.float32) / 90.0)
        elif key == "grid_moisture":
            parts.append(v * 2 - 1)
        else:
            parts.append(v)
    return np.concatenate(parts, dtype=np.float32)
