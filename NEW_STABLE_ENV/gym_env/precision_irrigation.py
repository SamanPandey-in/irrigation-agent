# gym_env/precision_irrigation.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from server.environment import PunjabPrecisionIrrigationEnvironment
from models import IrrigationAction

class PrecisionIrrigationEnv(gym.Env):
    def __init__(self, task="easy", seed=None):
        super().__init__()
        self.task, self._seed = task, seed
        self._inner_env = PunjabPrecisionIrrigationEnvironment()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "soil_moisture": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "crop_growth": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "power_status": spaces.Discrete(2),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._inner_env.reset(task_name=self.task, seed=seed or self._seed)
        return self._to_gym_obs(obs), {}

    def step(self, action):
        obs = self._inner_env.step(IrrigationAction(action=int(action)))
        return self._to_gym_obs(obs), float(obs.reward or 0.0), bool(obs.done), False, {}

    def _to_gym_obs(self, obs):
        return {
            "soil_moisture": np.array([obs.soil_moisture], dtype=np.float32),
            "crop_growth": np.array([obs.crop_growth], dtype=np.float32),
            "power_status": int(obs.power_status),
        }
# gym_env/__init__.py
from gymnasium.envs.registration import register
register(id="PrecisionIrrigation-v0", entry_point="gym_env.precision_irrigation:PrecisionIrrigationEnv")
