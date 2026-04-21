# gym_env/__init__.py
from gymnasium.envs.registration import register
register(id="PrecisionIrrigation-v0", entry_point="gym_env.precision_irrigation:PrecisionIrrigationEnv")
register(id="PrecisionIrrigation-easy-v0", entry_point="gym_env.precision_irrigation:PrecisionIrrigationEnv", kwargs={"task": "easy"})
