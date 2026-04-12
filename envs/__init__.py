"""
Registers PrecisionIrrigation-v0 with Gymnasium.
Import this module (or `import envs`) before calling gym.make().
Safe-guards against gymnasium not being installed.
"""
try:
    from gymnasium.envs.registration import register
    register(
        id="PrecisionIrrigation-v0",
        entry_point="envs.precision_irrigation:PrecisionIrrigationEnv",
        max_episode_steps=90,
        reward_threshold=25.0,
    )
except ImportError:
    pass  # gymnasium not installed; direct import still works
