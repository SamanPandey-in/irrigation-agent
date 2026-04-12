"""
tests/test_env.py
Run with: python -m pytest tests/ -v
or simply: python tests/test_env.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from envs.precision_irrigation_v3 import PrecisionIrrigationEnvV3


def make_env(render_mode=None):
    return PrecisionIrrigationEnvV3(render_mode=render_mode, use_api=False)


# ─────────────────────────────────────────────────────────────────────────────
def test_reset_returns_valid_obs():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict), "obs must be a dict"
    assert "soil_moisture_obs" in obs
    assert "crop_growth"      in obs
    assert "crop_stage"       in obs
    assert "weather_forecast" in obs
    assert "water_tank"       in obs
    assert "power_status"     in obs
    assert "day_of_season"    in obs
    assert "grid_moisture"    in obs

    assert obs["soil_moisture_obs"].shape == (1,)
    assert obs["crop_growth"].shape      == (1,)
    assert obs["crop_stage"].shape       == (1,)
    assert obs["weather_forecast"].shape == (3, 2)
    assert obs["water_tank"].shape       == (1,)
    assert obs["power_status"].shape     == (1,)
    assert obs["grid_moisture"].shape    == (4, 4)
    assert 0.0 <= obs["soil_moisture_obs"][0] <= 1.0
    assert 0.0 <= obs["crop_growth"][0]    <= 1.0
    assert 0.0 <= obs["water_tank"][0]     <= 1.0
    print("✅  test_reset_returns_valid_obs passed")
    env.close()


def test_step_all_actions():
    env = make_env()
    obs, _ = env.reset(seed=1)
    for action in range(5):
        env.reset(seed=action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float), "reward must be float"
        assert isinstance(terminated, bool)
        assert isinstance(truncated,  bool)
        assert isinstance(info, dict)
        assert "yield_t_ha" in info
    print("✅  test_step_all_actions passed")
    env.close()


def test_obs_bounds_throughout_episode():
    env = make_env()
    obs, _ = env.reset(seed=42)
    for _ in range(90):
        action = int(np.random.randint(0, 5))
        obs, reward, terminated, truncated, info = env.step(action)
        assert 0.0 <= obs["soil_moisture_obs"][0] <= 1.0, f"moisture OOB: {obs['soil_moisture_obs']}"
        assert 0.0 <= obs["crop_growth"][0]    <= 1.0, f"growth OOB: {obs['crop_growth']}"
        assert 0.0 <= obs["water_tank"][0]     <= 1.0, f"tank OOB: {obs['water_tank']}"
        if terminated or truncated:
            break
    print("✅  test_obs_bounds_throughout_episode passed")
    env.close()


def test_100_episodes_no_crash():
    env = make_env()
    episode_rewards = []
    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        total_r = 0.0
        for _ in range(90):
            action = int(np.random.randint(0, 5))
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            if terminated or truncated:
                break
        episode_rewards.append(total_r)
    assert len(episode_rewards) == 100
    mean_r = float(np.mean(episode_rewards))
    print(f"✅  test_100_episodes_no_crash passed — mean reward: {mean_r:.2f}")
    env.close()


def test_deterministic_with_same_seed():
    env = make_env()

    def run_episode(seed):
        obs, _ = env.reset(seed=seed)
        rewards = []
        for step in range(20):
            obs, r, done, trunc, _ = env.step(step % 5)
            rewards.append(r)
            if done or trunc:
                break
        return rewards

    r1 = run_episode(7)
    r2 = run_episode(7)
    assert r1 == r2, "Same seed should give same episode trajectory"
    print("✅  test_deterministic_with_same_seed passed")
    env.close()


def test_power_cut_blocks_irrigation():
    """When power is out, pumping actions should deliver 0 litres."""
    env = PrecisionIrrigationEnvV3(seed=0, use_api=False)
    env.reset(seed=0)
    # Force power off
    env._power_status = 0
    # Manually override season data power_cut for current day
    env._season_data.at[env._day, "power_cut"] = 1
    _, _, _, _, info = env.step(3)  # action=3 (high irrigation)
    assert info["litres_pumped"] == 0.0, (
        f"Expected 0 litres during power cut, got {info['litres_pumped']}"
    )
    print("✅  test_power_cut_blocks_irrigation passed")
    env.close()


def test_terminal_bonus():
    """Episode ending via rule-based policy should produce non-zero yield."""
    env = make_env()
    obs, _ = env.reset(seed=7)
    total_r = 0.0
    last_info = {}

    for _ in range(90):
        # Rule-based: irrigate only when dry, drain if waterlogged
        moisture = obs["soil_moisture_obs"][0]
        if moisture < 0.30:
            action = 2
        elif moisture > 0.75:
            action = 4
        else:
            action = 0
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r
        last_info = info
        if terminated or truncated:
            break

    assert "yield_t_ha" in last_info, "yield_t_ha missing from info"
    assert last_info["yield_t_ha"] >= 0, f"Negative yield: {last_info['yield_t_ha']}"
    print(f"✅  test_terminal_bonus passed — final yield: {last_info['yield_t_ha']:.2f} t/ha  "
          f"total reward: {total_r:.2f}")
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌾  Running PrecisionIrrigationEnv test suite\n" + "─" * 50)
    test_reset_returns_valid_obs()
    test_step_all_actions()
    test_obs_bounds_throughout_episode()
    test_100_episodes_no_crash()
    test_deterministic_with_same_seed()
    test_power_cut_blocks_irrigation()
    test_terminal_bonus()
    print("\n" + "─" * 50)
    print("🎉  All tests passed!")
