"""
demo.py — Phase 1 smoke-test & baseline comparison
---------------------------------------------------
Runs three baseline agents for one episode each and prints a metrics table.
Saves a PNG render of the PPO/trained-like "always medium water" agent.

Usage:
    python demo.py
    python demo.py --render        # shows live matplotlib window
    python demo.py --episodes 5    # run 5 episodes per agent
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ── Try gymnasium path, fall back to direct import ──────────────────────────
try:
    import gymnasium as gym
    import envs
    def make_env(render_mode=None):
        return gym.make("PrecisionIrrigation-v0", render_mode=render_mode)
    GYM_OK = True
except ImportError:
    from envs.precision_irrigation import PrecisionIrrigationEnv
    def make_env(render_mode=None):
        return PrecisionIrrigationEnv(render_mode=render_mode)
    GYM_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Agents
# ─────────────────────────────────────────────────────────────────────────────

def random_agent(obs):
    """Picks a random action each step."""
    return int(np.random.randint(0, 5))


def rule_based_agent(obs):
    """
    Simple threshold rule:
      - moisture < 0.30  → irrigate medium (2)
      - moisture > 0.75  → drain (4)
      - else             → no action (0)
    """
    moisture = obs["soil_moisture"][0]
    if moisture < 0.30:
        return 2      # medium irrigation
    elif moisture > 0.75:
        return 4      # drain
    return 0          # do nothing


def greedy_irrigate_agent(obs):
    """Always applies max irrigation — naive greedy baseline."""
    return 3


def smart_forecast_agent(obs):
    """
    Uses 3-day forecast:
      - If rain expected (>5mm in next 2 days) → no action
      - If moisture < 0.35 → high (3)
      - If moisture < 0.55 → medium (2)
      - Else → low (1) for maintenance
    """
    moisture  = obs["soil_moisture"][0]
    forecast  = obs["weather_forecast"]  # shape (3, 2): [[rain, temp], ...]
    rain_soon = forecast[:2, 0].max()    # max rain in next 2 days

    if rain_soon > 5.0:
        return 0   # rain coming — don't waste water
    if moisture < 0.35:
        return 3
    if moisture < 0.55:
        return 2
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(name, policy_fn, n_episodes=3, render_mode=None, seed_offset=0):
    env = make_env(render_mode=render_mode)
    total_rewards, yields, water_used_list = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + seed_offset)
        ep_reward = 0.0
        ep_water  = 0.0
        last_info = {}

        while True:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_water  += info.get("litres_pumped", 0.0)
            last_info  = info
            if terminated or truncated:
                break

        total_rewards.append(ep_reward)
        yields.append(last_info.get("yield_t_ha", 0.0))
        water_used_list.append(ep_water)

    env.close()
    return {
        "name":       name,
        "avg_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "avg_yield":  float(np.mean(yields)),
        "avg_water":  float(np.mean(water_used_list)),
        "efficiency": float(np.mean(yields) / (np.mean(water_used_list) / 1000 + 1e-9)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1 demo for Precision Irrigation Agent")
    parser.add_argument("--render",   action="store_true",  help="Show matplotlib render")
    parser.add_argument("--episodes", type=int, default=3,  help="Episodes per agent")
    parser.add_argument("--save-png", action="store_true",  help="Save final render as PNG")
    args = parser.parse_args()

    print("\n🌾  Precision Irrigation Agent — Phase 1 Demo")
    print(f"   {'Gymnasium' if GYM_OK else 'Direct import'} | {args.episodes} eps/agent\n")

    agents = [
        ("Random",           random_agent),
        ("Rule-Based",       rule_based_agent),
        ("Greedy (always high)", greedy_irrigate_agent),
        ("Smart Forecast",   smart_forecast_agent),
    ]

    results = []
    for name, fn in agents:
        print(f"  Evaluating {name}...", end=" ", flush=True)
        r = evaluate_agent(name, fn, n_episodes=args.episodes)
        results.append(r)
        print(f"reward={r['avg_reward']:.2f}  yield={r['avg_yield']:.2f}t/ha  "
              f"water={r['avg_water']/1000:.1f}k L")

    # ── Pretty table ────────────────────────────────────────────────────────
    col_w = [22, 12, 12, 12, 14, 14]
    headers = ["Agent", "Avg Reward", "Std Reward", "Yield t/ha", "Water (kL)", "Efficiency"]
    sep  = "─" * sum(col_w)
    fmt  = "".join(f"{{:<{w}}}" for w in col_w)

    print(f"\n{'─'*sum(col_w)}")
    print(fmt.format(*headers))
    print(sep)
    for r in results:
        print(fmt.format(
            r["name"][:21],
            f"{r['avg_reward']:>8.2f}",
            f"±{r['std_reward']:>6.2f}",
            f"{r['avg_yield']:>8.2f}",
            f"{r['avg_water']/1000:>8.1f}",
            f"{r['efficiency']:>10.4f}",
        ))
    print(sep)

    best = max(results, key=lambda x: x["avg_reward"])
    print(f"\n🏆  Best agent: {best['name']} (reward={best['avg_reward']:.2f})\n")

    # ── Optional: save a render PNG of the smart forecast agent ─────────────
    if args.save_png or args.render:
        render_mode = "human" if args.render else "rgb_array"
        env = make_env(render_mode=render_mode)
        obs, _ = env.reset(seed=7)
        while True:
            obs, _, terminated, truncated, _ = env.step(smart_forecast_agent(obs))
            if terminated or truncated:
                break
        frame = env.render()
        if frame is not None and not args.render:
            try:
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                out_path = os.path.join(os.path.dirname(__file__), "render_output.png")
                plt.imsave(out_path, frame)
                print(f"📸  Render saved → {out_path}")
            except ImportError:
                pass
        env.close()

    print("✅  Phase 1 complete — env is working.\n")
    print("Next steps:")
    print("  Phase 2 → Add POMDP partial obs, stochastic monsoon realism")
    print("  Phase 3 → Train with Stable-Baselines3 PPO/DQN")
    print("  Phase 4 → Docker + FastAPI mock IMD weather service")


if __name__ == "__main__":
    main()
