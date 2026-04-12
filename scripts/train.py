"""
scripts/train.py — Phase 3 RL training scaffold
-------------------------------------------------
Trains PPO on PrecisionIrrigation-v0 using Stable-Baselines3.

Usage:
    python scripts/train.py
    python scripts/train.py --timesteps 200000 --algo dqn

Requires:
    pip install stable-baselines3[extra] tensorboard
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import gymnasium as gym
    import envs  # noqa: F401 — triggers registration
except ImportError:
    print("❌  gymnasium not installed. Run: pip install gymnasium")
    sys.exit(1)

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
except ImportError:
    print("❌  stable-baselines3 not installed. Run: pip install stable-baselines3[extra]")
    sys.exit(1)


ALGO_MAP = {"ppo": PPO, "dqn": DQN}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOGS_DIR   = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)


def train(algo_name: str = "ppo", total_timesteps: int = 100_000, n_envs: int = 4):
    AlgoCls = ALGO_MAP[algo_name.lower()]

    print(f"\n🚜  Training {algo_name.upper()} on PrecisionIrrigation-v0")
    print(f"   timesteps={total_timesteps:,}  parallel_envs={n_envs}\n")

    # Vectorised environments for faster training
    vec_env  = make_vec_env("PrecisionIrrigation-v0", n_envs=n_envs, seed=0)
    eval_env = gym.make("PrecisionIrrigation-v0")

    # Stop early if reward threshold reached
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=25.0, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=LOGS_DIR,
        eval_freq=max(total_timesteps // 20, 1000),
        n_eval_episodes=10,
        callback_on_new_best=stop_cb,
        verbose=1,
    )

    model_kwargs: dict = {"verbose": 1, "tensorboard_log": LOGS_DIR}
    if algo_name == "ppo":
        model_kwargs.update({"n_steps": 256, "batch_size": 64, "learning_rate": 3e-4})

    model = AlgoCls("MultiInputPolicy", vec_env, **model_kwargs)
    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=True)

    save_path = os.path.join(MODELS_DIR, f"{algo_name}_irrigation_final")
    model.save(save_path)
    print(f"\n💾  Model saved → {save_path}.zip")

    # Final evaluation
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"📊  Final eval — mean reward: {mean_r:.2f} ± {std_r:.2f}")

    vec_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",       default="ppo",     choices=["ppo", "dqn"])
    parser.add_argument("--timesteps",  default=100_000,   type=int)
    parser.add_argument("--n-envs",     default=4,         type=int)
    args = parser.parse_args()

    train(args.algo, args.timesteps, args.n_envs)
