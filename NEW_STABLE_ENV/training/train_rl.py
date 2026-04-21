# training/train_rl.py
import sys, os
sys.path.insert(0, os.getcwd())
import gymnasium as gym
import gym_env
import numpy as np

def train():
    env = gym.make("PrecisionIrrigation-easy-v0")
    print("Starting Training (Simple Check)...")
    for ep in range(3):
        obs, _ = env.reset()
        done = False
        total_r = 0
        while not done:
            action = env.action_space.sample()
            obs, r, done, _, _ = env.step(action)
            total_r += r
        print(f"Ep {ep} | Reward: {total_r:.2f}")
    print("Training Check Complete.")

if __name__ == "__main__":
    train()
