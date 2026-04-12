"""
Creates a minimal untrained PPO model so the server starts instantly.
app.py retrains it properly in a background thread after startup.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.ppo import ActorCritic

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

model_path = os.path.join(MODELS_DIR, "ppo_irrigation.pkl")
flag_path  = os.path.join(MODELS_DIR, ".needs_training")

ActorCritic(28, 5).save(model_path)
open(flag_path, "w").close()

print(f"Stub model created at {model_path}")
print(f"Training flag created at {flag_path}")