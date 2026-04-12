"""
scripts/train_and_eval.py — Phase 3 full training pipeline (v2)
===============================================================
Key fixes vs v1:
  • Hand-coded observation normalisation (obs scaled to ~[-1,1])
  • Entropy coefficient annealing (0.05 → 0.005) prevents premature collapse
  • Reward clipping [-10, 10] during training for stable gradients
  • Larger network (256×256) and more rollout steps
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.precision_irrigation_v3 import PrecisionIrrigationEnvV3
from rl.ppo import ActorCritic, RolloutBuffer, softmax
from rl.obs_normalizer import manual_normalize

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def make_env(seed=None, **_):
    return PrecisionIrrigationEnvV3(seed=seed, use_api=False)

def get_obs_dim():
    env = make_env(seed=0); obs, _ = env.reset(seed=0)
    d = len(manual_normalize(obs)); env.close(); return d


# ─────────────────────────────────────────────────────────────────────────────
# Baseline policies
# ─────────────────────────────────────────────────────────────────────────────

def random_policy(obs, _=None):
    return int(np.random.randint(0, 5))

def rule_based_policy(obs, _=None):
    m = float(obs["soil_moisture_obs"][0])
    if m < 0.28: return 3
    if m < 0.40: return 2
    if m > 0.78: return 4
    return 0

def greedy_policy(obs, _=None):
    return 3

def smart_forecast_policy(obs, _=None):
    m        = float(obs["soil_moisture_obs"][0])
    rain_max = float(obs["weather_forecast"][:2, 0].max())
    stage    = int(obs["crop_stage"][0])
    dry_thr  = [0.35, 0.40, 0.45, 0.30][stage]
    wet_thr  = [0.70, 0.75, 0.70, 0.65][stage]
    if rain_max > 8:              return 0
    if rain_max > 4:              return 1 if m < dry_thr else 0
    if m < dry_thr - 0.10:       return 3
    if m < dry_thr:               return 2
    if m > wet_thr:               return 4
    return 0

def ppo_policy(obs, model):
    obs_n     = manual_normalize(obs)[np.newaxis]
    logits, _ = model.forward(obs_n)
    return int(np.argmax(logits[0]))


# ─────────────────────────────────────────────────────────────────────────────
# Improved PPO trainer
# ─────────────────────────────────────────────────────────────────────────────

class ImprovedPPOTrainer:
    def __init__(self, obs_dim, n_actions=5, n_steps=512, n_epochs=8,
                 batch_size=64, gamma=0.99, lam=0.95, clip_eps=0.2,
                 vf_coef=0.5, ent_start=0.05, ent_end=0.005, lr=2e-4, seed=42):

        self.n_steps   = n_steps
        self.n_epochs  = n_epochs
        self.batch_size= batch_size
        self.gamma     = gamma
        self.lam       = lam
        self.clip_eps  = clip_eps
        self.vf_coef   = vf_coef
        self.ent_start = ent_start
        self.ent_end   = ent_end

        self.model  = ActorCritic(obs_dim, n_actions, hidden=(256, 256), seed=seed)
        self.model.optim.lr = lr
        self.buffer = RolloutBuffer()
        self.env    = make_env(seed=seed)
        np.random.seed(seed)

        self.ep_rewards: list[float] = []
        self.ep_yields:  list[float] = []
        self.ep_lengths: list[int]   = []
        self.update_losses: list[dict] = []

    def _ent_coef(self, frac):
        return self.ent_start + frac * (self.ent_end - self.ent_start)

    def _collect(self):
        self.buffer.clear()
        obs, _ = self.env.reset()
        ep_r, ep_len = 0.0, 0
        for _ in range(self.n_steps):
            obs_n        = manual_normalize(obs)[np.newaxis]
            logits, v    = self.model.forward(obs_n)
            probs        = softmax(logits[0])
            action       = int(np.random.choice(5, p=probs))
            log_prob     = float(np.log(probs[action] + 1e-8))
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done         = terminated or truncated
            self.buffer.add(manual_normalize(obs), action, log_prob,
                            float(np.clip(reward, -10, 10)), float(v[0]), done)
            obs   = next_obs
            ep_r += reward
            ep_len += 1
            if done:
                self.ep_rewards.append(ep_r / max(ep_len, 1))
                self.ep_yields.append(info.get("yield_t_ha", 0))
                self.ep_lengths.append(ep_len)
                obs, _ = self.env.reset()
                ep_r, ep_len = 0.0, 0
        obs_n  = manual_normalize(obs)[np.newaxis]
        _, v   = self.model.forward(obs_n)
        return float(v[0])

    def _update(self, last_val, ent_coef):
        advantages, returns = self.buffer.compute_gae(last_val, self.gamma, self.lam)
        obs_arr = np.array(self.buffer.obs,       dtype=np.float32)
        act_arr = np.array(self.buffer.actions,   dtype=np.int32)
        olp_arr = np.array(self.buffer.log_probs, dtype=np.float32)
        ls = []
        for _ in range(self.n_epochs):
            idx = np.random.permutation(len(obs_arr))
            for s in range(0, len(idx), self.batch_size):
                mb = idx[s:s+self.batch_size]
                if len(mb) < 4: continue
                ls.append(self.model.update(
                    obs_arr[mb], act_arr[mb], olp_arr[mb],
                    returns[mb], advantages[mb],
                    clip_eps=self.clip_eps, vf_coef=self.vf_coef, ent_coef=ent_coef))
        if ls:
            avg = {k: float(np.mean([x[k] for x in ls])) for k in ls[0]}
            self.update_losses.append(avg)
            return avg
        return {}

    def train(self, total_steps=80_000, log_every=4096, save_path=None):
        print(f"\n🚜  PPO  steps={total_steps:,}  n_steps={self.n_steps}  "
              f"epochs={self.n_epochs}  ent={self.ent_start}→{self.ent_end}  lr={self.model.optim.lr}")
        steps_done, t0 = 0, time.time()
        while steps_done < total_steps:
            frac     = steps_done / total_steps
            last_val = self._collect()
            losses   = self._update(last_val, self._ent_coef(frac))
            steps_done += self.n_steps
            if steps_done % log_every < self.n_steps or steps_done >= total_steps:
                rec_r = self.ep_rewards[-10:] if self.ep_rewards else [0]
                rec_y = self.ep_yields[-10:]  if self.ep_yields  else [0]
                sps   = steps_done / max(time.time()-t0, 1)
                print(f"  step={steps_done:>7,}  eps={len(self.ep_rewards):>4}  "
                      f"r̄={np.mean(rec_r):>+9.2f}  yield̄={np.mean(rec_y):>5.2f}t  "
                      f"ent={losses.get('entropy',0):.3f}  ec={self._ent_coef(frac):.4f}  {sps:.0f}sps")
        if save_path: self.model.save(save_path)
        print(f"\n✅  Done — {steps_done:,} steps in {time.time()-t0:.1f}s")
        return self.model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(name, policy_fn, model=None, n_episodes=25,
                    drought_mode=False, no_forecast=False):
    rewards, yields, water, stress = [], [], [], []
    for ep in range(n_episodes):
        env = make_env(seed=ep+5000)
        obs, _ = env.reset(seed=ep+5000)
        if drought_mode:
            env._season_data = env._season_data.copy()
            env._season_data["rainfall_mm"] *= 0.25
        ep_r = 0.0
        ep_len = 0
        while True:
            if no_forecast:
                obs["weather_forecast"] = np.zeros((3,2), dtype=np.float32)
            obs, r, terminated, truncated, info = env.step(policy_fn(obs, model))
            ep_r += r
            ep_len += 1
            if terminated or truncated: break
        ep_reward_mean = ep_r / max(ep_len, 1)
        rewards.append(float(ep_reward_mean)); yields.append(info.get("yield_t_ha",0))
        water.append(info.get("cum_water_kl",0)); stress.append(info.get("stress_days",0))
        env.close()
    return {"name": name,
            "mean_reward":     float(np.mean(rewards)),
            "std_reward":      float(np.std(rewards)),
            "mean_yield":      float(np.mean(yields)),
            "mean_water_kl":   float(np.mean(water)),
            "mean_stress_d":   float(np.mean(stress)),
            "mean_efficiency": float(np.mean(yields)) / max(float(np.mean(water)), 0.001),
            "raw_rewards":     rewards,
            "raw_yields":      yields}


def print_table(results, title=""):
    print(f"\n{'═'*100}\n  {title}\n{'─'*100}")
    print(f"{'Agent':<26} {'Reward':>10} {'±':>6} {'Yield t/ha':>12} {'Water kL':>10} {'Stress d':>10} {'Efficiency':>12}")
    print("─"*100)
    for r in results:
        tag = " ★" if r["name"] == "PPO (trained)" else ""
        print(f"{r['name']+tag:<26} {r['mean_reward']:>+10.2f} {r['std_reward']:>6.2f} "
              f"{r['mean_yield']:>12.2f} {r['mean_water_kl']:>10.1f} "
              f"{r['mean_stress_d']:>10.1f} {r['mean_efficiency']:>12.4f}")
    print("═"*100)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(
    total_steps=80_000,
    n_eval=25,
    quick=False,
    quick_steps=25_000,
    quick_eval=10,
):
    if quick:
        total_steps, n_eval = int(quick_steps), int(quick_eval)

    dim = get_obs_dim()
    print(f"\n🌾  Precision Irrigation — All Phases Pipeline")
    print(f"   obs_dim={dim}  actions=5  steps={total_steps:,}  eval_eps={n_eval}")

    trainer    = ImprovedPPOTrainer(obs_dim=dim, seed=42)
    model_path = os.path.join(MODELS_DIR, "ppo_irrigation.pkl")
    trainer.train(total_steps=total_steps, log_every=4096, save_path=model_path)
    ppo_model  = ActorCritic.load(model_path)

    print(f"\n📊  Evaluating ({n_eval} eps each)...")
    main_results = []
    for name, fn, mdl in [
        ("Random",         random_policy,        None),
        ("Rule-Based",     rule_based_policy,     None),
        ("Greedy",         greedy_policy,         None),
        ("Smart Forecast", smart_forecast_policy, None),
        ("PPO (trained)",  ppo_policy,            ppo_model),
    ]:
        print(f"  ▸ {name}...", end=" ", flush=True)
        r = evaluate_policy(name, fn, model=mdl, n_episodes=n_eval)
        main_results.append(r)
        print(f"r={r['mean_reward']:+.2f}  yield={r['mean_yield']:.2f}t  water={r['mean_water_kl']:.1f}kL")

    print(f"\n🔬  Ablations...")
    ablation_results = []
    for name, fn, mdl, kw in [
        ("PPO — no forecast",    ppo_policy,            ppo_model, {"no_forecast":  True}),
        ("PPO — drought",        ppo_policy,            ppo_model, {"drought_mode": True}),
        ("Forecast — drought",   smart_forecast_policy, None,      {"drought_mode": True}),
        ("Rule-Based — drought", rule_based_policy,     None,      {"drought_mode": True}),
    ]:
        print(f"  ▸ {name}...", end=" ", flush=True)
        r = evaluate_policy(name, fn, model=mdl, n_episodes=n_eval, **kw)
        ablation_results.append(r)
        print(f"r={r['mean_reward']:+.2f}  yield={r['mean_yield']:.2f}t")

    # Save
    metrics_path = os.path.join(OUTPUTS_DIR, "metrics.json")
    training_log = {"ep_rewards": trainer.ep_rewards, "ep_yields": trainer.ep_yields,
                    "ep_lengths": trainer.ep_lengths, "update_losses": trainer.update_losses,
                    "total_steps": total_steps}
    with open(metrics_path, "w") as f:
        json.dump({"results": main_results + ablation_results, "training": training_log}, f, indent=2)
    print(f"\n💾  Metrics → {metrics_path}")

    print_table(main_results,     "Main Agents")
    print_table(ablation_results, "Ablation Studies")

    rb  = next((r for r in main_results if r["name"] == "Rule-Based"), None)
    ppo = next((r for r in main_results if r["name"] == "PPO (trained)"), None)
    if rb and ppo and rb["mean_yield"] > 0.1:
        yg = (ppo["mean_yield"] - rb["mean_yield"]) / rb["mean_yield"] * 100
        ws = (rb["mean_water_kl"] - ppo["mean_water_kl"]) / max(rb["mean_water_kl"], 0.001) * 100
        print(f"\n🏆  PPO vs Rule-Based: yield Δ{yg:+.1f}%   water Δ{ws:+.1f}%")

    return main_results, ablation_results, training_log, ppo_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80_000)
    parser.add_argument("--eval",  type=int, default=25)
    parser.add_argument("--quick", action="store_true",
                        help="Run lightweight startup training (defaults: 25k steps, 10 eval episodes).")
    parser.add_argument("--quick-steps", type=int, default=25_000,
                        help="Training steps to use when --quick is enabled.")
    parser.add_argument("--quick-eval", type=int, default=10,
                        help="Eval episodes to use when --quick is enabled.")
    parser.add_argument("--fast", action="store_true",
                        help="Alias for --quick (kept for backwards compatibility).")
    args = parser.parse_args()
    quick_mode = bool(args.quick or args.fast)
    main(
        total_steps=args.steps,
        n_eval=args.eval,
        quick=quick_mode,
        quick_steps=args.quick_steps,
        quick_eval=args.quick_eval,
    )
