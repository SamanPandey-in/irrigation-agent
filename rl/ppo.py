"""
rl/ppo.py — Proximal Policy Optimisation in pure NumPy
=======================================================
Implements a minimal but complete PPO with:
  • MLP actor-critic (shared trunk)
  • GAE (λ-return) advantage estimation
  • Clipped surrogate objective
  • Value function loss with clipping
  • Entropy bonus
  • Mini-batch updates

No PyTorch, TensorFlow, or gymnasium required.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Activation functions
# ─────────────────────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Dense layer
# ─────────────────────────────────────────────────────────────────────────────

class Dense:
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", seed=None):
        rng   = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_dim)   # He init
        self.W  = rng.normal(0, scale, (in_dim, out_dim)).astype(np.float32)
        self.b  = np.zeros(out_dim, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.activation = activation
        self._x = None
        self._pre = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x   = x
        self._pre = x @ self.W + self.b
        if self.activation == "relu":
            return relu(self._pre)
        elif self.activation == "tanh":
            return tanh(self._pre)
        elif self.activation == "linear":
            return self._pre
        elif self.activation == "softmax":
            return softmax(self._pre)
        raise ValueError(self.activation)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            d_act = d_out * relu_grad(self._pre)
        elif self.activation == "tanh":
            d_act = d_out * tanh_grad(self._pre)
        elif self.activation == "linear":
            d_act = d_out
        elif self.activation == "softmax":
            # handled outside for cross-entropy shortcut
            d_act = d_out
        else:
            raise ValueError(self.activation)

        self.dW = self._x.T @ d_act / len(self._x)
        self.db = d_act.mean(axis=0)
        return d_act @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


# ─────────────────────────────────────────────────────────────────────────────
# Adam optimiser (stateful, per-parameter)
# ─────────────────────────────────────────────────────────────────────────────

class Adam:
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}
        self.t  = 0

    def step(self, params_grads: list[tuple[np.ndarray, np.ndarray]]):
        self.t += 1
        for (p, g) in params_grads:
            pid = id(p)
            if pid not in self._m:
                self._m[pid] = np.zeros_like(p)
                self._v[pid] = np.zeros_like(p)
            m = self._m[pid]
            v = self._v[pid]
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * g**2
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p[:] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic MLP
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic:
    """
    Shared trunk → policy head (actor) + value head (critic).
    Input: flat observation vector.
    Output: action logits, state value.
    """

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden: tuple[int, ...] = (128, 128), seed: int = 0):
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.hidden    = hidden

        # Shared trunk
        self.trunk: list[Dense] = []
        in_d = obs_dim
        for i, h in enumerate(hidden):
            self.trunk.append(Dense(in_d, h, activation="tanh", seed=seed + i))
            in_d = h

        # Heads
        self.actor_head  = Dense(in_d, n_actions, activation="linear", seed=seed + 99)
        self.critic_head = Dense(in_d, 1,          activation="linear", seed=seed + 100)

        self.optim = Adam(lr=3e-4)

    def _trunk_forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for layer in self.trunk:
            h = layer.forward(h)
        return h

    def forward(self, x: np.ndarray):
        h      = self._trunk_forward(x)
        logits = self.actor_head.forward(h)
        value  = self.critic_head.forward(h).squeeze(-1)
        return logits, value

    def get_action(self, obs_flat: np.ndarray) -> tuple[int, float, float]:
        """Sample action; return (action, log_prob, value)."""
        x       = obs_flat[np.newaxis]      # (1, obs_dim)
        logits, value = self.forward(x)
        probs   = softmax(logits[0])
        action  = int(np.random.choice(len(probs), p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob, float(value[0])

    def evaluate(self, obs_batch: np.ndarray, actions_batch: np.ndarray):
        """Return log_probs, values, entropy for a batch."""
        logits, values = self.forward(obs_batch)
        probs          = softmax(logits)
        log_probs_all  = np.log(probs + 1e-8)
        log_probs      = log_probs_all[np.arange(len(actions_batch)), actions_batch]
        entropy        = -(probs * log_probs_all).sum(axis=-1).mean()
        return log_probs, values, entropy

    def update(self, obs_b, actions_b, old_log_probs_b, returns_b, advantages_b,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        """Single mini-batch PPO update. Returns dict of losses."""

        # Forward
        logits, values = self.forward(obs_b)
        probs          = softmax(logits)
        log_probs_all  = np.log(probs + 1e-8)
        log_probs      = log_probs_all[np.arange(len(actions_b)), actions_b]
        entropy        = -(probs * log_probs_all).sum(axis=-1).mean()

        # Policy loss (clipped surrogate)
        ratio       = np.exp(log_probs - old_log_probs_b)
        surr1       = ratio * advantages_b
        surr2       = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_b
        policy_loss = -np.minimum(surr1, surr2).mean()

        # Value loss
        value_loss  = 0.5 * ((values - returns_b) ** 2).mean()

        # Total loss
        total_loss  = policy_loss + vf_coef * value_loss - ent_coef * entropy

        # ── Backprop ──────────────────────────────────────────────────────────
        N = len(obs_b)

        # --- Critic head gradient ---
        d_val = (values - returns_b) * vf_coef / N
        d_h_critic = self.critic_head.backward(d_val[:, np.newaxis])

        # --- Actor head gradient (policy + entropy) ---
        # d(policy_loss)/d(logits): use REINFORCE-style gradient
        # Clipping mask: zero gradient where ratio is outside [1-e, 1+e]
        clipped   = (ratio < (1 - clip_eps)) | (ratio > (1 + clip_eps))
        pg_weight = np.where(clipped, 0.0, advantages_b)

        # Policy gradient: d(policy_loss)/d(logits)
        # policy_loss = -mean[min(r*A, clip(r)*A)]
        # d(policy_loss)/d(logits) = A*(probs - e_a)/N
        d_logits  = probs.copy()
        d_logits[np.arange(N), actions_b] -= 1.0                  # probs - e_a
        d_logits  = d_logits * pg_weight[:, None] / N             # A*(probs-e_a)/N [FIXED sign]

        # Entropy: total_loss includes -ent_coef*entropy, so gradient = +ent_coef*(logp+1)*p/N
        d_ent     = (log_probs_all + 1) * probs / N * ent_coef    # [FIXED sign]
        d_logits += d_ent

        d_h_actor = self.actor_head.backward(d_logits)

        # --- Trunk gradient (sum from both heads) ---
        d_h = d_h_actor + d_h_critic
        for layer in reversed(self.trunk):
            d_h = layer.backward(d_h)

        # --- Collect all params & grads ---
        all_params: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in self.trunk:
            all_params.extend(layer.params())
        all_params.extend(self.actor_head.params())
        all_params.extend(self.critic_head.params())

        # Gradient clipping (global norm)
        grads     = [g for _, g in all_params]
        gnorm     = np.sqrt(sum((g**2).sum() for g in grads))
        if gnorm > max_grad_norm:
            scale = max_grad_norm / (gnorm + 1e-8)
            for g in grads:
                g *= scale

        self.optim.step(all_params)

        return {"policy_loss": float(policy_loss),
                "value_loss":  float(value_loss),
                "entropy":     float(entropy),
                "total_loss":  float(total_loss)}

    def save(self, path: str):
        state = {
            "obs_dim":   self.obs_dim,
            "n_actions": self.n_actions,
            "hidden":    self.hidden,
            "trunk":     [(l.W.copy(), l.b.copy(), l.activation) for l in self.trunk],
            "actor":     (self.actor_head.W.copy(), self.actor_head.b.copy()),
            "critic":    (self.critic_head.W.copy(), self.critic_head.b.copy()),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  💾  Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ActorCritic":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(state["obs_dim"], state["n_actions"], state["hidden"])
        for layer, (W, b, act) in zip(obj.trunk, state["trunk"]):
            layer.W[:] = W; layer.b[:] = b; layer.activation = act
        obj.actor_head.W[:],  obj.actor_head.b[:]  = state["actor"]
        obj.critic_head.W[:], obj.critic_head.b[:] = state["critic"]
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Observation flattener (handles dict obs)
# ─────────────────────────────────────────────────────────────────────────────

def flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten dict observation to 1-D float32 vector."""
    parts = []
    for key in sorted(obs.keys()):
        v = obs[key].astype(np.float32).ravel()
        parts.append(v)
    return np.concatenate(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.obs:          list[np.ndarray] = []
        self.actions:      list[int]        = []
        self.log_probs:    list[float]      = []
        self.rewards:      list[float]      = []
        self.values:       list[float]      = []
        self.dones:        list[bool]       = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def compute_gae(self, last_value: float, gamma=0.99, lam=0.95):
        """Compute GAE advantages and returns."""
        T          = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0
        vals       = self.values + [last_value]

        for t in reversed(range(T)):
            delta = self.rewards[t] + gamma * vals[t+1] * (1 - self.dones[t]) - vals[t]
            gae   = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns


# ─────────────────────────────────────────────────────────────────────────────
# PPO Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PPOTrainer:
    def __init__(
        self,
        env_factory,            # callable → env instance
        obs_dim: int,
        n_actions: int = 5,
        n_steps: int = 512,     # steps per rollout
        n_epochs: int = 6,      # PPO epochs per update
        batch_size: int = 64,
        gamma: float = 0.99,
        lam: float   = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float  = 0.5,
        ent_coef: float = 0.015,
        lr: float = 3e-4,
        seed: int = 0,
    ):
        self.env_factory = env_factory
        self.n_steps     = n_steps
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.gamma       = gamma
        self.lam         = lam
        self.clip_eps    = clip_eps
        self.vf_coef     = vf_coef
        self.ent_coef    = ent_coef

        self.model   = ActorCritic(obs_dim, n_actions, seed=seed)
        self.model.optim.lr = lr
        self.buffer  = RolloutBuffer()
        self.env     = env_factory(seed=seed)

        np.random.seed(seed)

        # Logging
        self.ep_rewards:  list[float] = []
        self.ep_lengths:  list[int]   = []
        self.update_losses: list[dict] = []
        self.ep_yields:   list[float] = []

    def _collect_rollout(self):
        """Collect n_steps of experience."""
        self.buffer.clear()
        obs, _ = self.env.reset()
        ep_r, ep_len = 0.0, 0

        for _ in range(self.n_steps):
            obs_flat             = flatten_obs(obs)
            action, log_prob, v  = self.model.get_action(obs_flat)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(obs_flat, action, log_prob, reward, v, done)
            obs   = next_obs
            ep_r += reward
            ep_len += 1

            if done:
                self.ep_rewards.append(ep_r)
                self.ep_lengths.append(ep_len)
                self.ep_yields.append(info.get("yield_t_ha", 0.0))
                obs, _  = self.env.reset()
                ep_r, ep_len = 0.0, 0

        # Bootstrap value for last state
        obs_flat    = flatten_obs(obs)
        _, last_val = self.model.forward(obs_flat[np.newaxis])
        return float(last_val[0])

    def _update(self, last_val: float):
        advantages, returns = self.buffer.compute_gae(
            last_val, gamma=self.gamma, lam=self.lam)

        obs_arr      = np.array(self.buffer.obs,       dtype=np.float32)
        actions_arr  = np.array(self.buffer.actions,   dtype=np.int32)
        old_lp_arr   = np.array(self.buffer.log_probs, dtype=np.float32)

        losses_ep = []
        for _ in range(self.n_epochs):
            idx = np.random.permutation(len(obs_arr))
            for start in range(0, len(idx), self.batch_size):
                mb = idx[start:start + self.batch_size]
                if len(mb) < 4:
                    continue
                loss = self.model.update(
                    obs_arr[mb], actions_arr[mb], old_lp_arr[mb],
                    returns[mb], advantages[mb],
                    clip_eps=self.clip_eps, vf_coef=self.vf_coef, ent_coef=self.ent_coef)
                losses_ep.append(loss)

        if losses_ep:
            avg = {k: float(np.mean([l[k] for l in losses_ep])) for k in losses_ep[0]}
            self.update_losses.append(avg)
            return avg
        return {}

    def train(self, total_steps: int = 60_000, log_every: int = 2048,
              save_path: Optional[str] = None):
        print(f"\n🚜  PPO Training | total_steps={total_steps:,} | "
              f"n_steps={self.n_steps} | epochs={self.n_epochs}")

        steps_done = 0
        n_updates  = 0
        t0         = time.time()

        while steps_done < total_steps:
            last_val = self._collect_rollout()
            losses   = self._update(last_val)
            steps_done += self.n_steps
            n_updates  += 1

            if steps_done % log_every < self.n_steps or steps_done >= total_steps:
                recent_r  = self.ep_rewards[-10:] if self.ep_rewards else [0]
                recent_y  = self.ep_yields[-10:]  if self.ep_yields  else [0]
                elapsed   = time.time() - t0
                sps       = steps_done / max(elapsed, 1)
                print(f"  step={steps_done:>7,}  "
                      f"eps={len(self.ep_rewards):>4}  "
                      f"r̄={np.mean(recent_r):>8.2f}  "
                      f"yield̄={np.mean(recent_y):>5.2f}t/ha  "
                      f"ent={losses.get('entropy',0):.3f}  "
                      f"{sps:.0f} sps")

        if save_path:
            self.model.save(save_path)

        elapsed = time.time() - t0
        print(f"\n✅  Training done — {steps_done:,} steps in {elapsed:.1f}s  "
              f"({steps_done/elapsed:.0f} sps)")
        return self.model

    def evaluate(self, n_episodes: int = 20, deterministic: bool = True) -> dict[str, float]:
        """Evaluate trained policy."""
        env = self.env_factory(seed=999)
        rewards, yields, water_used = [], [], []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep + 1000)
            ep_r   = 0.0
            while True:
                obs_flat = flatten_obs(obs)
                logits, _ = self.model.forward(obs_flat[np.newaxis])
                if deterministic:
                    action = int(np.argmax(logits[0]))
                else:
                    probs  = softmax(logits[0])
                    action = int(np.random.choice(len(probs), p=probs))
                obs, r, terminated, truncated, info = env.step(action)
                ep_r += r
                if terminated or truncated:
                    break
            rewards.append(ep_r)
            yields.append(info.get("yield_t_ha", 0))
            water_used.append(info.get("cum_water_kl", 0))

        env.close()
        return {
            "mean_reward":    float(np.mean(rewards)),
            "std_reward":     float(np.std(rewards)),
            "mean_yield":     float(np.mean(yields)),
            "mean_water_kl":  float(np.mean(water_used)),
            "efficiency":     float(np.mean(yields)) / max(float(np.mean(water_used)), 0.001),
        }
