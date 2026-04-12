"""
app.py — Precision Irrigation RL Agent  |  OpenEnv Hackathon
=============================================================
Gradio UI for Hugging Face Spaces.

Tabs:
  ⚙️  Env Config        — tune every physics/reward constant, pick presets, export JSON
  🎮  Play as Agent     — interactive step-by-step human play using the active config
  ⌨️  Terminal Eval     — paste comma-separated actions, get scored
  📊  Benchmark         — compare you vs PPO vs baselines under the same config
  📈  Training Metrics  — view pre-trained PPO training curve + agent comparison
  ℹ️  About             — architecture, OpenEnv alignment, deploy guide
"""
from __future__ import annotations

import copy, json, os, sys, pickle, subprocess, time
import asyncio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(__file__))

from envs.env_config      import EnvConfig
from envs.configurable_env import make_env
from rl.obs_normalizer    import manual_normalize
from rl.ppo               import ActorCritic, softmax
from evaluator.programmatic_grader import ProgrammaticGrader
from evaluator.llm_grader          import LLMGrader

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(__file__)
MODELS_DIR  = os.path.join(ROOT, "models")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
DATA_DIR    = os.path.join(ROOT, "data")
MODEL_PATH  = os.path.join(MODELS_DIR, "ppo_irrigation.pkl")
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "train_and_eval.py")
UI_MOUNT_PATH = "/gradio"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

AUTO_TRAIN_STEPS = int(os.getenv("AUTO_TRAIN_STEPS", "25000"))
AUTO_TRAIN_EVAL = int(os.getenv("AUTO_TRAIN_EVAL", "10"))
AUTO_TRAIN_TIMEOUT = int(os.getenv("AUTO_TRAIN_TIMEOUT", "3600"))

# ── Colour palette ─────────────────────────────────────────────────────────────
BG, PANEL, BORDER = "#0d1117", "#161b22", "#30363d"
TEXT, MUTED       = "#c9d1d9", "#8b949e"
GREEN, BLUE, ORANGE, RED, PURPLE = "#3fb950","#58a6ff","#e3b341","#f85149","#bc8cff"

ACTION_LABELS = ["0 – No Water", "1 – Low (1kL)", "2 – Medium (3kL)", "3 – High (5kL)", "4 – Drain"]
STAGE_NAMES   = {0:"🌱 Seedling", 1:"🌿 Vegetative", 2:"🌾 Reproductive", 3:"🍂 Maturity"}
WATER_VOLS    = [0, 1000, 3000, 5000, 0]

# ── Load trained PPO ───────────────────────────────────────────────────────────
_ppo_model: ActorCritic | None = None

def _load_policy(path: str) -> ActorCritic:
    try:
        return ActorCritic.load(path)
    except Exception:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, ActorCritic):
            return obj
        raise

def _model_ready(path: str = MODEL_PATH) -> bool:
    if not os.path.exists(path):
        return False
    try:
        _load_policy(path)
        return True
    except Exception:
        return False

def _ensure_ppo_model() -> None:
    if _model_ready():
        return
    print(f"[startup] PPO model missing/unreadable at {MODEL_PATH}. Running quick training...", flush=True)
    try:
        result = subprocess.run(
            [
                sys.executable,
                TRAIN_SCRIPT,
                "--quick",
                "--quick-steps",
                str(AUTO_TRAIN_STEPS),
                "--quick-eval",
                str(AUTO_TRAIN_EVAL),
            ],
            cwd=ROOT,
            check=False,
            timeout=AUTO_TRAIN_TIMEOUT,
        )
        if result.returncode != 0:
            print(f"[startup] WARNING: training script exited {result.returncode} — continuing without PPO model", flush=True)
            return
    except subprocess.TimeoutExpired:
        print(f"[startup] WARNING: training timed out after {AUTO_TRAIN_TIMEOUT}s — continuing without PPO model", flush=True)
        return
    except Exception as e:
        print(f"[startup] WARNING: training failed ({e}) — continuing without PPO model", flush=True)
        return

    if not _model_ready():
        print(f"[startup] WARNING: training finished but no model at {MODEL_PATH} — continuing without it", flush=True)

def _load_ppo():
    global _ppo_model
    if _ppo_model is not None:
        return _ppo_model
    if os.path.exists(MODEL_PATH):
        try:
            _ppo_model = _load_policy(MODEL_PATH)
        except Exception as e:
            print(f"[startup] Could not load PPO model at {MODEL_PATH}: {e}")
    return _ppo_model

def _ppo_action(obs: dict) -> int:
    m = _load_ppo()
    if m is None:
        return _rule_based_action(obs)
    logits, _ = m.forward(manual_normalize(obs)[np.newaxis])
    return int(np.argmax(softmax(logits[0])))

def _rule_based_action(obs: dict) -> int:
    m = float(obs["soil_moisture_obs"][0])
    if m < 0.28: return 3
    if m < 0.40: return 2
    if m > 0.78: return 4
    return 0

# ── API Components ─────────────────────────────────────────────────────────────
app = FastAPI(title="Precision Irrigation - OpenEnv API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_1"
    seed: Optional[int] = None

class ActionRequest(BaseModel):
    action: int

_api_env = None
_api_history = []
_api_last_obs = None
_api_lock = asyncio.Lock()


def _to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value

def _get_api_env():
    global _api_env
    if _api_env is None:
        _api_env = make_env()
    return _api_env

@app.post("/reset")
async def reset_env(req: Optional[ResetRequest] = None):
    global _api_env, _api_history, _api_last_obs

    async with _api_lock:
        # Map task_id to presets
        preset = "🌦️ Normal"
        if req:
            if req.task_id == "task_2": preset = "🏜️ Drought"
            elif req.task_id == "task_3": preset = "🌊 Flood"

        cfg = EnvConfig.from_preset(preset)
        # Re-create env for each reset so task preset and seed are consistently applied.
        _api_env = make_env(cfg, seed=req.seed if req else None)
        obs, info = _api_env.reset(seed=req.seed if req else None)
        _api_history = []
        _api_last_obs = obs

        return {"observation": _to_builtin(obs), "info": _to_builtin(info)}

@app.post("/step")
async def step_env(req: ActionRequest):
    global _api_history, _api_last_obs

    async with _api_lock:
        env = _get_api_env()
        obs, reward, terminated, truncated, info = env.step(req.action)
        _api_last_obs = obs

        # Store history for grading (simplified)
        step_record = {
            "day": int(info.get("day", getattr(env, "_day", 0))),
            "action": int(req.action),
            "moisture": float(obs["soil_moisture_obs"][0]),
            "crop_growth": float(obs["crop_growth"][0]),
            "crop_stage": int(obs["crop_stage"][0]),
            "rainfall_mm": float(info.get("rain_mm", 0.0)),
            "reward": float(reward),
            "tank_level": float(obs["water_tank"][0]),
            "water_used_L": float(info.get("litres", info.get("litres_pumped", 0.0))),
        }
        _api_history.append(step_record)

        response = {
            "observation": _to_builtin(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": _to_builtin(info)
        }

        if terminated or truncated:
            grader = ProgrammaticGrader(_api_history)
            result = grader.evaluate()
            # Scale scores to 0-1 as required
            response["score"] = float(result["scores"]["overall"] / 100.0)
            response["results"] = result

        return response

@app.get("/state")
async def get_state():
    global _api_last_obs

    async with _api_lock:
        env = _get_api_env()
        if getattr(env, "_season_data", None) is None or len(env._season_data) == 0:
            obs, _ = env.reset()
            _api_last_obs = obs
        obs = _api_last_obs if _api_last_obs is not None else env._build_obs()
        return {"observation": _to_builtin(obs)}

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "time": time.time()}


@app.get("/")
async def ui_root():
    return RedirectResponse(url=f"{UI_MOUNT_PATH}/", status_code=307)


@app.get(UI_MOUNT_PATH)
async def ui_root_slash():
    return RedirectResponse(url=f"{UI_MOUNT_PATH}/", status_code=307)


# ══════════════════════════════════════════════════════════════════════════════
# Config helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cfg_from_sliders(
    # Episode
    max_steps, grid_size,
    # Tank
    tank_capacity, pump_efficiency, canal_refill_rate,
    # Action volumes
    vol_low, vol_medium, vol_high, vol_drain,
    # Moisture thresholds
    opt_lo, opt_hi, dry_thr, wet_thr,
    # Weather
    et_coeff, rain_scale, rain_multiplier,
    flash_flood_thr, sensor_noise, forecast_noise, power_cut_prob,
    # Reward weights
    w_growth, w_optimal, w_water, w_stress, w_overwet, w_terminal,
    # Crop growth rates
    rate_seed, rate_veg, rate_repro, rate_mat, stress_penalty,
    # Init conditions
    init_m_lo, init_m_hi, init_t_lo, init_t_hi,
) -> EnvConfig:
    cfg = EnvConfig()

    # Episode
    cfg.episode.max_steps = int(max_steps)
    cfg.episode.grid_size = int(grid_size)
    cfg.episode.tank_capacity = float(tank_capacity)

    # Physics
    cfg.physics.pump_efficiency = float(pump_efficiency)
    cfg.physics.tank_refill_rate = float(canal_refill_rate)
    cfg.physics.et_coeff = float(et_coeff)
    cfg.physics.rain_scale = float(rain_scale) * float(rain_multiplier)
    cfg.physics.flash_flood_thr = float(flash_flood_thr)
    cfg.physics.power_cut_prob = float(power_cut_prob)

    # Actions (drain stored as negative volume)
    cfg.actions.vol_low = float(vol_low)
    cfg.actions.vol_medium = float(vol_medium)
    cfg.actions.vol_high = float(vol_high)
    cfg.actions.vol_drain = -abs(float(vol_drain))

    # Moisture + reward
    cfg.moisture.opt_lo = float(opt_lo)
    cfg.moisture.opt_hi = float(opt_hi)
    cfg.moisture.dry_thr = float(dry_thr)
    cfg.moisture.wet_thr = float(wet_thr)
    cfg.reward.w_growth = float(w_growth)
    cfg.reward.w_optimal = float(w_optimal)
    cfg.reward.w_water = float(w_water)
    cfg.reward.w_stress = float(w_stress)
    cfg.reward.w_overwet = float(w_overwet)
    cfg.reward.w_terminal = float(w_terminal)
    cfg.reward.stress_growth_penalty = float(stress_penalty)

    # Sensor noise (forecast noise currently fixed in env)
    cfg.sensor_noise_std = float(sensor_noise)

    return cfg

def _all_slider_defaults():
    cfg = EnvConfig()
    return [
        cfg.episode.max_steps, cfg.episode.grid_size,
        cfg.episode.tank_capacity, cfg.physics.pump_efficiency, cfg.physics.tank_refill_rate,
        cfg.actions.vol_low, cfg.actions.vol_medium, cfg.actions.vol_high, abs(cfg.actions.vol_drain),
        cfg.moisture.opt_lo, cfg.moisture.opt_hi, cfg.moisture.dry_thr, cfg.moisture.wet_thr,
        cfg.physics.et_coeff, cfg.physics.rain_scale, 1.0,
        cfg.physics.flash_flood_thr, cfg.sensor_noise_std, 1.5, cfg.physics.power_cut_prob,
        cfg.reward.w_growth, cfg.reward.w_optimal, cfg.reward.w_water, cfg.reward.w_stress, cfg.reward.w_overwet, cfg.reward.w_terminal,
        0.014, 0.022,
        0.018, 0.010, cfg.reward.stress_growth_penalty,
        0.40, 0.60, 0.50, 0.80,
    ]

def _preset_to_slider_values(preset_name: str):
    cfg = EnvConfig.from_preset(preset_name)
    return [
        cfg.episode.max_steps, cfg.episode.grid_size,
        cfg.episode.tank_capacity, cfg.physics.pump_efficiency, cfg.physics.tank_refill_rate,
        cfg.actions.vol_low, cfg.actions.vol_medium, cfg.actions.vol_high, abs(cfg.actions.vol_drain),
        cfg.moisture.opt_lo, cfg.moisture.opt_hi, cfg.moisture.dry_thr, cfg.moisture.wet_thr,
        cfg.physics.et_coeff, cfg.physics.rain_scale, 1.0,
        cfg.physics.flash_flood_thr, cfg.sensor_noise_std, 1.5, cfg.physics.power_cut_prob,
        cfg.reward.w_growth, cfg.reward.w_optimal, cfg.reward.w_water, cfg.reward.w_stress, cfg.reward.w_overwet, cfg.reward.w_terminal,
        0.014, 0.022,
        0.018, 0.010, cfg.reward.stress_growth_penalty,
        0.40, 0.60, 0.50, 0.80,
    ]

# ══════════════════════════════════════════════════════════════════════════════
# Episode helpers
# ══════════════════════════════════════════════════════════════════════════════

def _blank_state() -> dict:
    return {"env": None, "obs": None, "done": False, "history": [], "total_reward": 0.0, "step": 0}

def _bar(v: float, lo=0.0, hi=1.0, w=20) -> str:
    b = "█"*int(v*w) + "░"*(w-int(v*w))
    return f"[{b}] {'✓' if lo<=v<=hi else '✗'}"

def _obs_display(obs, reward, step, total_reward) -> str:
    m  = float(obs["soil_moisture_obs"][0])
    cg = float(obs["crop_growth"][0])
    cs = int(round(float(obs["crop_stage"][0])))
    wt = float(obs["water_tank"][0])
    fc = obs["weather_forecast"]
    status = "✅ Optimal" if 0.40<=m<=0.72 else ("🔴 Dry" if m<0.40 else "💧 Wet")
    fc_str = " | ".join(f"{fc[i,0]:.1f}mm/{fc[i,1]:.0f}°C" for i in range(3))
    return "\n".join([
        f"📅  Day {step:>3}/–    💰 Reward: {reward:+.2f}   Total: {total_reward:+.2f}",
        "",
        f"🌡️  Moisture : {m:.3f}  {_bar(m,0.40,0.72)}  {status}",
        f"🌾  Growth   : {cg:.3f}  {_bar(cg)}",
        f"🪴  Stage    : {STAGE_NAMES.get(cs, str(cs))}",
        f"🪣  Tank     : {wt:.1%}",
        "",
        f"🌦️  Forecast : {fc_str}",
    ])

def _record_step(state, action, obs, reward, rain=0.0) -> dict:
    vols = [0,1000,3000,5000,0]
    state["history"].append({
        "day":         state["step"],
        "action":      action,
        "moisture":    float(obs["soil_moisture_obs"][0]),
        "crop_growth": float(obs["crop_growth"][0]),
        "crop_stage":  int(round(float(obs["crop_stage"][0]))),
        "rainfall_mm": rain,
        "reward":      float(reward),
        "tank_level":  float(obs["water_tank"][0]),
        "water_used_L": max(0, vols[action]),
    })
    return state

# ══════════════════════════════════════════════════════════════════════════════
# Tab actions — Play
# ══════════════════════════════════════════════════════════════════════════════

def start_episode(seed, cfg_state, *slider_vals):
    """Build config from sliders, create env, reset."""
    cfg = _cfg_from_sliders(*slider_vals)
    errors = cfg.validate()
    if errors:
        return (_blank_state(), f"❌ Config errors:\n" + "\n".join(errors),
                "", gr.update(interactive=False), gr.update(interactive=False))
    env = make_env(cfg, seed=int(seed))
    obs, _ = env.reset(seed=int(seed))
    state = {"env": env, "obs": obs, "done": False,
             "history": [], "total_reward": 0.0, "step": 0}
    display = _obs_display(obs, 0.0, 0, 0.0)
    info = f"✅ Episode started  |  {cfg.summary()}"
    return (state, display, info, gr.update(interactive=True), gr.update(interactive=False))

def take_action(action_idx: int, state: dict):
    if state["env"] is None or state["done"]:
        return state, "⚠️ Start an episode first.", "", False, True
    env = state["env"]
    obs, reward, terminated, truncated, info = env.step(action_idx)
    done = terminated or truncated
    state["total_reward"] += reward
    state["step"] += 1
    state = _record_step(state, action_idx, obs, reward, rain=info.get("rain_mm", 0.0))
    state["obs"] = obs
    state["done"] = done
    display = _obs_display(obs, reward, state["step"], state["total_reward"])
    label   = ACTION_LABELS[action_idx].split("–")[1].strip()
    log_line = (f"Day {state['step']:3d}: {label:14s} | "
                f"M={float(obs['soil_moisture_obs'][0]):.2f} | "
                f"R={reward:+.2f} | Total={state['total_reward']:+.2f}")
    if done:
        return state, display, log_line + "\n✅ Episode done — click Evaluate!", False, True
    return state, display, log_line, True, False

def get_ppo_hint(state: dict) -> str:
    if state["obs"] is None: return "Start an episode first."
    a = _ppo_action(state["obs"])
    return f"PPO → {ACTION_LABELS[a]}"

def evaluate_episode(state: dict, api_key: str) -> str:
    if not state["history"]: return "⚠️ No data. Play an episode first."
    prog = ProgrammaticGrader(state["history"]).evaluate()
    key  = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY","")
    llm  = LLMGrader(api_key=key)
    return (ProgrammaticGrader.format_scores_text(prog) + "\n\n"
            + LLMGrader.format_evaluation_text(llm.evaluate(state["history"], prog)))

# ══════════════════════════════════════════════════════════════════════════════
# Tab actions — Terminal
# ══════════════════════════════════════════════════════════════════════════════

def terminal_evaluate(actions_csv: str, api_key: str, *slider_vals):
    try:
        actions = [int(x.strip()) for x in actions_csv.split(",") if x.strip()]
    except ValueError:
        return "❌ Invalid actions — enter comma-separated integers 0-4."
    if not actions:
        return "❌ No actions provided."
    cfg = _cfg_from_sliders(*slider_vals)
    errors = cfg.validate()
    if errors:
        return "❌ Config errors:\n" + "\n".join(errors)

    env = make_env(cfg, seed=42)
    obs, _ = env.reset(seed=42)
    history, total_r = [], 0.0
    for step, action in enumerate(actions):
        action = max(0, min(4, action))
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        history.append({
            "day": step+1, "action": action,
            "moisture": float(obs["soil_moisture_obs"][0]),
            "crop_growth": float(obs["crop_growth"][0]),
            "crop_stage": int(round(float(obs["crop_stage"][0]))),
            "rainfall_mm": info.get("rain_mm", 0.0),
            "reward": float(reward),
            "tank_level": float(obs["water_tank"][0]),
            "water_used_L": max(0, [0,1000,3000,5000,0][action]),
        })
        if terminated or truncated: break
    env.close()

    prog = ProgrammaticGrader(history).evaluate()
    key  = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY","")
    llm  = LLMGrader(api_key=key)
    header = f"Config: {cfg.summary()}\nRan {len(history)} steps\n\n"
    return header + ProgrammaticGrader.format_scores_text(prog) + "\n\n" + LLMGrader.format_evaluation_text(llm.evaluate(history, prog))

# ══════════════════════════════════════════════════════════════════════════════
# Tab actions — Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def _run_policy(policy_fn, cfg: EnvConfig, seed=42) -> dict:
    env = make_env(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    history, total_r = [], 0.0
    for step in range(cfg.max_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        history.append({
            "day":step+1,"action":action,
            "moisture":float(obs["soil_moisture_obs"][0]),
            "crop_growth":float(obs["crop_growth"][0]),
            "crop_stage":int(round(float(obs["crop_stage"][0]))),
            "rainfall_mm":info.get("rain_mm",0.0),
            "reward":float(reward),
            "tank_level":float(obs["water_tank"][0]),
            "water_used_L":max(0,[0,1000,3000,5000,0][action]),
        })
        if terminated or truncated: break
    env.close()
    ev = ProgrammaticGrader(history).evaluate()
    sm = ev["summary"]
    return {"reward":round(total_r,2),"yield":sm["estimated_yield_tha"],
            "water_kL":sm["total_water_kL"],"score":ev["scores"]["overall"]}

def run_benchmark(human_history: list, seed: int, *slider_vals):
    cfg  = _cfg_from_sliders(*slider_vals)
    seed = int(seed)
    results = {
        "Random":      _run_policy(lambda o: int(np.random.randint(0,5)), cfg, seed),
        "Rule-Based":  _run_policy(_rule_based_action, cfg, seed),
        "PPO (train)": _run_policy(_ppo_action, cfg, seed),
    }
    if human_history:
        hev = ProgrammaticGrader(human_history).evaluate()
        sm  = hev["summary"]
        results["You 🧑"] = {
            "reward":sm["total_reward"],"yield":sm["estimated_yield_tha"],
            "water_kL":sm["total_water_kL"],"score":hev["scores"]["overall"],
        }
    fig = _benchmark_chart(results, cfg)
    path = os.path.join(OUTPUTS_DIR,"benchmark_chart.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=BG); plt.close(fig)

    lines = [f"Config: {cfg.summary()}", ""]
    hdr = f"{'Agent':<16} {'Score':>6} {'Reward':>8} {'Yield t/ha':>10} {'Water kL':>9}"
    lines += [hdr, "─"*len(hdr)]
    for name, r in sorted(results.items(), key=lambda x: -x[1]["score"]):
        lines.append(f"{name:<16} {r['score']:>6.1f} {r['reward']:>8.2f} {r['yield']:>10.2f} {r['water_kL']:>9.2f}")
    return path, "\n".join(lines)

def _benchmark_chart(results: dict, cfg: EnvConfig):
    agents = list(results.keys())
    colors = [GREEN if "You" in a else BLUE if "PPO" in a else ORANGE if "Rule" in a else "#6e7681" for a in agents]
    fig, axes = plt.subplots(1,3,figsize=(14,4)); fig.patch.set_facecolor(BG)
    for ax,(vals,title,unit) in zip(axes,[
        ([results[a]["score"]  for a in agents],"Overall Score","/100"),
        ([results[a]["yield"]  for a in agents],"Yield (t/ha)"," t/ha"),
        ([results[a]["water_kL"] for a in agents],"Water Used (kL)"," kL"),
    ]):
        ax.set_facecolor(PANEL)
        bars = ax.bar(agents,vals,color=colors,edgecolor=BG,linewidth=1.5)
        ax.set_title(title,color=TEXT,fontsize=11,pad=8); ax.tick_params(colors=TEXT,labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        for bar,val in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
                    f"{val:.1f}{unit}",ha="center",color=TEXT,fontsize=8)
    fig.suptitle(f"Benchmark  —  {cfg.summary()}", color=MUTED, fontsize=8, y=1.01)
    plt.tight_layout(pad=2); return fig

# ══════════════════════════════════════════════════════════════════════════════
# Tab actions — Training Metrics
# ══════════════════════════════════════════════════════════════════════════════

def load_metrics_chart():
    path = os.path.join(OUTPUTS_DIR,"metrics.json")
    if not os.path.exists(path):
        return None, "⚠️ Run `python scripts/train_and_eval.py` first."
    with open(path) as f: metrics = json.load(f)
    tlog = metrics.get("training_log",[])
    if not tlog: return None, "No training_log in metrics.json."
    steps   = [e["steps"]       for e in tlog]
    rewards = [e["mean_reward"] for e in tlog]
    fig, ax = plt.subplots(figsize=(10,4)); fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.plot(steps,rewards,color=GREEN,lw=2,label="PPO Mean Reward")
    ax.axhline(max(rewards),color=ORANGE,ls="--",alpha=0.5,label=f"Best: {max(rewards):.2f}")
    ax.set_xlabel("Training Steps",color=TEXT); ax.set_ylabel("Mean Reward",color=TEXT)
    ax.set_title("PPO Training Curve",color=TEXT,fontsize=12); ax.tick_params(colors=TEXT)
    ax.legend(facecolor=PANEL,edgecolor=BORDER,labelcolor=TEXT)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR,"training_curve.png")
    fig.savefig(out,dpi=120,bbox_inches="tight",facecolor=BG); plt.close(fig)
    agents = metrics.get("agent_comparison",{})
    lines  = ["📊 Agent Comparison from Training Run:",""]
    for agent,stats in agents.items():
        lines.append(f"  {agent:<18} Reward:{stats.get('mean_reward','?'):>8}  "
                     f"Yield:{stats.get('mean_yield','?'):>6}t/ha  "
                     f"Water:{stats.get('mean_water_kl','?'):>6}kL")
    return out, "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"),"ui-sans-serif","sans-serif"],
).set(
    body_background_fill="#0d1117", body_text_color="#c9d1d9",
    block_background_fill="#161b22", block_border_color="#30363d",
    input_background_fill="#0d1117",
    button_primary_background_fill="#238636", button_primary_text_color="white",
)
CSS = ".mono{font-family:'Courier New',monospace;font-size:13px} footer{display:none!important}"

def build_ui():
    with gr.Blocks(title="🌾 Precision Irrigation RL | OpenEnv") as demo:

        gr.HTML("""
        <div style="text-align:center;padding:20px 0 10px">
          <h1 style="color:#3fb950;font-size:2em;margin:0">🌾 Precision Irrigation RL Agent</h1>
          <p style="color:#8b949e;margin:6px 0">OpenEnv Hackathon · Meta × Hugging Face · Punjab Rice Farming</p>
          <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:8px">
            <span style="background:#238636;color:white;padding:3px 10px;border-radius:20px;font-size:12px">5 Phases Complete</span>
            <span style="background:#1f6feb;color:white;padding:3px 10px;border-radius:20px;font-size:12px">Pure-NumPy PPO</span>
            <span style="background:#6e40c9;color:white;padding:3px 10px;border-radius:20px;font-size:12px">LLM Evaluation</span>
            <span style="background:#b08800;color:white;padding:3px 10px;border-radius:20px;font-size:12px">HF Spaces Ready</span>
          </div>
        </div>""")

        # ── Shared state ──────────────────────────────────────────────────────
        ep_state  = gr.State(_blank_state())
        cfg_state = gr.State(EnvConfig().to_dict())

        # ══════════════════════════════════════════════════════════════════════
        # Build all slider refs in a list so we can reuse them across tabs
        # ══════════════════════════════════════════════════════════════════════

        with gr.Tabs():

            # ████ TAB 0: ENV CONFIG ██████████████████████████████████████████
            with gr.Tab("⚙️ Env Config"):
                gr.Markdown("""
                **Tune every physics constant, reward weight, and scenario knob.**
                Changes apply globally — all other tabs (Play, Terminal, Benchmark) use this config.
                """)

                with gr.Row():
                    preset_dd  = gr.Dropdown(
                        choices=EnvConfig.preset_names(), value=EnvConfig.preset_names()[0],
                        label="📦 Load Preset", scale=2,
                    )
                    load_preset_btn = gr.Button("Apply Preset", variant="primary", scale=1)
                    reset_btn       = gr.Button("↺ Reset to Defaults", scale=1)

                cfg_summary_box = gr.Textbox(
                    label="Active Config Summary", interactive=False,
                    value=EnvConfig().summary(), elem_classes=["mono"],
                )

                with gr.Tabs():

                    # ── Episode ──────────────────────────────────────────────
                    with gr.Tab("📅 Episode"):
                        with gr.Row():
                            s_max_steps  = gr.Slider(30,180,value=90,step=1, label="Max Steps (season length)")
                            s_grid_size  = gr.Slider(1,8,value=4,step=1, label="Farm Grid Size (N×N)")

                    # ── Tank & Pump ───────────────────────────────────────────
                    with gr.Tab("🪣 Tank & Pump"):
                        with gr.Row():
                            s_tank_cap    = gr.Slider(1000,50000,value=10000,step=500,  label="Tank Capacity (L)")
                            s_pump_eff    = gr.Slider(0.1,1.0,  value=0.85,  step=0.01, label="Pump Efficiency")
                            s_canal_rate  = gr.Slider(0.01,0.50,value=0.10,  step=0.01, label="Canal Refill Rate / day")
                        gr.Markdown("**Action Volumes** (as fraction of tank capacity)")
                        with gr.Row():
                            s_vol_low    = gr.Slider(0.01,0.30,value=0.10,step=0.01,label="Low action vol")
                            s_vol_med    = gr.Slider(0.05,0.60,value=0.30,step=0.01,label="Medium action vol")
                            s_vol_high   = gr.Slider(0.10,1.00,value=0.50,step=0.01,label="High action vol")
                            s_vol_drain  = gr.Slider(0.01,0.30,value=0.10,step=0.01,label="Drain action vol")

                    # ── Moisture Thresholds ───────────────────────────────────
                    with gr.Tab("💧 Moisture"):
                        gr.Markdown("Optimal band: green zone. Dry / Wet thresholds: stress begins outside these.")
                        with gr.Row():
                            s_opt_lo  = gr.Slider(0.10,0.65,value=0.40,step=0.01,label="Optimal Low  (opt_lo)")
                            s_opt_hi  = gr.Slider(0.35,0.95,value=0.72,step=0.01,label="Optimal High (opt_hi)")
                        with gr.Row():
                            s_dry_thr = gr.Slider(0.05,0.40,value=0.22,step=0.01,label="Dry Stress Threshold")
                            s_wet_thr = gr.Slider(0.50,0.99,value=0.78,step=0.01,label="Wet/Flood Threshold")

                    # ── Weather ───────────────────────────────────────────────
                    with gr.Tab("🌦️ Weather"):
                        with gr.Row():
                            s_et_coeff    = gr.Slider(0.001,0.020,value=0.004,step=0.001,label="ET Coefficient")
                            s_rain_scale  = gr.Slider(0.0001,0.005,value=0.0008,step=0.0001,label="Rain Scale (mm→moisture)")
                        with gr.Row():
                            s_rain_mult   = gr.Slider(0.0,3.0,value=1.0,step=0.05,label="🌧️ Rain Multiplier  (0=drought, 2=flood)")
                            s_flash_thr   = gr.Slider(10.0,100.0,value=35.0,step=1.0,label="Flash Flood Threshold (mm/day)")
                        with gr.Row():
                            s_sensor_ns   = gr.Slider(0.0,0.15,value=0.02,step=0.005,label="Sensor Noise Std")
                            s_forecast_ns = gr.Slider(0.0,10.0,value=1.5,step=0.1,label="Forecast Noise Std")
                            s_power_cut   = gr.Slider(0.0,0.50,value=0.10,step=0.01,label="Power Cut Probability")

                    # ── Reward ────────────────────────────────────────────────
                    with gr.Tab("🏆 Reward Weights"):
                        gr.Markdown("Negative = penalty. Adjust to make the environment harder/easier or change the objective.")
                        with gr.Row():
                            s_w_growth   = gr.Slider(0.0,20.0,  value=4.0,   step=0.1,   label="w_growth (crop delta)")
                            s_w_optimal  = gr.Slider(0.0,1.0,   value=0.08,  step=0.01,  label="w_optimal (in-band bonus)")
                            s_w_terminal = gr.Slider(0.0,50.0,  value=15.0,  step=0.5,   label="w_terminal (end bonus)")
                        with gr.Row():
                            s_w_water    = gr.Slider(-0.01,0.0, value=-0.00001,step=0.00001,label="w_water (per-litre cost)")
                            s_w_stress   = gr.Slider(-5.0,0.0,  value=-0.30, step=0.05,  label="w_stress (dry penalty)")
                            s_w_overwet  = gr.Slider(-5.0,0.0,  value=-0.20, step=0.05,  label="w_overwet (flood penalty)")

                    # ── Crop Growth ───────────────────────────────────────────
                    with gr.Tab("🌱 Crop Growth"):
                        gr.Markdown("Growth rates per day per stage. Reproductive stage is critical for grain fill.")
                        with gr.Row():
                            s_rate_seed  = gr.Slider(0.001,0.05,value=0.014,step=0.001,label="Seedling rate")
                            s_rate_veg   = gr.Slider(0.001,0.05,value=0.022,step=0.001,label="Vegetative rate")
                            s_rate_repro = gr.Slider(0.001,0.05,value=0.018,step=0.001,label="Reproductive rate")
                            s_rate_mat   = gr.Slider(0.001,0.05,value=0.010,step=0.001,label="Maturity rate")
                        s_stress_pen = gr.Slider(0.0,0.10,value=0.010,step=0.001,label="Stress Growth Penalty")

                    # ── Init Conditions ───────────────────────────────────────
                    with gr.Tab("🎲 Init Conditions"):
                        gr.Markdown("Randomisation bounds for episode starting state.")
                        with gr.Row():
                            s_init_m_lo = gr.Slider(0.10,0.60,value=0.40,step=0.01,label="Init Moisture Low")
                            s_init_m_hi = gr.Slider(0.20,0.90,value=0.60,step=0.01,label="Init Moisture High")
                        with gr.Row():
                            s_init_t_lo = gr.Slider(0.10,0.90,value=0.50,step=0.01,label="Init Tank Low (fraction)")
                            s_init_t_hi = gr.Slider(0.20,1.00,value=0.80,step=0.01,label="Init Tank High (fraction)")

                # ── Export / Import ───────────────────────────────────────────
                gr.Markdown("### 📤 Export / Import Config")
                with gr.Row():
                    export_btn  = gr.Button("Export Config as JSON", variant="secondary")
                    validate_btn = gr.Button("✅ Validate Config", variant="secondary")
                config_json_box = gr.Textbox(
                    label="Config JSON (edit & paste to import, or copy to save)",
                    lines=8, interactive=True, elem_classes=["mono"],
                    value=EnvConfig().to_json(),
                )
                import_btn   = gr.Button("📥 Import from JSON above", variant="secondary")
                config_status = gr.Textbox(label="Status", interactive=False, lines=2)

                # Collect ALL slider components into a list (order must match _cfg_from_sliders)
                ALL_SLIDERS = [
                    s_max_steps, s_grid_size,
                    s_tank_cap, s_pump_eff, s_canal_rate,
                    s_vol_low, s_vol_med, s_vol_high, s_vol_drain,
                    s_opt_lo, s_opt_hi, s_dry_thr, s_wet_thr,
                    s_et_coeff, s_rain_scale, s_rain_mult,
                    s_flash_thr, s_sensor_ns, s_forecast_ns, s_power_cut,
                    s_w_growth, s_w_optimal, s_w_water, s_w_stress, s_w_overwet, s_w_terminal,
                    s_rate_seed, s_rate_veg, s_rate_repro, s_rate_mat, s_stress_pen,
                    s_init_m_lo, s_init_m_hi, s_init_t_lo, s_init_t_hi,
                ]


                # ?? Config tab wiring ?????????????????????????????????????????????

                def _update_summary(*vals):
                    cfg = _cfg_from_sliders(*vals)
                    errs = cfg.validate()
                    if errs:
                        return f"?? {errs[0]}", cfg.to_json()
                    return cfg.summary(), cfg.to_json()

                def _apply_preset(preset_name):
                    vals = _preset_to_slider_values(preset_name)
                    cfg = EnvConfig.from_preset(preset_name)
                    return [cfg.summary(), cfg.to_json(), f"? Preset '{preset_name}' loaded"] + vals

                def _reset_defaults():
                    vals = _all_slider_defaults()
                    cfg = EnvConfig()
                    return [cfg.summary(), cfg.to_json(), "? Reset to defaults"] + vals

                def _export_json(*vals):
                    cfg = _cfg_from_sliders(*vals)
                    errs = cfg.validate()
                    if errs:
                        return cfg.to_json(), "?? " + "; ".join(errs)
                    return cfg.to_json(), "? JSON exported above ? copy to save"

                def _validate_json(*vals):
                    cfg = _cfg_from_sliders(*vals)
                    errs = cfg.validate()
                    if errs:
                        return "? " + "; ".join(errs)
                    diff = cfg.diff_from_defaults()
                    if diff:
                        return (
                            f"? Config valid. {len(diff)} fields differ from defaults: "
                            + ", ".join(diff)
                        )
                    return "? Config valid ? all defaults."

                def _import_json(json_str):
                    try:
                        cfg = EnvConfig.from_json(json_str)
                        errs = cfg.validate()
                        if errs:
                            return ["? " + "; ".join(errs), cfg.summary()] + _all_slider_defaults()
                        vals = [
                            cfg.episode.max_steps, cfg.episode.grid_size,
                            cfg.episode.tank_capacity, cfg.physics.pump_efficiency, cfg.physics.tank_refill_rate,
                            cfg.actions.vol_low, cfg.actions.vol_medium, cfg.actions.vol_high, abs(cfg.actions.vol_drain),
                            cfg.moisture.opt_lo, cfg.moisture.opt_hi, cfg.moisture.dry_thr, cfg.moisture.wet_thr,
                            cfg.physics.et_coeff, cfg.physics.rain_scale, 1.0,
                            cfg.physics.flash_flood_thr, cfg.sensor_noise_std, 1.5, cfg.physics.power_cut_prob,
                            cfg.reward.w_growth, cfg.reward.w_optimal, cfg.reward.w_water, cfg.reward.w_stress, cfg.reward.w_overwet, cfg.reward.w_terminal,
                            0.014, 0.022,
                            0.018, 0.010, cfg.reward.stress_growth_penalty,
                            0.40, 0.60, 0.50, 0.80,
                        ]
                        return ["? JSON imported", cfg.summary()] + vals
                    except Exception as e:
                        return [f"? Parse error: {e}", ""] + _all_slider_defaults()

                # Live summary update on any slider change
                for s in ALL_SLIDERS:
                    s.change(_update_summary, inputs=ALL_SLIDERS, outputs=[cfg_summary_box, config_json_box])

                load_preset_btn.click(
                    _apply_preset, inputs=[preset_dd],
                    outputs=[cfg_summary_box, config_json_box, config_status] + ALL_SLIDERS,
                )
                reset_btn.click(
                    _reset_defaults,
                    outputs=[cfg_summary_box, config_json_box, config_status] + ALL_SLIDERS,
                )
                export_btn.click(_export_json, inputs=ALL_SLIDERS, outputs=[config_json_box, config_status])
                validate_btn.click(_validate_json, inputs=ALL_SLIDERS, outputs=[config_status])
                import_btn.click(
                    _import_json, inputs=[config_json_box],
                    outputs=[config_status, cfg_summary_box] + ALL_SLIDERS,
                )
            # ████ TAB 1: PLAY ████████████████████████████████████████████████
            with gr.Tab("🎮 Play as Agent"):
                gr.Markdown("""
                **Step through an irrigation season day by day.**
                The environment uses whatever config is set in ⚙️ Env Config.
                At the end, Claude evaluates your performance and gives you a personalised improvement plan.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        play_seed    = gr.Slider(0,100,value=42,step=1,label="🎲 Random Seed")
                        play_cfg_lbl = gr.Textbox(label="Active Config", interactive=False,
                                                  value=EnvConfig().summary(), elem_classes=["mono"])
                        start_btn    = gr.Button("▶ Start New Episode", variant="primary")

                        gr.Markdown("### 💧 Choose Action")
                        action_btns = []
                        for lbl in ["⬜ No Water","🔵 Low (1kL)","💧 Medium (3kL)","🌊 High (5kL)","⬇️ Drain"]:
                            b = gr.Button(lbl, interactive=False)
                            action_btns.append(b)

                        ppo_btn = gr.Button("🤖 PPO Hint", variant="secondary", interactive=False)
                        ppo_out = gr.Textbox(label="PPO would choose:", lines=1, interactive=False)

                    with gr.Column(scale=2):
                        obs_box  = gr.Textbox(label="📡 Environment State", lines=12,
                                              interactive=False, elem_classes=["mono"],
                                              value="Press ▶ Start New Episode to begin.")
                        log_box  = gr.Textbox(label="📋 Action Log", lines=8,
                                              interactive=False, elem_classes=["mono"])
                        eval_btn = gr.Button("📊 Evaluate My Performance", variant="primary", interactive=False)

                api_key_p = gr.Textbox(label="🔑 Anthropic API Key (optional)", type="password",
                                       placeholder="sk-ant-...")
                eval_out  = gr.Textbox(label="🧠 Evaluation & Improvement Plan",
                                       lines=30, interactive=False, elem_classes=["mono"])

                log_state = gr.State("")

                def _append_log(existing, new_line):
                    if not new_line: return existing
                    lines = (existing or "").split("\n")
                    lines.append(new_line)
                    return "\n".join(lines[-25:])

                def _sync_cfg_label(*vals):
                    cfg = _cfg_from_sliders(*vals)
                    return cfg.summary()

                # Sync play tab's config label whenever sliders change
                for s in ALL_SLIDERS:
                    s.change(_sync_cfg_label, inputs=ALL_SLIDERS, outputs=[play_cfg_lbl])

                def _start(seed, log, *vals):
                    state, display, info, act_upd, eval_upd = start_episode(seed, {}, *vals)
                    return (state, display, info,
                            gr.update(interactive=True), gr.update(interactive=False),
                            "", gr.update(interactive=True))

                start_btn.click(
                    _start,
                    inputs=[play_seed, log_state] + ALL_SLIDERS,
                    outputs=[ep_state, obs_box, log_box, eval_btn, eval_btn, log_state, ppo_btn],
                ).then(lambda: [gr.update(interactive=True)]*5, outputs=action_btns)

                def _make_action_fn(idx):
                    def _fn(state, log):
                        new_state, display, log_line, acts_on, eval_on = take_action(idx, state)
                        new_log = _append_log(log, log_line)
                        acts_update  = gr.update(interactive=acts_on)
                        eval_update  = gr.update(interactive=eval_on)
                        ppo_update   = gr.update(interactive=acts_on)
                        return [new_state, display, new_log, eval_update] + [acts_update]*5 + [ppo_update]
                    return _fn

                for i, btn in enumerate(action_btns):
                    btn.click(
                        _make_action_fn(i),
                        inputs=[ep_state, log_state],
                        outputs=[ep_state, obs_box, log_state, eval_btn] + action_btns + [ppo_btn],
                    ).then(lambda s: s, inputs=[log_state], outputs=[log_box])

                ppo_btn.click(get_ppo_hint, inputs=[ep_state], outputs=[ppo_out])
                eval_btn.click(evaluate_episode, inputs=[ep_state, api_key_p], outputs=[eval_out])

            # ████ TAB 2: TERMINAL ████████████████████████████████████████████
            with gr.Tab("⌨️ Terminal Evaluator"):
                gr.Markdown("""
                **Paste a comma-separated action sequence (0–4) — the env runs it end-to-end and scores you.**
                Uses the config set in ⚙️ Env Config. Perfect for batch-testing scripted strategies.
                """)
                with gr.Row():
                    with gr.Column():
                        term_cfg_lbl = gr.Textbox(label="Active Config", interactive=False,
                                                  value=EnvConfig().summary(), elem_classes=["mono"])
                        term_actions = gr.Textbox(
                            label="Action Sequence (comma-separated, up to max_steps values)",
                            placeholder="e.g.  0,2,2,0,1,3,0,2,4,0,0,2...",
                            lines=4,
                        )
                        gr.Examples(
                            examples=[
                                ["0,2,2,0,1,3,0,2,0,0,2,2,0,1,0,2,3,0,0,2,1,0,2,0,0,3,2,0,2,0"],
                                ["2,2,2,2,1,3,3,2,2,3,2,3,2,2,3,3,2,2,3,2,2,3,3,3,2,2,3,2,3,2"],
                                ["0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"],
                                ["3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3"],
                            ],
                            inputs=[term_actions],
                            label="Quick Examples (Balanced / Drought / Idle / Greedy)",
                        )
                        api_key_t  = gr.Textbox(label="🔑 Anthropic API Key (optional)", type="password")
                        run_t_btn  = gr.Button("▶ Run & Evaluate", variant="primary")

                term_out = gr.Textbox(label="Evaluation Results", lines=35,
                                      interactive=False, elem_classes=["mono"])

                for s in ALL_SLIDERS:
                    s.change(_sync_cfg_label, inputs=ALL_SLIDERS, outputs=[term_cfg_lbl])

                run_t_btn.click(terminal_evaluate,
                                inputs=[term_actions, api_key_t] + ALL_SLIDERS,
                                outputs=[term_out])

            # ████ TAB 3: BENCHMARK ███████████████████████████████████████████
            with gr.Tab("📊 Benchmark"):
                gr.Markdown("""
                **Compare your last played episode against PPO, Rule-Based, and Random agents.**
                All agents run under the same config & seed for a fair comparison.
                """)
                bench_cfg_lbl = gr.Textbox(label="Active Config", interactive=False,
                                           value=EnvConfig().summary(), elem_classes=["mono"])
                bench_seed  = gr.Slider(0,100,value=42,step=1,label="Seed")
                bench_btn   = gr.Button("🏁 Run Benchmark", variant="primary")
                bench_chart = gr.Image(label="Agent Comparison", type="filepath")
                bench_table = gr.Textbox(label="Results Table", lines=12, elem_classes=["mono"])

                for s in ALL_SLIDERS:
                    s.change(_sync_cfg_label, inputs=ALL_SLIDERS, outputs=[bench_cfg_lbl])

                bench_btn.click(
                    lambda state, seed, *vals: run_benchmark(state.get("history",[]), seed, *vals),
                    inputs=[ep_state, bench_seed] + ALL_SLIDERS,
                    outputs=[bench_chart, bench_table],
                )

            # ████ TAB 4: TRAINING METRICS ████████████████████████████████████
            with gr.Tab("📈 Training Metrics"):
                gr.Markdown("Pre-trained PPO training curve and agent comparison. "
                            "Run `python scripts/train_and_eval.py` to regenerate.")
                load_m_btn   = gr.Button("📥 Load Metrics", variant="secondary")
                metrics_img  = gr.Image(label="PPO Training Curve", type="filepath")
                metrics_text = gr.Textbox(label="Agent Comparison Summary",
                                          lines=15, elem_classes=["mono"])
                load_m_btn.click(load_metrics_chart, outputs=[metrics_img, metrics_text])
                dash = os.path.join(OUTPUTS_DIR,"dashboard.png")
                if os.path.exists(dash):
                    gr.Image(value=dash, label="Full Dashboard (Phase 5)", type="filepath")

            # ████ TAB 5: ABOUT ████████████████████████████████████████████████
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## 🌾 Precision Irrigation RL Agent

                Built for the **OpenEnv AI Hackathon** — Meta × Hugging Face × PyTorch.
                Trains a RL agent to manage irrigation for a 90-day Punjab rice season,
                maximising yield while conserving groundwater for 140M farmers.

                ### Architecture
                | Component | Description |
                |-----------|-------------|
                | `EnvConfig` | All tunable constants in one dataclass — presets, JSON export/import |
                | `make_env` | Gym-compatible env driven entirely by `EnvConfig` |
                | `PrecisionIrrigationEnvV3` | Original env (unchanged, still usable) |
                | `PPO (Pure NumPy)` | From-scratch PPO: Adam, GAE, clipped surrogate, entropy annealing |
                | `ProgrammaticGrader` | Rule-based scoring: water efficiency, crop health, timing |
                | `LLMGrader` | Claude-powered evaluation + personalised improvement plan |
                | `Gradio UI` | Interactive play, terminal evaluator, config panel, benchmarking |

                ### Env Config Knobs
                | Category | Knobs |
                |----------|-------|
                | Episode | max_steps, grid_size |
                | Tank/Pump | tank_capacity, pump_efficiency, canal_refill_rate, action volumes |
                | Moisture | opt_lo, opt_hi, dry_thr, wet_thr |
                | Weather | et_coeff, rain_scale, rain_multiplier, flash_flood_thr, noise std devs, power_cut_prob |
                | Reward | w_growth, w_optimal, w_water, w_stress, w_overwet, w_terminal |
                | Crop | per-stage growth rates, stress_growth_penalty |
                | Init | init_moisture range, init_tank range |

                ### Presets
                `normal` · `drought` · `flood` · `power_outage` · `water_scarce` · `high_yield_challenge` · `easy` · `hard`

                ### OpenEnv Alignment
                ✅ Gymnasium-style `step()` / `reset()` API  
                ✅ Programmatic + LLM evaluation  
                ✅ Configurable, reproducible, isolated  
                ✅ Hugging Face Spaces deployable  
                ✅ Docker containerised  

                ### Deploy to HF Spaces
                See `DEPLOY.md` in the repo for full instructions.
                """)

        gr.HTML('<div style="text-align:center;color:#484f58;font-size:12px;padding:16px 0 8px">'
                'Built for OpenEnv Hackathon · Meta × Hugging Face · MIT License</div>')

    return demo

if __name__ == "__main__":
    import argparse
    import uvicorn
    import threading
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    def _background_setup():
        """Run slow startup tasks in background so server binds immediately."""
        try:
            rainfall = os.path.join(DATA_DIR, "rainfall.csv")
            if not os.path.exists(rainfall):
                print("Generating weather data...", flush=True)
                try:
                    subprocess.run([sys.executable, "scripts/generate_data.py"], check=True, timeout=120)
                except Exception as e:
                    print(f"[startup] WARNING: weather data generation failed ({e})", flush=True)
        except Exception as e:
            print(f"[startup] WARNING: setup failed ({e})", flush=True)

        if os.getenv("AUTO_TRAIN_STEPS", "25000") != "0":
            _ensure_ppo_model()

    # Start weather + training in background — server binds to port immediately
    t = threading.Thread(target=_background_setup, daemon=True)
    t.start()

    demo = build_ui()

    print(f"🚀  Starting Precision Irrigation Agent on port {args.port}", flush=True)
    app = gr.mount_gradio_app(
        app,
        demo,
        path=UI_MOUNT_PATH,
        root_path=UI_MOUNT_PATH,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)