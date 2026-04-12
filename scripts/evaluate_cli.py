#!/usr/bin/env python3
"""
scripts/evaluate_cli.py — Terminal Human Agent Evaluator
=========================================================
Play through the irrigation environment in the terminal, OR submit a
pre-made action sequence, and receive programmatic + LLM evaluation.

Usage:
  # Interactive (default env):
  python scripts/evaluate_cli.py --mode interactive

  # Interactive with drought preset:
  python scripts/evaluate_cli.py --mode interactive --preset drought

  # Batch mode — comma-separated actions:
  python scripts/evaluate_cli.py --mode batch --actions "0,2,2,0,1,3,0,2,0,0"

  # Batch with heavy drought + extra sensor noise:
  python scripts/evaluate_cli.py --mode batch --actions "2,3,2,3,2" \
      --preset drought --rain-mult 0.1 --sensor-noise 0.08

  # Load a saved JSON config:
  python scripts/evaluate_cli.py --mode batch --actions "0,2,2" --config-file my_env.json

  # Dump active config as JSON (for saving / editing):
  python scripts/evaluate_cli.py --preset hard --dump-config

  # List all presets:
  python scripts/evaluate_cli.py --list-presets

  # Enable Claude AI coaching:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/evaluate_cli.py --mode interactive
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.env_config           import EnvConfig
from envs.configurable_env     import make_env
from evaluator.programmatic_grader import ProgrammaticGrader
from evaluator.llm_grader          import LLMGrader

ACTION_LABELS = {0:"No Water", 1:"Low (1kL)", 2:"Medium (3kL)", 3:"High (5kL)", 4:"Drain"}
STAGE_NAMES   = {0:"Seedling", 1:"Vegetative", 2:"Reproductive", 3:"Maturity"}
WATER_VOLS    = [0, 1000, 3000, 5000, 0]

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  🌾  Precision Irrigation RL  ·  Terminal Evaluator         ║
║      OpenEnv Hackathon  ·  Meta × Hugging Face              ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Display helpers ────────────────────────────────────────────────────────────

def _bar(v, width=20, lo=0.0, hi=1.0):
    b = "█"*int(v*width) + "░"*(width-int(v*width))
    return f"[{b}] {'✓' if lo<=v<=hi else '✗'}"

def _show_state(obs, day, last_r, total_r, cfg):
    m  = float(obs["soil_moisture_obs"][0])
    cg = float(obs["crop_growth"][0])
    cs = int(round(float(obs["crop_stage"][0])))
    wt = float(obs["water_tank"][0])
    fc = obs["weather_forecast"]
    status = "OPTIMAL ✅" if cfg.opt_lo<=m<=cfg.opt_hi else ("DRY 🔴" if m<cfg.opt_lo else "WET 💧")
    print(f"\n{'─'*62}")
    print(f" Day {day:>3}/{cfg.max_steps}  Last reward: {last_r:+.2f}   Total: {total_r:+.2f}")
    print(f"{'─'*62}")
    print(f" Soil Moisture : {m:.3f}  {_bar(m,lo=cfg.opt_lo,hi=cfg.opt_hi)}  {status}")
    print(f" Crop Growth   : {cg:.3f}  {_bar(cg)}")
    print(f" Growth Stage  : {STAGE_NAMES.get(cs, str(cs))}")
    print(f" Water Tank    : {wt:.1%}  (capacity={cfg.tank_capacity/1000:.0f}kL)")
    rain_fc = " | ".join(f"{fc[i,0]:.1f}mm/{fc[i,1]:.0f}°C" for i in range(3))
    print(f" 3-Day Forecast: {rain_fc}")
    print(f"{'─'*62}")

def _menu():
    print("\n Actions:")
    for k,v in ACTION_LABELS.items():
        print(f"   [{k}] {v}")
    print("   [h] Get PPO hint")
    print("   [q] Quit early")

def _human_action(obs):
    while True:
        raw = input("  Choose > ").strip().lower()
        if raw == "q": return None
        if raw == "h": return "hint"
        try:
            a = int(raw)
            if 0 <= a <= 4: return a
        except ValueError:
            pass
        print("  ❌ Enter 0-4, h, or q.")

# ── Episode runners ────────────────────────────────────────────────────────────

def run_interactive(cfg: EnvConfig, seed=42):
    print(BANNER)
    print(f" Config : {cfg.summary()}")
    print(f" Optimal moisture band: [{cfg.opt_lo}, {cfg.opt_hi}]")
    input("\n Press Enter to start...\n")

    from rl.obs_normalizer import manual_normalize
    from rl.ppo import ActorCritic, softmax
    import pickle, pathlib
    ppo = None
    pkl = pathlib.Path(__file__).parent.parent / "models" / "ppo_irrigation.pkl"
    if pkl.exists():
        try:
            # Preferred path for current PPO checkpoints (state dict).
            ppo = ActorCritic.load(str(pkl))
        except Exception:
            # Backward compatibility for older pickled object checkpoints.
            with open(pkl, "rb") as f:
                ppo = pickle.load(f)

    def _ppo_action(obs):
        if ppo is None:
            return None

        x = manual_normalize(obs)

        # Current API: ActorCritic.forward returns (logits, value).
        if hasattr(ppo, "forward"):
            logits, _ = ppo.forward(x[np.newaxis])
            return int(np.argmax(softmax(logits[0])))

        # Legacy compatibility path (older custom actor wrappers).
        if hasattr(ppo, "actor"):
            return int(np.argmax(softmax(ppo.actor(x))))

        return None

    env = make_env(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    history, total_r, last_r = [], 0.0, 0.0

    for step in range(1, cfg.max_steps+1):
        _show_state(obs, step, last_r, total_r, cfg)
        _menu()
        action = _human_action(obs)
        if action is None:
            print("\n⚠️  Quit early."); break
        if action == "hint":
            try:
                ha = _ppo_action(obs)
            except Exception:
                ha = None
            print(f"  🤖 PPO → {ACTION_LABELS.get(ha,'N/A')}" if ha is not None else "  🤖 PPO model not loaded.")
            action = _human_action(obs)
            if action is None: break

        obs, reward, terminated, truncated, info = env.step(action)
        last_r = reward; total_r += reward
        history.append({
            "day":step,"action":action,
            "moisture":float(obs["soil_moisture_obs"][0]),
            "crop_growth":float(obs["crop_growth"][0]),
            "crop_stage":int(round(float(obs["crop_stage"][0]))),
            "rainfall_mm":info.get("rain_mm",0.0),
            "reward":float(reward),
            "tank_level":float(obs["water_tank"][0]),
            "water_used_L":max(0,WATER_VOLS[action]),
        })
        print(f"  ✔ {ACTION_LABELS[action]:14s}  → Reward: {reward:+.2f}  Total: {total_r:+.2f}")
        if info.get("flash_flood"): print("  ⚠️  Flash flood!")
        if info.get("power_cut"):   print("  ⚡ Power cut — pump unavailable today!")
        if terminated or truncated:
            print(f"\n✅ Episode complete at Day {step}"); break

    env.close()
    return history


def run_batch(actions, cfg: EnvConfig, seed=42):
    env = make_env(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    history, total_r = [], 0.0
    print(f"\n Running {len(actions)} actions (seed={seed})")
    print(f" Config : {cfg.summary()}\n")
    for step, action in enumerate(actions, 1):
        action = max(0, min(4, action))
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        history.append({
            "day":step,"action":action,
            "moisture":float(obs["soil_moisture_obs"][0]),
            "crop_growth":float(obs["crop_growth"][0]),
            "crop_stage":int(round(float(obs["crop_stage"][0]))),
            "rainfall_mm":info.get("rain_mm",0.0),
            "reward":float(reward),
            "tank_level":float(obs["water_tank"][0]),
            "water_used_L":max(0,WATER_VOLS[action]),
        })
        if step % 15 == 0 or step <= 3:
            m = float(obs["soil_moisture_obs"][0])
            print(f"  Day {step:>3}: action={action}({ACTION_LABELS[action]:<14}) "
                  f"moisture={m:.2f}  reward={reward:+.2f}  cumulative={total_r:+.2f}")
        if terminated or truncated: break
    env.close()
    print(f"\n  Completed {len(history)} steps  |  Total reward: {total_r:.2f}")
    return history

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_and_print(history, cfg: EnvConfig, api_key=None):
    print(f"\n{'═'*62}")
    print(f" EVALUATING YOUR PERFORMANCE  ({len(history)} steps)")
    print(f" Config: {cfg.summary()}")
    print(f"{'═'*62}")
    grader  = ProgrammaticGrader(history)
    prog    = grader.evaluate()
    print("\n" + ProgrammaticGrader.format_scores_text(prog))

    key = api_key or os.environ.get("ANTHROPIC_API_KEY","")
    llm = LLMGrader(api_key=key)
    if llm.is_available:
        print("\n🤖 Consulting Claude AI Coach...")
        t0 = time.time()
        result = llm.evaluate(history, prog)
        print(f"   (response in {time.time()-t0:.1f}s)\n")
    else:
        print("\nℹ️  No API key — using rule-based coaching (set ANTHROPIC_API_KEY for Claude analysis)")
        result = llm.evaluate(history, prog)
    print(LLMGrader.format_evaluation_text(result))

    out_path = os.path.join(os.path.dirname(__file__),"..","outputs","human_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path,"w") as f:
        json.dump({
            "history_length":len(history),
            "config":cfg.to_dict(),
            "programmatic":prog,
            "llm_evaluation":result,
        }, f, indent=2)
    print(f"\n📁 Full results → {os.path.abspath(out_path)}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Precision Irrigation RL — Terminal Evaluator",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Available presets: " + ", ".join(EnvConfig.preset_names()),
    )
    parser.add_argument("--mode",         choices=["interactive","batch"], default="interactive")
    parser.add_argument("--actions",      type=str,   default=None)
    parser.add_argument("--actions-file", type=str,   default=None)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--api-key",      type=str,   default=None)
    # Config
    parser.add_argument("--preset",       type=str,   default="normal",
                        help="Env preset (normal|drought|flood|power_outage|water_scarce|easy|hard|high_yield_challenge)")
    parser.add_argument("--config-json",  type=str,   default=None,
                        help='JSON overrides, e.g. \'{"rain_multiplier":0.3,"et_coeff":0.007}\'')
    parser.add_argument("--config-file",  type=str,   default=None,
                        help="Path to a saved JSON config file")
    parser.add_argument("--dump-config",  action="store_true",
                        help="Print active config JSON and exit")
    parser.add_argument("--list-presets", action="store_true",
                        help="List all available presets and exit")
    # Quick individual overrides
    parser.add_argument("--rain-mult",    type=float, default=None, help="Rain multiplier (0=drought, 2=flood)")
    parser.add_argument("--et-coeff",     type=float, default=None, help="Evapotranspiration coefficient")
    parser.add_argument("--max-steps",    type=int,   default=None, help="Episode length in days")
    parser.add_argument("--tank",         type=float, default=None, help="Tank capacity (litres)")
    parser.add_argument("--power-cut",    type=float, default=None, help="Power cut probability 0–1")
    parser.add_argument("--sensor-noise", type=float, default=None, help="Sensor noise std dev")
    args = parser.parse_args()

    if args.list_presets:
        print("\nAvailable presets:\n")
        for name in EnvConfig.preset_names():
            cfg = EnvConfig.from_preset(name)
            print(f"  {name:<25} {cfg.summary()}")
        return

    # ── Build config ──────────────────────────────────────────────────────────
    if args.config_file:
        cfg = EnvConfig.from_json(args.config_file)
    else:
        cfg = EnvConfig.from_preset(args.preset)

    if args.config_json:
        overrides = json.loads(args.config_json)
        d = cfg.to_dict(); d.update(overrides)
        cfg = EnvConfig.from_dict(d)

    # Individual quick overrides
    quick = {
        "rain_multiplier":  args.rain_mult,
        "et_coeff":         args.et_coeff,
        "max_steps":        args.max_steps,
        "tank_capacity":    args.tank,
        "power_cut_prob":   args.power_cut,
        "sensor_noise_std": args.sensor_noise,
    }
    d = cfg.to_dict()
    for k,v in quick.items():
        if v is not None: d[k] = v
    cfg = EnvConfig.from_dict(d)
    if any(v is not None for v in quick.values()):
        cfg.preset = "custom"

    errs = cfg.validate()
    if errs:
        print("❌ Config validation errors:"); [print(f"  {e}") for e in errs]; sys.exit(1)

    if args.dump_config:
        print(cfg.to_json()); return

    # ── Run episode ───────────────────────────────────────────────────────────
    if args.mode == "interactive":
        history = run_interactive(cfg, seed=args.seed)
    else:
        if args.actions_file:
            with open(args.actions_file) as f:
                actions = [int(l.strip()) for l in f if l.strip()]
        elif args.actions:
            actions = [int(x.strip()) for x in args.actions.split(",") if x.strip()]
        else:
            print("❌ Batch mode requires --actions or --actions-file"); sys.exit(1)
        history = run_batch(actions, cfg, seed=args.seed)

    if not history:
        print("No history to evaluate."); return

    evaluate_and_print(history, cfg, api_key=args.api_key)

if __name__ == "__main__":
    main()
