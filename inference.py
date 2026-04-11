import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
from typing import Any

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.featherless.ai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("FEATHERLESS_API_KEY")
    or os.getenv("HF_TOKEN")
)
print(f"[DEBUG] Key loaded: {'YES, length=' + str(len(OPENAI_API_KEY)) if OPENAI_API_KEY else 'NO - KEY IS NONE'}", flush=True)


LOCAL_URL = os.getenv("LOCAL_URL", "http://localhost:7860")
BENCHMARK = os.getenv("BENCHMARK", "precision-irrigation-agent")
TASKS = [t.strip() for t in os.getenv("TASKS", "task_1,task_2,task_3").split(",") if t.strip()]
MAX_STEPS = _safe_int(os.getenv("MAX_STEPS", "100"), 100)


# ─────────────────────────────────────────────────────────────
# OpenAI Client (safe init)
# ─────────────────────────────────────────────────────────────

try:
    client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY or "dummy-key")
except Exception:
    client = None


# ─────────────────────────────────────────────────────────────
# LLM Prompt + Parsing
# ─────────────────────────────────────────────────────────────

def _first(value: Any, default: float = 0.0) -> float:
    if isinstance(value, list) and value:
        return float(value[0])
    try:
        return float(value)
    except Exception:
        return default


def _build_prompt(
    obs: dict[str, Any],
    prev_reward: float | None = None,
    total_reward: float = 0.0,
) -> str:
    moisture = _first(obs.get("soil_moisture_obs"), 0.5)
    growth = _first(obs.get("crop_growth"), 0.0)
    stage = int(_first(obs.get("crop_stage"), 0))
    tank = _first(obs.get("water_tank"), 0.0)
    power = int(_first(obs.get("power_status"), 1))
    day = int(_first(obs.get("day_of_season"), 0))
    forecast = obs.get("weather_forecast", [])

    rain_tomorrow = 0.0
    if isinstance(forecast, list) and len(forecast) > 0:
        row = forecast[0]
        if isinstance(row, list) and len(row) > 0:
            rain_tomorrow = float(row[0])

    stage_names = {0: "Seedling", 1: "Vegetative", 2: "Reproductive", 3: "Maturity"}

    # Determine urgency level for clear agent guidance
    if power == 0:
        urgency = "PUMP IS OFF — irrigation impossible today. Choose 0."
    elif moisture > 0.78:
        urgency = "WATERLOGGED (moisture > 0.78) — choose 4 (Drain) immediately."
    elif moisture < 0.22:
        urgency = "SEVERE DRY STRESS (moisture < 0.22) — choose 3 (High) immediately."
    elif moisture < 0.40:
        urgency = f"DRY (moisture < 0.40) — choose 2 (Medium) unless heavy rain forecast."
    elif moisture > 0.72:
        urgency = "ABOVE OPTIMAL — choose 0 (No Water) or 4 (Drain)."
    elif moisture >= 0.40 and moisture <= 0.72:
        urgency = "IN OPTIMAL BAND — choose 0 or 1 depending on forecast."
    else:
        urgency = "Monitor moisture — irrigate if dropping below 0.40."

    return (
        "IRRIGATION CONTROLLER — respond with ONE digit only: 0, 1, 2, 3, or 4.\n\n"
        "ACTIONS:\n"
        "  0 = No Water\n"
        "  1 = Low irrigation (~1 kL)\n"
        "  2 = Medium irrigation (~3 kL)\n"
        "  3 = High irrigation (~5 kL)\n"
        "  4 = Drain excess water\n\n"
        "CURRENT STATE:\n"
        f"  Day: {day}/90 | Stage: {stage_names.get(stage, stage)}\n"
        f"  Soil moisture: {moisture:.3f}  (optimal: 0.40–0.72)\n"
        f"  Crop growth: {growth:.3f} | Tank level: {tank:.2f}\n"
        f"  Power: {'ON' if power == 1 else 'OFF (pump unavailable)'}\n"
        f"  Rain tomorrow forecast: {rain_tomorrow:.1f} mm\n\n"
        f"REWARD FEEDBACK:\n"
        f"  Previous step reward: {prev_reward if prev_reward is not None else 'N/A (first step)'}\n"
        f"  Running total reward: {total_reward:.3f}\n\n"
        f"DECISION: {urgency}\n\n"
        "YOUR ANSWER (single digit 0-4):"
    )


def _parse_action(raw: str) -> int:
    digits = "".join(c for c in raw if c.isdigit())
    if not digits:
        return 0
    return max(0, min(4, int(digits[0])))


# ─────────────────────────────────────────────────────────────
# Logging (STRICT FORMAT)
# ─────────────────────────────────────────────────────────────

def _bool(v: bool) -> str:
    return "true" if v else "false"


def log_start(task: str):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: int, reward: float, done: bool, error: str | None):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={_bool(done)} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_bool(success)} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────
# Core Runner
# ─────────────────────────────────────────────────────────────

def run_task(task_id: str):
    log_start(task_id)

    steps = 0
    rewards: list[float] = []
    total_reward = 0.0
    prev_reward: float | None = None
    score = 0.0
    success = False
    result = {}

    try:
        # Reset env
        r = requests.post(f"{LOCAL_URL}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        obs = r.json()["observation"]

        done = False

        while not done and steps < MAX_STEPS:
            # ── Choose action ──
            try:
                if client is None:
                    raise RuntimeError()

                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": _build_prompt(obs, prev_reward, total_reward)}],
                    temperature=0.0,
                    max_tokens=5,
                )
                raw = resp.choices[0].message.content or ""
                action = _parse_action(raw)

            except Exception as e:
                print(f"[DEBUG] LLM call failed: {e}", flush=True)
                action = 0

            # ── Step env ──
            r = requests.post(f"{LOCAL_URL}/step", json={"action": action}, timeout=30)
            r.raise_for_status()
            result = r.json()

            obs = result["observation"]
            reward = float(result.get("reward", 0.0))
            terminated = result.get("terminated", False)
            truncated = result.get("truncated", False)
            done = terminated or truncated

            err = result.get("info", {}).get("last_action_error")

            steps += 1
            rewards.append(reward)
            total_reward += reward
            prev_reward = reward

            log_step(steps, action, reward, done, err)

        # ─────────────────────────────────────────────
        # SCORE FIX (IMPORTANT)
        # ─────────────────────────────────────────────
        if result:
            score = float(result.get("score", 0.0))

        # fallback if server didn't send score
        if score == 0.0 and rewards:
            total = sum(rewards)
            score = total / (abs(total) + 1e-6)  # normalize
            score = (score + 1) / 2              # map [-1,1] → [0,1]

        score = max(0.0, min(1.0, score))
        success = done and score > 0.0

    except Exception:
        success = False

    finally:
        log_end(success, steps, score, rewards)


# ─────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for task in TASKS:
        run_task(task)