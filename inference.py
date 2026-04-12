Copy

import sys
import os
import time
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
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("FEATHERLESS_API_KEY")
    or os.getenv("HF_TOKEN")
)
 
LOCAL_URL  = os.getenv("LOCAL_URL",  "http://localhost:7860")
BENCHMARK  = os.getenv("BENCHMARK",  "precision-irrigation-agent")
TASKS      = [t.strip() for t in os.getenv("TASKS", "task_1,task_2,task_3").split(",") if t.strip()]
MAX_STEPS  = _safe_int(os.getenv("MAX_STEPS", "100"), 100)
 
# How long to wait for the server to become ready before giving up
SERVER_WAIT_SECS   = int(os.getenv("SERVER_WAIT_SECS", "180"))
SERVER_RETRY_DELAY = 5   # seconds between each health-check retry
 
# ─────────────────────────────────────────────────────────────
# OpenAI Client (safe init)
# ─────────────────────────────────────────────────────────────
 
try:
    client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY or "dummy-key")
except Exception:
    client = None
 
 
# ─────────────────────────────────────────────────────────────
# Server readiness wait
# ─────────────────────────────────────────────────────────────
 
def _wait_for_server(url: str, timeout: int = SERVER_WAIT_SECS) -> bool:
    """
    Poll /health until the environment server responds 200.
    Returns True if ready, False if timed out.
    This handles the case where the Docker container starts but the
    FastAPI/Gradio app inside it hasn't finished loading yet
    (e.g. auto-training the PPO model on first run can take 2-3 min).
    """
    deadline = time.time() + timeout
    attempt = 0
    print(f"[DEBUG] Waiting up to {timeout}s for server at {url} ...", flush=True)
    while time.time() < deadline:
        attempt += 1
        try:
            r = requests.get(f"{url}/health", timeout=10)
            if r.status_code == 200:
                elapsed = int(time.time() - (deadline - timeout))
                print(f"[DEBUG] Server ready after {attempt} attempt(s) ({elapsed}s elapsed)", flush=True)
                return True
            else:
                print(f"[DEBUG] /health returned {r.status_code} (attempt {attempt})", flush=True)
        except requests.exceptions.ConnectionError:
            # Container still starting — expected during PPO auto-train
            print(f"[DEBUG] Connection refused (attempt {attempt}) — server still starting", flush=True)
        except Exception as e:
            print(f"[DEBUG] Server not ready yet (attempt {attempt}): {type(e).__name__}: {e}", flush=True)
        time.sleep(SERVER_RETRY_DELAY)
    print(f"[DEBUG] Server did not become ready within {timeout}s after {attempt} attempts", flush=True)
    return False
 
 
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
    obs: dict,
    prev_reward=None,
    total_reward: float = 0.0,
) -> str:
    moisture = _first(obs.get("soil_moisture_obs"), 0.5)
    growth   = _first(obs.get("crop_growth"),       0.0)
    stage    = int(_first(obs.get("crop_stage"),    0))
    tank     = _first(obs.get("water_tank"),        0.0)
    power    = int(_first(obs.get("power_status"),  1))
    day      = int(_first(obs.get("day_of_season"), 0))
    forecast = obs.get("weather_forecast", [])
 
    rain_tomorrow = 0.0
    try:
        if isinstance(forecast, list) and len(forecast) > 0:
            row = forecast[0]
            if isinstance(row, list) and len(row) > 0:
                rain_tomorrow = float(row[0])
    except Exception:
        pass
 
    stage_names = {0: "Seedling", 1: "Vegetative", 2: "Reproductive", 3: "Maturity"}
 
    if power == 0:
        urgency = "PUMP IS OFF — irrigation impossible today. Choose 0."
    elif moisture > 0.78:
        urgency = "WATERLOGGED (moisture > 0.78) — choose 4 (Drain) immediately."
    elif moisture < 0.22:
        urgency = "SEVERE DRY STRESS (moisture < 0.22) — choose 3 (High) immediately."
    elif moisture < 0.40:
        urgency = "DRY (moisture < 0.40) — choose 2 (Medium) unless heavy rain forecast."
    elif moisture > 0.72:
        urgency = "ABOVE OPTIMAL — choose 0 (No Water) or 4 (Drain)."
    else:
        urgency = "IN OPTIMAL BAND — choose 0 or 1 depending on forecast."
 
    prev_str = f"{prev_reward:.3f}" if prev_reward is not None else "N/A (first step)"
 
    return (
        "IRRIGATION CONTROLLER — respond with ONE digit only: 0, 1, 2, 3, or 4.\n\n"
        "ACTIONS:\n"
        "  0 = No Water\n"
        "  1 = Low irrigation (~1 kL)\n"
        "  2 = Medium irrigation (~3 kL)\n"
        "  3 = High irrigation (~5 kL)\n"
        "  4 = Drain excess water\n\n"
        "CURRENT STATE:\n"
        f"  Day: {day}/90 | Stage: {stage_names.get(stage, str(stage))}\n"
        f"  Soil moisture: {moisture:.3f}  (optimal: 0.40-0.72)\n"
        f"  Crop growth: {growth:.3f} | Tank level: {tank:.2f}\n"
        f"  Power: {'ON' if power == 1 else 'OFF (pump unavailable)'}\n"
        f"  Rain tomorrow forecast: {rain_tomorrow:.1f} mm\n\n"
        f"REWARD FEEDBACK:\n"
        f"  Previous step reward: {prev_str}\n"
        f"  Running total reward: {total_reward:.3f}\n\n"
        f"DECISION: {urgency}\n\n"
        "YOUR ANSWER (single digit 0-4):"
    )
 
 
def _parse_action(raw: str) -> int:
    try:
        digits = "".join(c for c in raw if c.isdigit())
        if not digits:
            return 0
        return max(0, min(4, int(digits[0])))
    except Exception:
        return 0
 
 
# ─────────────────────────────────────────────────────────────
# Logging  (STRICT FORMAT — do not change field names/order)
# ─────────────────────────────────────────────────────────────
 
def _bool(v: bool) -> str:
    return "true" if v else "false"
 
 
def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)
 
 
def log_step(step: int, action: int, reward: float, done: bool, error=None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={_bool(done)} error={error if error else 'null'}",
        flush=True,
    )
 
 
def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_text = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_bool(success)} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )
 
 
# ─────────────────────────────────────────────────────────────
# Core Runner  — every network call is wrapped
# ─────────────────────────────────────────────────────────────
 
def run_task(task_id: str) -> None:
    log_start(task_id)
 
    steps        = 0
    rewards: list = []
    total_reward = 0.0
    prev_reward  = None
    score        = 0.0
    success      = False
    result       = {}
 
    try:
        # ── 1. Reset environment ──────────────────────────────
        try:
            r = requests.post(
                f"{LOCAL_URL}/reset",
                json={"task_id": task_id},
                timeout=30,
            )
            r.raise_for_status()
            obs = r.json()["observation"]
        except Exception as e:
            print(f"[DEBUG] /reset failed for {task_id}: {e}", flush=True)
            # Can't run — emit [END] with zero score and continue to next task
            return
 
        done = False
 
        while not done and steps < MAX_STEPS:
 
            # ── 2. Choose action (LLM call, fully protected) ──
            action = 0
            try:
                if client is not None:
                    resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{
                            "role": "user",
                            "content": _build_prompt(obs, prev_reward, total_reward),
                        }],
                        temperature=0.0,
                        max_tokens=5,
                    )
                    raw    = (resp.choices[0].message.content or "").strip()
                    action = _parse_action(raw)
            except Exception as e:
                print(f"[DEBUG] LLM call failed: {e}", flush=True)
                action = 0
 
            # ── 3. Step environment (fully protected) ─────────
            try:
                r = requests.post(
                    f"{LOCAL_URL}/step",
                    json={"action": action},
                    timeout=30,
                )
                r.raise_for_status()
                result = r.json()
            except Exception as e:
                print(f"[DEBUG] /step failed at step {steps+1}: {e}", flush=True)
                # Treat as terminal — break cleanly
                log_step(steps + 1, action, 0.0, True, str(e))
                steps += 1
                rewards.append(0.0)
                break
 
            # ── 4. Parse response ─────────────────────────────
            try:
                obs        = result["observation"]
                reward     = float(result.get("reward", 0.0))
                terminated = bool(result.get("terminated", False))
                truncated  = bool(result.get("truncated",  False))
                done       = terminated or truncated
                err        = result.get("info", {}).get("last_action_error") if isinstance(result.get("info"), dict) else None
            except Exception as e:
                print(f"[DEBUG] Response parse error: {e}", flush=True)
                reward, done, err = 0.0, True, str(e)
 
            steps        += 1
            rewards.append(reward)
            total_reward += reward
            prev_reward   = reward
 
            log_step(steps, action, reward, done, err)
 
        # ── 5. Compute final score ────────────────────────────
        try:
            if result:
                score = float(result.get("score", 0.0))
        except Exception:
            score = 0.0
 
        # Fallback if server didn't return a grader score
        if score == 0.0 and rewards:
            try:
                total = sum(rewards)
                score = total / (abs(total) + 1e-6)
                score = (score + 1) / 2          # map [-1,1] → [0,1]
            except Exception:
                score = 0.0
 
        score   = max(0.001, min(0.999, score))  # strict (0, 1) exclusive
        success = done and score > 0.0
 
    except Exception as e:
        # Absolute last-resort catch — should never reach here
        print(f"[DEBUG] Unexpected error in run_task({task_id}): {e}", flush=True)
        success = False
 
    finally:
        # Always emits [END] — even on any exception above
        log_end(success, steps, score, rewards)
 
 
# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    try:
        print(f"[DEBUG] LOCAL_URL={LOCAL_URL}  MODEL={MODEL_NAME}", flush=True)
        print(f"[DEBUG] API key present: {'YES' if OPENAI_API_KEY else 'NO'}", flush=True)
 
        # Wait for the environment server to be ready before starting.
        # This is critical when both containers start at the same time —
        # inference.py must not race ahead of app.py's startup.
        server_ready = _wait_for_server(LOCAL_URL, timeout=SERVER_WAIT_SECS)
        if not server_ready:
            print("[DEBUG] WARNING: server never became ready — attempting tasks anyway", flush=True)
 
        for task in TASKS:
            try:
                run_task(task)
            except Exception as e:
                # Per-task safety net — always emit [END] and continue
                print(f"[DEBUG] run_task({task}) crashed: {e}", flush=True)
                log_end(False, 0, 0.0, [])
 
    except Exception as e:
        print(f"[DEBUG] Top-level crash: {e}", flush=True)
 
    # Always exit 0 — inference.py must never exit with non-zero status
    sys.exit(0)