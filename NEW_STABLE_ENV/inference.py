# inference.py
import asyncio
import os
import textwrap
import httpx
from typing import List, Optional
from openai import OpenAI

# Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Task Configuration
TASK_NAME = os.getenv("TASK_NAME", "easy")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
MAX_STEPS = 90  # Maximum possible steps for 'hard' task

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI agent managing a rice-paddy farm in Punjab, India.
    Your goal is to optimize daily irrigation to maximize crop growth while conserving water and minimizing stress.
    
    Actions:
    0: no_water - No irrigation.
    1: low - 50L (10% pump capacity).
    2: medium - 150L (30% pump capacity).
    3: high - 250L (50% pump capacity).
    4: drain - Drain excess water if soil moisture > 0.70.

    Strategy:
    - Keep soil moisture between 0.45 and 0.70.
    - Check the 3-day weather forecast; if rain is expected, save water.
    - Be mindful of power cuts (0 = grid OFF, 1 = grid ON); you cannot pump water during a cut.
    - Monitor water_reserve; it recharges with rain but is finite.

    Reply with exactly one number (0, 1, 2, 3, or 4) — nothing else.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    clean_score = max(0.001, min(0.999, score))
    print(f"[END] success={str(success).lower()} steps={steps} score={clean_score:.3f} rewards={rewards_str}", flush=True)

async def get_model_action(client: OpenAI, obs: dict) -> int:
    prompt = f"Day {obs['day']}/{obs['total_days']}\nSoil Moisture: {obs['soil_moisture']:.2f}\nCrop Growth: {obs['crop_growth']:.2f}\nWater Reserve: {obs['water_reserve']:.1f}L\nPower Status: {obs['power_status']}\nForecast (Next 3 Days): {obs['weather_forecast']}\nLast Result: {obs['last_action_result']}\n\nDecision (0-4):"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        content = (completion.choices[0].message.content or "").strip()
        # Extract the first digit found
        for char in content:
            if char.isdigit() and char in "01234":
                return int(char)
        return 0
    except Exception as e:
        # print(f"[DEBUG] LLM Error: {e}")
        return 0

async def main():
    if not HF_TOKEN:
        print("Error: HF_TOKEN or API_KEY environment variable is required.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    async with httpx.AsyncClient(timeout=30.0) as http:
        # Reset Environment
        try:
            resp = await http.post(f"{ENV_URL}/reset", json={"task_name": TASK_NAME})
            resp.raise_for_status()
            data = resp.json()
            obs = data["observation"]
            episode_id = obs["episode_id"]
        except Exception as e:
            print(f"Error connecting to environment at {ENV_URL}: {e}")
            return

        log_start(task=TASK_NAME, env="punjab-precision-irrigation", model=MODEL_NAME)

        rewards = []
        steps_taken = 0
        success = False
        final_score = 0.001
        
        try:
            for step in range(1, MAX_STEPS + 1):
                action_idx = await get_model_action(client, obs)
                
                resp = await http.post(f"{ENV_URL}/step", json={
                    "episode_id": episode_id,
                    "action": {"action": action_idx}
                })
                resp.raise_for_status()
                step_data = resp.json()
                
                obs = step_data["observation"]
                reward = step_data.get("reward", 0.0)
                done = step_data.get("done", False)
                
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=str(action_idx), reward=reward, done=done, error=None)
                
                if done:
                    final_score = step_data.get("score", 0.0)
                    success = final_score > 0.7  # Basic threshold for success log
                    break
        except Exception as e:
            log_step(step=steps_taken+1, action="error", reward=0.0, done=True, error=str(e))
        
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
