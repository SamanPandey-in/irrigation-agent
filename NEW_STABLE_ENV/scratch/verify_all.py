import httpx
import asyncio
import os

async def verify():
    print("--- Verification Start ---")
    async with httpx.AsyncClient(base_url="http://localhost:7860") as client:
        # 1. Health
        r = await client.get("/health")
        print(f"Health: {r.status_code} {r.json()}")
        
        # 2. Reset
        r = await client.post("/reset", json={"task_name": "easy"})
        data = r.json()
        eid = data["observation"]["episode_id"]
        print(f"Reset: {r.status_code}, Episode ID: {eid}")
        
        # 3. State
        r = await client.get(f"/state?episode_id={eid}")
        print(f"State: {r.status_code} {r.json()}")
        
        # 4. Step
        r = await client.post("/step", json={"episode_id": eid, "action": {"action": 1}})
        print(f"Step: {r.status_code} {r.json().get('reward')}")
        
    print("--- Inference Script Smoke Test ---")
    # Set env vars for inference.py
    os.environ["HF_TOKEN"] = "dummy"
    os.environ["ENV_URL"] = "http://localhost:7860"
    os.environ["TASK_NAME"] = "easy"
    
    # We run inference.py as a subprocess and just check if it prints [START]
    import subprocess
    proc = subprocess.Popen(["python", "inference.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Wait for the first line
        out, err = proc.communicate(timeout=5)
        print("Inference Output Snippet:")
        for line in out.splitlines():
            if "[START]" in line:
                print(f"  FOUND: {line}")
            if "[STEP]" in line:
                print(f"  FOUND: {line}")
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        print("Inference Output (Timeout):")
        for line in out.splitlines():
            if "[START]" in line:
                print(f"  FOUND: {line}")
    
    print("--- Verification End ---")

if __name__ == "__main__":
    asyncio.run(verify())
