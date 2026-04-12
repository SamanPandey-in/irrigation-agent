import requests
import json
import sys

URL = "http://localhost:7860"

def test_api():
    print("🔍 Testing OpenEnv API compliance...")
    
    try:
        # 1. Health check
        r = requests.get(f"{URL}/health")
        r.raise_for_status()
        print(f"✅ Health: {r.json()}")
        
        # 2. Reset (Task 1)
        r = requests.post(f"{URL}/reset", json={"task_id": "task_1"})
        r.raise_for_status()
        obs = r.json()["observation"]
        print("✅ Reset (task_1) successful")
        
        # 3. State check
        r = requests.get(f"{URL}/state")
        r.raise_for_status()
        assert r.json()["observation"] == obs
        print("✅ State retrieval matches last reset")
        
        # 4. Step
        r = requests.post(f"{URL}/step", json={"action": 0})
        r.raise_for_status()
        data = r.json()
        assert "reward" in data
        assert "observation" in data
        print("✅ Step successful")
        
        # 5. Task 2 Reset
        r = requests.post(f"{URL}/reset", json={"task_id": "task_2"})
        r.raise_for_status()
        print("✅ Reset (task_2) successful")
        
        print("\n🎉 API looks OpenEnv-ready!")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_api()
