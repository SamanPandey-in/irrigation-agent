import urllib.request
import json

BASE_URL = "http://localhost:7860"

def test_endpoint(name, method, path, data=None):
    print(f"Testing {name}...")
    url = f"{BASE_URL}{path}"
    try:
        req = urllib.request.Request(url, method=method)
        if data:
            req.add_header('Content-Type', 'application/json')
            req.data = json.dumps(data).encode('utf-8')
        
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            print(f"Status: {status}")
            if status == 200:
                resp_data = json.loads(response.read().decode('utf-8'))
                print(f"Response: {json.dumps(resp_data, indent=2)[:100]}...")
                return resp_data
            else:
                print(f"Error: {response.read().decode('utf-8')}")
    except Exception as e:
        print(f"Exception: {e}")
    return None

if __name__ == "__main__":
    test_endpoint("Health", "GET", "/health")
    test_endpoint("Metadata", "GET", "/metadata")
    reset_data = test_endpoint("Reset", "POST", "/reset", {"task_name": "easy"})
    if reset_data:
        episode_id = reset_data["observation"]["episode_id"]
        test_endpoint("Step", "POST", "/step", {"episode_id": episode_id, "action": {"action": 2}})
