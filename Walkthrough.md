Here’s a detailed explanation of how the precision irrigation RL agent project works, how to use it, and where to find outputs and metrics:

---

## 1. **Project Overview & Architecture**

- **Purpose:** A full reinforcement learning (RL) system for smart irrigation of Punjab rice farms, built for the OpenEnv Hackathon. It has no external ML dependencies (no PyTorch, TensorFlow, or gymnasium required).
- **Phases:** The project is organized into 5 phases, from environment and baseline agents to a dashboard and ablation studies.
- **Key Folders:**
  - envs: RL environments (v1, v2, v3)
  - rl: RL algorithms (PPO, normalization)
  - api: Mock IMD weather API server
  - viz: Dashboard and visualization scripts
  - scripts: Data generation, training, and evaluation scripts
  - server: Multi-mode OpenEnv entrypoint (server/app.py)
  - openenv.yaml: Mandatory task and environment specification
  - inference.py: Mandatory LLM inference pipeline with [START]/[STEP]/[END] logging
  - outputs: All results, metrics, and dashboard images
  - models: Trained RL models

---

## 2. **How to Run the Project**

### **Setup**
```bash
git clone https://github.com/SamanPandey-in/irrigation-agent.git
cd irrigation-agent
pip install numpy pandas matplotlib
```
Recommended:
```bash
pip install -r requirements.txt
```

### **Data Generation**
```bash
python scripts/generate_data.py
```
- Generates 6 years of Punjab IMD-style weather data (rainfall.csv).

### **Testing the Environment**
```bash
python tests/test_env.py
```
- Runs unit tests on the RL environment to ensure everything is working.

### **Training and Evaluation**
```bash
python scripts/train_and_eval.py
```
- Trains the PPO agent and all baselines, runs evaluations, and saves metrics and renders.
Quick mode (for deployment checks):
```bash
python scripts/train_and_eval.py --quick --quick-steps 25000 --quick-eval 10
```

### **Dashboard Generation**
```bash
python viz/dashboard.py
```
- Produces a multi-panel dashboard image and episode renders in outputs.

### **Mock Weather API**
```bash
python api/mock_imd_server.py
```
- Starts a local HTTP server for weather data (for advanced use).

### **OpenEnv Inference (Baseline)**
```bash
# Requires API_BASE_URL, MODEL_NAME, HF_TOKEN in env
# Optional: LOCAL_URL, TASKS, BENCHMARK, MAX_STEPS
python inference.py
```
- Runs an LLM-driven agent through 3 tasks (Normal, Drought, Flood).
- Emits structured logs for automated scoring.

---

## 3. **OpenEnv Compliance**

- **Specification (`openenv.yaml`):** Defines types for observation (dict with Box/Discrete) and action (Discrete 5) spaces.
- **REST API:** The application runs a dual FastAPI + Gradio server on port 7860.
  - `POST /reset`: Initialize a specific task preset.
  - `POST /step`: Execute irrigation action and receive next obs + reward + 0–1 score.
- **Structured Logging:** `inference.py` emits strict key-value lines:
  - `[START] task=... env=... model=...`
  - `[STEP] step=... action=... reward=0.00 done=true|false error=null`
  - `[END] success=true|false steps=... score=0.000 rewards=0.00,...`
- **Multi-Mode Entry:** `server/app.py` defines `main()` and is referenced by `[project.scripts] server` in `pyproject.toml`.
- **Validation:** run `uv lock` and `openenv validate`.

To run the local validator, the API must be live on port 7860:
```bash
python server/app.py   # serves on http://localhost:7860
python validator.py
```

## 4. **Custom Inputs for Test Environments**

- **Test Environment:** See tests/test_env.py.
- **Custom Inputs:** You can modify or extend the test cases in this file. The environment is created via `make_env()`, which can accept a `render_mode` or be customized by editing the environment constructors in envs/.
- **Direct Run:** You can also run demo.py with arguments:
  - `python demo.py --episodes 5` (run 5 episodes per agent)
  - `python demo.py --render` (show live matplotlib window)

---

## 5. **Where and How to See Outputs & Metrics**

- **Terminal Output:** Training and evaluation scripts print metrics tables and logs directly to the terminal.
- **Saved Files:**
  - **Metrics:** outputs/metrics.json — contains full evaluation results and training logs.
  - **Dashboard:** outputs/dashboard.png — main dashboard image.
  - **Episode Renders:** outputs/episode_render_ppo.png, outputs/episode_render_rb.png
  - **Comparison:** outputs/comparison_render.png

---

## 6. **UI / Dashboard**

- **Dashboard Script:** viz/dashboard.py
  - Generates a multi-panel dashboard with:
    - Training reward/yield curves
    - Agent comparison bar charts
    - Ablation study table
    - Policy heatmap
    - Impact statement
  - Output is a PNG image, not a web UI.

---

## 7. **How to Enter Custom Inputs**

- **For Testing:** Edit tests/test_env.py to add or modify test cases.
- **For Training/Evaluation:** You can pass arguments to scripts, or modify the environment and agent logic in scripts/train_and_eval.py and envs/precision_irrigation_v3.py.
- **For Demo:** Use command-line arguments with demo.py.

---

## 8. **CLI config flags**

```
python scripts/evaluate_cli.py --mode batch \
  --actions "2,3,2,3,2,2,3" \
  --preset drought \
  --rain-mult 0.1 \
  --sensor-noise 0.08 \
  --api-key sk-ant-...
```

### Or load/save a full JSON config:
```
python scripts/evaluate_cli.py --preset hard --dump-config > hard.json
python scripts/evaluate_cli.py --mode interactive --config-file hard.json
```

---

## 9. **Summary Table**

| Task                | How to Do It                                                                 | Output Location                        |
|---------------------|------------------------------------------------------------------------------|----------------------------------------|
| Run tests           | `python tests/test_env.py`                                                   | Terminal                               |
| Run demo            | `python demo.py --episodes 5`                                                | Terminal, PNG render                   |
| Train & eval        | `python scripts/train_and_eval.py`                                           | Terminal, outputs/metrics.json, renders|
| Generate dashboard  | `python viz/dashboard.py`                                                    | dashboard.png                  |
| Start mock API      | `python api/mock_imd_server.py`                                              | http://localhost:8765                  |
| Start OpenEnv API   | `python server/app.py`                                                       | http://localhost:7860                  |
| Run OpenEnv Baseline| `python inference.py`                                                        | Stdout (START/STEP/END logs)           |
| Validate API        | `python validator.py` (requires server on port 7860)                        | http://localhost:7860                  |
| Custom test inputs  | Edit test_env.py or precision_irrigation_v3.py                               | Terminal, metrics.json         |

---
