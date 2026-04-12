---
title: Precision Crop Irrigation RL Agent
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
license: mit
tags:
  - openenv-v1
  - reinforcement-learning
  - openenv
  - gymnasium
  - agriculture
  - water-conservation
  - meta-hackathon
  - rl-environment
short_description: RL agent for smart irrigation — OpenEnv Hackathon
---

# 🌾 Precision Irrigation RL Agent

A **complete, from-scratch** Reinforcement Learning system for smart irrigation of Punjab rice farms. Built for the **OpenEnv Hackathon** — zero external ML dependencies.

> *"Train an agent to save groundwater for 140M Indian farmers."*

---

## ✅ All 5 Phases Complete

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | Core Gymnasium env · 7 unit tests · baseline agents |
| 2 | ✅ | POMDP v2 · crop stages · farm grid · flash floods · IMD API client |
| 3 | ✅ | Pure-NumPy PPO (Adam, GAE, clipped surrogate) · 100k steps |
| 4 | ✅ | Mock IMD HTTP API (`/forecast` `/summary` `/health`) |
| 5 | ✅ | Dashboard · policy heatmap · episode renders · ablations |

---

## Quick Start

```bash
git clone https://github.com/SamanPandey-in/irrigation-agent.git
cd irrigation-agent
pip install -r requirements.txt

# Generate weather data
python scripts/generate_data.py

# Run Phase 1 tests
python tests/test_env.py

# Train PPO + all baselines + ablations (100k steps, ~2 min CPU)
python scripts/train_and_eval.py

# Generate full dashboard
python viz/dashboard.py

# Start mock IMD weather API
python api/mock_imd_server.py   # → http://localhost:8765

# Run OpenEnv Inference (LLM-driven baseline)
# Requires: API_BASE_URL, MODEL_NAME, HF_TOKEN
# Optional: LOCAL_URL, TASKS, BENCHMARK, MAX_STEPS
python inference.py
```

### Docker

```bash
docker build -t irrigation-agent .
docker run irrigation-agent
```

---

## CLI config flags

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

## Architecture

```
📁 envs/
   precision_irrigation.py      Phase 1 — Gymnasium v0 env
   precision_irrigation_v2.py   Phase 2 — POMDP + farm grid
   precision_irrigation_v3.py   Phase 3 — Calibrated RL-ready env ← used for training

📁 rl/
   ppo.py                       Pure NumPy PPO (Adam, GAE, clipped surrogate)
   obs_normalizer.py            Hand-coded obs normalisation

📁 api/
   mock_imd_server.py           Built-in HTTP server, no fastapi needed

📁 viz/
   dashboard.py                 Phase 5 matplotlib dashboard

📁 scripts/
   generate_data.py             6yr Punjab IMD-style weather (2021-2026)
   train_and_eval.py            Full pipeline: train → eval → ablations

📁 server/
   app.py                       Multi-mode OpenEnv entrypoint (server:main)

📄 openenv.yaml                 Full environment & task specification
📄 inference.py                 LLM-driven inference script with [START]/[STEP]/[END] logging
📄 validator.py                 API compliance verification script

📁 data/
   rainfall.csv                 918 rows: 2021–2026, Jun–Oct season

📁 models/
   ppo_irrigation.pkl           Trained policy (pickle, ~290 KB)

📁 outputs/
   dashboard.png                Full dashboard image
   episode_render_ppo.png       PPO agent episode render
   episode_render_rb.png        Rule-Based agent episode render
   comparison_render.png        Side-by-side comparison
   metrics.json                 Full eval results + training log
```

---

## Environment: `PrecisionIrrigationEnvV3`

### Observation (Dict)
| Key | Shape | Description |
|-----|-------|-------------|
| `soil_moisture_obs` | (1,) | Noisy sensor reading [0,1] |
| `crop_growth` | (1,) | Growth fraction [0,1] |
| `crop_stage` | (1,) | 0=seedling 1=vegetative 2=reproductive 3=maturity |
| `weather_forecast` | (3,2) | 3-day noisy [[rain_mm, temp_C], …] |
| `water_tank` | (1,) | Normalised tank level |
| `power_status` | (1,) | 1=pump available |
| `day_of_season` | (1,) | Current day [0,90] |
| `grid_moisture` | (4,4) | Sub-field heterogeneous moisture |

### Actions (`Discrete 5`)
| # | Label | Volume |
|---|-------|--------|
| 0 | No water | 0 L |
| 1 | Low | ~1 000 L |
| 2 | Medium | ~3 000 L |
| 3 | High | ~5 000 L |
| 4 | Drain | remove excess |

Note: `grid_moisture` is an observational signal for heterogeneity/risk awareness, while actions are intentionally field-wide pump controls (single actuator). The policy uses grid spread to choose a balanced uniform action.

---

## Results (100k steps, 25 eval episodes)

### Normalized Task Scores (0 to 1)

- Task 1 (Normal):  0.98  (easy — agent succeeds reliably)
- Task 2 (Drought): 0.53  (medium — requires strategic irrigation)
- Task 3 (Hard):    0.16  (hard — crop fails under adversarial conditions)

| Agent | Reward | Yield t/ha | Water kL | Efficiency |
|-------|--------|------------|----------|------------|
| Random | +11.68 | 5.06 | 92.9 | 0.054 |
| Rule-Based | +14.56 | 6.22 | 92.6 | 0.067 |
| Greedy | +15.28 | 6.31 | 95.3 | 0.066 |
| Smart Forecast | +12.75 | 5.67 | 76.3 | 0.074 |
| **PPO (trained) ★** | **+15.37** | **6.33** | **95.3** | **0.066** |

### Ablations
| Scenario | Reward | Yield |
|----------|--------|-------|
| PPO — no forecast | +15.46 | 6.35 t |
| PPO — drought (−75% rain) | +13.06 | 5.54 t |
| Forecast — drought | +11.57 | 5.24 t |
| Rule-Based — drought | +12.41 | 5.45 t |

> PPO shows **drought robustness** (+5.5% yield over Smart Forecast in drought).

---

## PPO Implementation (Pure NumPy)

No PyTorch. No TensorFlow. No gymnasium required.

- `ActorCritic` — shared MLP trunk (256×256 tanh) + actor/critic heads
- `Adam` optimiser — stateful per-parameter m/v moments
- `GAE` — generalised advantage estimation (γ=0.99, λ=0.95)
- `Clipped surrogate` — PPO-clip (ε=0.2)
- `Entropy bonus` — annealed 0.05→0.005
- `Reward clipping` — [-10, 10] during training
- **~870 steps/sec** on a single CPU core

---

## OpenEnv REST API

The application exposes a standard FastAPI endpoint under `/` for automated evaluation.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | `POST` | Reset env with `{"task_id": "task_1", "seed": 42}` |
| `/step`  | `POST` | Take action `{"action": 2}`. Returns score in [0.0, 1.0] |
| `/state` | `GET`  | Get current observation dict |
| `/health`| `GET`  | Check server status and version |

---

## OpenEnv Multi-Mode Deployment

This repo is multi-mode compatible and passes `openenv validate`.

- `pyproject.toml` includes `[project.scripts] server = "server.app:main"`
- `server/app.py` provides a callable `main()` entrypoint
- `uv.lock` is generated via `uv lock`

Run locally with:
```bash
python server/app.py
```

Validate:
```bash
openenv validate
```

---

## Mock IMD API

```bash
python api/mock_imd_server.py
curl "http://localhost:8765/forecast?year=2023&day=15&days=3"
curl "http://localhost:8765/summary?year=2022"
curl "http://localhost:8765/health"
```

---

## Future Scope

- **Phase 6**: Multi-agent MARL — neighbourhood farms sharing canal resources
- **Real IoT**: Raspberry Pi soil sensors → live obs
- **Mobile deploy**: Export policy → TFLite → Flutter Kisan app
- **Research**: Hierarchical RL, weather-LLM hybrid forecasts

---

## License · MIT
