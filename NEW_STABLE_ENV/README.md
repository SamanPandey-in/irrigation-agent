---
title: Punjab Precision Irrigation
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# Punjab Precision Irrigation

An RL environment for optimising rice-paddy irrigation in Punjab, India. Agent controls daily irrigation level to maximise yield while minimising groundwater overuse under monsoon, drought and power-cut dynamics.

**Tags:** `openenv`

## Action Space
The environment uses a `Discrete(5)` action space:
- `0`: **no_water** - No irrigation today.
- `1`: **low** - 50L (10% pump capacity).
- `2`: **medium** - 150L (30% pump capacity).
- `3`: **high** - 250L (50% pump capacity).
- `4`: **drain** - Drain excess water (lowers soil moisture).

## Observation Space
The agent receives a `Dict` observation containing:
- `soil_moisture`: Current field saturation (0.0 - 1.0).
- `crop_growth`: Realized yield potential (0.0 - 1.0).
- `water_reserve`: Litres available in the local tube-well tank.
- `power_status`: Grid availability (1 = ON, 0 = OFF).
- `weather_forecast`: 3-day projection of rain, temp, and power-cut risk.
- `echoed_message`: A natural language summary of the state for LLM compatibility.

## Tasks
1. **Easy**: 30-day monsoon season (July). High rainfall, stable power.
2. **Medium**: 60-day pre-monsoon season (June). Occasional power cuts.
3. **Hard**: 90-day drought season (April-June). Frequent power cuts, high heat.

## Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Environment Server
```bash
python app.py
```

### 3. Run Baseline Inference
Ensure `HF_TOKEN` is set, then:
```bash
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export TASK_NAME="easy"
python inference.py
```

### Using Featherless AI Provider
If you prefer to use [Featherless AI](https://featherless.ai), set these environment variables:
```bash
export HF_TOKEN="your_featherless_api_key"
export API_BASE_URL="https://api.featherless.ai/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" 
python inference.py
```

## Hugging Face Deployment Guide
This environment is designed to run as a containerised **Hugging Face Space**.

### 1. Create a New Space
- Go to [huggingface.co/new-space](https://huggingface.co/new-space).
- **Space Name**: `punjab-precision-irrigation` (or your choice).
- **SDK**: Select **Docker**.
- **Template**: Choose **Blank** (it will use the `Dockerfile` in this repo).

### 2. Configure Secrets and Variables
Go to **Settings > Variables and secrets** in your Space:
- **HF_TOKEN** (Secret): Your Hugging Face or Featherless AI API Key.
- **API_BASE_URL** (Variable): `https://router.huggingface.co/v1` or `https://api.featherless.ai/v1`.
- **MODEL_NAME** (Variable): preferred model ID (e.g. `Qwen/Qwen2.5-72B-Instruct`).

### 3. Deploy
- Push this repository to the Hugging Face Space repository.
- The Space will automatically build the Docker image and start the server on port `7860`.

## Folder Structure

```
├── Dockerfile                  # Container configuration
├── README.md                   # Includes setup and description
├── openenv.yaml                # Project metadata and task definitions
├── requirements.txt            # Python dependencies
├── inference.py                # LLM-driven inference script (baseline)
├── models.py                   # Pydantic models for Actions, Observations, and State
├── app.py                      # Root entry point
├── server/
│   ├── app.py                  # FastAPI implementation
│   ├── environment.py          # Core simulation logic
│   └── graders.py               # Task grading logic
├── data/
│   └── imd_mock.py             # Mock IMD weather generator
├── gym_env/
│   └── precision_irrigation.py # Gymnasium-compatible wrapper
```
