# server/app.py
import os, sys, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from server.environment import PunjabPrecisionIrrigationEnvironment
from server.graders import grade_task
from models import IrrigationAction

app = FastAPI(title="Punjab Precision Irrigation")
ENV_STORE = {}
class ResetRequest(BaseModel):
    task_name: str = "easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None
class StepRequest(BaseModel):
    episode_id: str
    action: IrrigationAction

def _obs_to_dict(obs, reward=None, done=False):
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": {}}

@app.get("/")
def root(): return RedirectResponse(url="/docs")

@app.post("/reset")
def reset(req: ResetRequest = None):
    if req is None: req = ResetRequest()
    env = PunjabPrecisionIrrigationEnvironment()
    obs = env.reset(task_name=req.task_name, seed=req.seed, episode_id=req.episode_id)
    ENV_STORE[obs.episode_id] = {"env": env, "rewards": [], "steps": 0}
    return _obs_to_dict(obs)

@app.post("/step")
def step(req: StepRequest):
    entry = ENV_STORE.get(req.episode_id)
    if not entry: return JSONResponse(status_code=404, content={"error": "Not found"})
    obs = entry["env"].step(req.action)
    entry["rewards"].append(obs.reward)
    entry["steps"] += 1
    if obs.done:
        res = _obs_to_dict(obs, obs.reward, True)
        res["score"] = grade_task(entry["env"].state.task_name, res["observation"], entry["rewards"], entry["steps"])
        ENV_STORE.pop(req.episode_id)
        return res
    return _obs_to_dict(obs, obs.reward, False)

@app.get("/state")
def state(episode_id: str):
    entry = ENV_STORE.get(episode_id)
    if not entry: return JSONResponse(status_code=404, content={"error": "Not found"})
    return entry["env"].state.model_dump()
@app.get("/health")
def health(): return {"status": "healthy"}
@app.get("/metadata")
def metadata():
    return {
        "name": "punjab-precision-irrigation", 
        "description": "Punjab Rice Farm Precision Irrigation", 
        "tasks": [
            "easy", "easy_1", "easy_2", "easy_3",
            "medium", "medium_1", "medium_2", "medium_3",
            "hard", "hard_1", "hard_2", "hard_3"
        ]
    }

@app.get("/schema")
def schema():
    from models import IrrigationAction, IrrigationObservation, IrrigationState
    return {
        "action": IrrigationAction.model_json_schema(),
        "observation": IrrigationObservation.model_json_schema(),
        "state": IrrigationState.model_json_schema()
    }
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
