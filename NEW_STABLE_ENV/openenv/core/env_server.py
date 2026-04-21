# openenv/core/env_server.py
from pydantic import BaseModel
from typing import Optional
class Action(BaseModel): pass
class Observation(BaseModel):
    reward: Optional[float] = None
    done: bool = False
class State(BaseModel):
    episode_id: str = ""
class Environment:
    def reset(self, **kwargs): pass
    def step(self, action, **kwargs): pass
