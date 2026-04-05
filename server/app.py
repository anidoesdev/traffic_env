"""
server/app.py — FastAPI HTTP server.

This file wraps TrafficEnvironment in a web server so any client
(an LLM agent, a training loop, the openenv CLI) can talk to it
over standard HTTP.

Endpoints:
  GET  /health  → {"status": "healthy"}
  POST /reset   → TrafficObservation  (starts new episode)
  POST /step    → TrafficObservation  (applies action, returns next obs + reward)
  GET  /state   → TrafficState        (episode metadata)

TASK_LEVEL is read from the TASK_LEVEL environment variable.
Set it to "easy", "medium", or "hard" before starting the server.
Default is "easy".

Example:
  TASK_LEVEL=hard uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException

#  Dual-import pattern ----------------------------
# Try relative first (works in-repo with PYTHONPATH=src:envs)
# Fall back to bare import (works inside Docker with PYTHONPATH=/app)
try:
    from ..models import TrafficAction, TrafficObservation, TrafficState
    from .traffic_environment import TrafficEnvironment
except ImportError:
    from models import TrafficAction, TrafficObservation, TrafficState
    from server.traffic_environment import TrafficEnvironment

# Environment setup ----------------------------
# Read task level from environment variable (set in Docker / openenv.yaml)
TASK_LEVEL = os.environ.get("TASK_LEVEL", "easy")
env = TrafficEnvironment(task_level=TASK_LEVEL)

#  FastAPI app ----------------------------
app = FastAPI(
    title="Smart City Traffic Flow — OpenEnv",
    description=(
        "RL environment for adaptive traffic signal control. "
        "An agent controls signal phases to minimise vehicle wait time and "
        "maximise throughput across urban intersections."
    ),
    version="1.0.0",
)


# -- Endpoints ----------------------------

@app.get("/health")
def health():
    """
    Health check — used by Docker HEALTHCHECK and openenv validate.
    Must return 200 with {"status": "healthy"} for the validator to pass.
    """
    return {"status": "healthy", "task_level": TASK_LEVEL}


@app.post("/reset", response_model=TrafficObservation)
def reset(seed: int = None):
    """
    Start a fresh episode.
    Pass ?seed=42 to get a reproducible episode.
    Returns the initial observation (no action taken yet).
    """
    try:
        obs = env.reset(seed=seed)
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=TrafficObservation)
def step(action: TrafficAction):
    """
    Apply one action and advance the simulation by 5 seconds.
    Returns the new observation including the reward for this step.

    Action body example:
        {"action_type": "extend_green", "intersection_id": 0}
        {"action_type": "next_phase",   "intersection_id": 2}
    """
    try:
        obs = env.step(action)
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=TrafficState)
def state():
    """
    Return episode-level metadata (step count, cumulative reward, etc.)
    This is the OpenEnv state() method exposed over HTTP.
    """
    try:
        return env.state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
