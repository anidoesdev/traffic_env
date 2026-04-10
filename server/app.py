"""
server/app.py -- FastAPI server for Smart City Traffic Flow OpenEnv.

Endpoints:
  GET  /health          health check
  POST /reset           start new episode
  POST /step            apply action, return obs + reward
  GET  /state           episode metadata
  GET  /tasks           list all tasks (returns catalog)
  POST /grader          grade a task by task_id, return score 0.0-1.0
  GET  /baseline        run all 3 graders, return reference scores
"""

import os
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from ..models import TrafficAction, TrafficObservation, TrafficState
    from .traffic_environment import TrafficEnvironment
    from .graders import (
        grade_easy, grade_medium, grade_hard,
        heuristic_policy,
    )
except ImportError:
    from models import TrafficAction, TrafficObservation, TrafficState
    from server.traffic_environment import TrafficEnvironment
    from server.graders import (
        grade_easy, grade_medium, grade_hard,
        heuristic_policy,
    )

TASK_LEVEL = os.environ.get("TASK_LEVEL", "easy")
env = TrafficEnvironment(task_level=TASK_LEVEL)

app = FastAPI(
    title="Smart City Traffic Flow -- OpenEnv",
    description="RL environment for adaptive traffic signal control.",
    version="1.0.0",
)

TASK_CATALOG = [
    {
        "id": "easy",
        "description": "Single intersection, 4 lanes, steady vehicle arrival rate",
        "difficulty": "easy",
        "max_steps": 100,
        "reward_range": [-1.0, 1.0],
        "graders": [
            {
                "id": "easy_score",
                "endpoint": "/grader",
                "method": "POST",
                "payload": {"task_id": "easy"},
            }
        ],
    },
    {
        "id": "medium",
        "description": "3-intersection corridor with rush-hour demand spike",
        "difficulty": "medium",
        "max_steps": 200,
        "reward_range": [-1.0, 1.0],
        "graders": [
            {
                "id": "medium_score",
                "endpoint": "/grader",
                "method": "POST",
                "payload": {"task_id": "medium"},
            }
        ],
    },
    {
        "id": "hard",
        "description": "3x3 grid of 9 intersections with random incidents",
        "difficulty": "hard",
        "max_steps": 300,
        "reward_range": [-1.0, 1.0],
        "graders": [
            {
                "id": "hard_score",
                "endpoint": "/grader",
                "method": "POST",
                "payload": {"task_id": "hard"},
            }
        ],
    },
]


# ── Standard OpenEnv endpoints ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "task_level": TASK_LEVEL}


@app.post("/reset", response_model=TrafficObservation)
def reset(seed: Optional[int] = None):
    try:
        return env.reset(seed=seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=TrafficObservation)
def step(action: TrafficAction):
    try:
        return env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=TrafficState)
def state():
    try:
        return env.state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def tasks():
    """Return the task catalog. Validator calls this to discover tasks."""
    return {"tasks": TASK_CATALOG}


# ── Grader endpoint ────────────────────────────────────────────────────────
# The validator calls POST /grader with {"task_id": "easy"} (or medium/hard)
# and expects a response containing a "score" field in [0.0, 1.0].

class GraderRequest(BaseModel):
    task_id: str
    n_seeds: Optional[int] = 3


@app.post("/grader")
def grader(request: GraderRequest):
    """
    Grade the specified task using the built-in heuristic policy.
    Returns a normalised score in [0.0, 1.0].

    Request body:
        {"task_id": "easy"}   or "medium" or "hard"

    Response:
        {"task_id": "easy", "score": 0.72, "min": 0.0, "max": 1.0}
    """
    task_id = request.task_id.lower().strip()
    n_seeds = max(1, min(request.n_seeds or 3, 5))

    grader_map = {
        "easy":   grade_easy,
        "medium": grade_medium,
        "hard":   grade_hard,
    }

    if task_id not in grader_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: easy, medium, hard.",
        )

    try:
        score = grader_map[task_id](policy=heuristic_policy, n_seeds=n_seeds)
        return {
            "task_id": task_id,
            "score":   round(float(score), 4),
            "min":     0.0,
            "max":     1.0,
            "n_seeds": n_seeds,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/baseline")
def baseline():
    """Run heuristic oracle on all 3 tasks. Returns reproducible reference scores."""
    try:
        results = {}
        for task_id, fn in [("easy", grade_easy), ("medium", grade_medium), ("hard", grade_hard)]:
            results[task_id] = round(float(fn(policy=heuristic_policy, n_seeds=3)), 4)
        return {"policy": "heuristic_oracle", "n_seeds": 3, "scores": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
