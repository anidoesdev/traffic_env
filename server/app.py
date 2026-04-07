"""
server/app.py -- FastAPI HTTP server for the Traffic Flow OpenEnv environment.

Standard OpenEnv endpoints:
  GET  /health   -- health check
  POST /reset    -- start new episode
  POST /step     -- apply action, return observation + reward
  GET  /state    -- episode metadata

Grader endpoints (one per task, declared in openenv.yaml):
  GET  /grade/easy    -- run grader for easy task, return score 0.0-1.0
  GET  /grade/medium  -- run grader for medium task, return score 0.0-1.0
  GET  /grade/hard    -- run grader for hard task, return score 0.0-1.0

Discovery endpoints:
  GET  /tasks    -- list all tasks
  GET  /baseline -- run all 3 graders, return reproducible scores
"""

import os
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException

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


# ── Per-task grader endpoints (declared in openenv.yaml grader.path) ───────
# The validator reads openenv.yaml, finds grader.path for each task,
# then calls that path and expects {"score": float} in the response.
# Each endpoint runs the grader and returns the result immediately.

@app.get("/grade/easy")
def grade_task_easy():
    """
    Grader for the easy task (single intersection).
    Returns normalised score: 0.0 = random policy, 1.0 = heuristic oracle.
    Uses 3 seeds for speed. Full evaluation uses 5.
    """
    try:
        score = grade_easy(policy=heuristic_policy, n_seeds=3)
        return {
            "task_id": "easy",
            "score": round(score, 4),
            "min": 0.0,
            "max": 1.0,
            "description": "Single intersection, 4 lanes, steady arrival rate",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/grade/medium")
def grade_task_medium():
    """
    Grader for the medium task (3-intersection corridor).
    Returns normalised score: 0.0 = random policy, 1.0 = heuristic oracle.
    """
    try:
        score = grade_medium(policy=heuristic_policy, n_seeds=3)
        return {
            "task_id": "medium",
            "score": round(score, 4),
            "min": 0.0,
            "max": 1.0,
            "description": "3-intersection corridor with rush-hour spike",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/grade/hard")
def grade_task_hard():
    """
    Grader for the hard task (9-intersection grid).
    Returns normalised score: 0.0 = random policy, 1.0 = heuristic oracle.
    """
    try:
        score = grade_hard(policy=heuristic_policy, n_seeds=3)
        return {
            "task_id": "hard",
            "score": round(score, 4),
            "min": 0.0,
            "max": 1.0,
            "description": "3x3 grid of 9 intersections with random incidents",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Discovery endpoints ─────────────────────────────────────────────────────

@app.get("/tasks")
def tasks():
    """List all 3 tasks with grader paths."""
    return {
        "tasks": [
            {
                "id": "easy",
                "description": "Single intersection, 4 lanes, steady arrival rate",
                "difficulty": "easy",
                "max_steps": 100,
                "grader_path": "/grade/easy",
            },
            {
                "id": "medium",
                "description": "3-intersection corridor with rush-hour spike at step 50",
                "difficulty": "medium",
                "max_steps": 200,
                "grader_path": "/grade/medium",
            },
            {
                "id": "hard",
                "description": "3x3 grid of 9 intersections with random incidents",
                "difficulty": "hard",
                "max_steps": 300,
                "grader_path": "/grade/hard",
            },
        ]
    }


@app.get("/baseline")
def baseline():
    """Run heuristic oracle on all 3 tasks. Reproducible reference scores."""
    try:
        scores = {
            "easy":   round(grade_easy(policy=heuristic_policy,   n_seeds=3), 4),
            "medium": round(grade_medium(policy=heuristic_policy, n_seeds=3), 4),
            "hard":   round(grade_hard(policy=heuristic_policy,   n_seeds=3), 4),
        }
        return {"policy": "heuristic_oracle", "n_seeds": 3, "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
