"""
server/app.py -- FastAPI HTTP server.

Endpoints required by the hackathon validator:
  GET  /health         health check
  POST /reset          start new episode
  POST /step           apply action, return observation + reward
  GET  /state          episode metadata
  GET  /tasks          list all tasks with descriptions  <-- REQUIRED by validator
  POST /grader         grade a full episode for a task   <-- REQUIRED by validator
  GET  /baseline       run heuristic on all 3 tasks      <-- REQUIRED by validator
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

try:
    from ..models import TrafficAction, TrafficObservation, TrafficState
    from .traffic_environment import TrafficEnvironment
    from .graders import grade_easy, grade_medium, grade_hard, heuristic_policy, run_episode
except ImportError:
    from models import TrafficAction, TrafficObservation, TrafficState
    from server.traffic_environment import TrafficEnvironment
    from server.graders import grade_easy, grade_medium, grade_hard, heuristic_policy, run_episode

TASK_LEVEL = os.environ.get("TASK_LEVEL", "easy")
env = TrafficEnvironment(task_level=TASK_LEVEL)

app = FastAPI(
    title="Smart City Traffic Flow -- OpenEnv",
    description=(
        "RL environment for adaptive traffic signal control. "
        "Agent controls signal phases to minimise vehicle wait time and "
        "maximise throughput across urban intersections."
    ),
    version="1.0.0",
)


# ── Standard OpenEnv endpoints ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check -- openenv validate calls this first."""
    return {"status": "healthy", "task_level": TASK_LEVEL}


@app.post("/reset", response_model=TrafficObservation)
def reset(seed: Optional[int] = None):
    """Start a fresh episode. Pass ?seed=42 for reproducibility."""
    try:
        return env.reset(seed=seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=TrafficObservation)
def step(action: TrafficAction):
    """
    Apply one action, advance simulation 5 seconds, return obs + reward.

    Body: {"action_type": "extend_green", "intersection_id": 0}
    """
    try:
        return env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=TrafficState)
def state():
    """Return episode metadata (step count, cumulative reward, etc.)"""
    try:
        return env.state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Grader endpoints (REQUIRED by hackathon validator) ─────────────────────

@app.get("/tasks")
def tasks():
    """
    List all 3 tasks with descriptions and scoring info.
    The validator checks this endpoint exists and returns at least 3 tasks.
    """
    return {
        "tasks": [
            {
                "id": "easy",
                "description": "Single intersection, 4 lanes, steady vehicle arrival rate",
                "difficulty": "easy",
                "max_steps": 100,
                "reward_range": [-1.0, 1.0],
                "scoring": "0.0-1.0 normalised against random and heuristic baselines",
                "env_vars": {"TASK_LEVEL": "easy"},
            },
            {
                "id": "medium",
                "description": "3-intersection corridor with rush-hour demand spike at step 50",
                "difficulty": "medium",
                "max_steps": 200,
                "reward_range": [-1.0, 1.0],
                "scoring": "0.0-1.0 normalised against random and heuristic baselines",
                "env_vars": {"TASK_LEVEL": "medium"},
            },
            {
                "id": "hard",
                "description": "3x3 grid of 9 intersections with random incidents and demand spikes",
                "difficulty": "hard",
                "max_steps": 300,
                "reward_range": [-1.0, 1.0],
                "scoring": "0.0-1.0 normalised against random and heuristic baselines",
                "env_vars": {"TASK_LEVEL": "hard"},
            },
        ]
    }


class GraderRequest(BaseModel):
    """Request body for /grader endpoint."""
    task_id: str           # "easy" | "medium" | "hard"
    n_seeds: int = 3       # number of seeds to average over (default 3 for speed)


@app.post("/grader")
def grader(request: GraderRequest):
    """
    Grade a task using the built-in heuristic policy.
    Returns a normalised score in [0.0, 1.0].

    The validator calls this endpoint to confirm graders exist for all 3 tasks.

    Body: {"task_id": "easy", "n_seeds": 3}
    """
    try:
        task_id = request.task_id.lower()
        n_seeds = max(1, min(request.n_seeds, 10))  # clamp between 1 and 10

        if task_id == "easy":
            score = grade_easy(policy=heuristic_policy, n_seeds=n_seeds)
        elif task_id == "medium":
            score = grade_medium(policy=heuristic_policy, n_seeds=n_seeds)
        elif task_id == "hard":
            score = grade_hard(policy=heuristic_policy, n_seeds=n_seeds)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task_id '{task_id}'. Must be easy, medium, or hard."
            )

        return {
            "task_id":  task_id,
            "score":    score,
            "min":      0.0,
            "max":      1.0,
            "n_seeds":  n_seeds,
            "policy":   "heuristic",
            "description": (
                "Score is normalised: 0.0 = random policy, 1.0 = heuristic oracle. "
                "Score > 0.5 means the agent is meaningfully better than random."
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/baseline")
def baseline():
    """
    Run heuristic oracle baseline on all 3 tasks and return scores.
    Provides reproducible reference scores for the submission.
    Uses 3 seeds per task for speed (full evaluation uses 5).
    """
    try:
        results = {}
        for task_id in ["easy", "medium", "hard"]:
            grader_fn = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}[task_id]
            score = grader_fn(policy=heuristic_policy, n_seeds=3)
            results[task_id] = round(score, 4)

        return {
            "policy":  "heuristic_oracle",
            "n_seeds": 3,
            "scores":  results,
            "reward_function": (
                "reward = 0.6 * throughput_bonus + 0.4 * wait_penalty -- dense, every step"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point (required by openenv validate) ─────────────────────────────

def main():
    """
    Server entry point.
    Called by: openenv serve, uv run server, python -m server.app
    """
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()