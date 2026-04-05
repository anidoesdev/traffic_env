"""
inference.py -- Traffic Flow OpenEnv Inference Script
======================================================

MANDATORY environment variables:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use.
    HF_TOKEN            Your Hugging Face / API key.
    LOCAL_IMAGE_NAME    Docker image name (if using from_docker_image).
    TASK_LEVEL          Which task to run: easy | medium | hard (default: easy)

Defaults:
    API_BASE_URL = "https://router.huggingface.co/v1"
    MODEL_NAME   = "Qwen/Qwen2.5-72B-Instruct"
    TASK_LEVEL   = "easy"

STDOUT FORMAT (exactly as required by the hackathon):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Usage:
    # Against a running local server
    python inference.py

    # Against a HF Space
    set ENV_BASE_URL=https://YOUR-USERNAME-traffic-env.hf.space
    python inference.py

    # Against a local Docker container
    set LOCAL_IMAGE_NAME=traffic-env
    python inference.py
"""

import asyncio
import json
import os
import random
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables  (read at startup, never hardcoded)
# ---------------------------------------------------------------------------
API_KEY        = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME     = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME     = os.getenv("LOCAL_IMAGE_NAME")   # Docker image, if using from_docker_image
ENV_BASE_URL   = os.getenv("ENV_BASE_URL")        # Direct URL to a running server

TASK_NAME      = os.getenv("TASK_LEVEL", "easy")  # easy | medium | hard
BENCHMARK      = "traffic-env"

# Task-level step limits (matches TASK_CONFIG in traffic_environment.py)
MAX_STEPS_MAP  = {"easy": 100, "medium": 200, "hard": 300}
MAX_STEPS      = MAX_STEPS_MAP.get(TASK_NAME, 100)

# An episode is considered successful if normalised cumulative reward > this
SUCCESS_SCORE_THRESHOLD = 0.3

# LLM call settings
TEMPERATURE  = 0.0    # deterministic — same obs always gives same action
MAX_TOKENS   = 80     # we only need a small JSON object back

# ---------------------------------------------------------------------------
# System prompt — tells the LLM exactly what it is and how to respond
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an adaptive traffic signal controller agent.

    At each step you receive sensor data from traffic intersections and must
    choose one action to keep traffic flowing smoothly.

    ALWAYS respond with ONLY a valid JSON object. No explanation, no markdown,
    no code blocks. Just raw JSON on a single line.

    Response format:
    {"action_type": "extend_green", "intersection_id": 0}

    Rules:
    - action_type must be exactly "extend_green" OR "next_phase"
    - intersection_id must be an integer (0-indexed)
    - extend_green = hold current green phase 5 extra seconds
    - next_phase   = switch to next signal phase immediately

    Strategy:
    - If a lane has queue_length > 8, it is critically congested
    - Use extend_green when the current green lanes are heavily loaded
    - Use next_phase when a different direction has higher queue lengths
    - Target the intersection with the most total waiting vehicles
""").strip()


# ---------------------------------------------------------------------------
# Logging helpers  (exact format required by the hackathon spec)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Observation → prompt converter
# Turns the observation dict into plain text the LLM can reason about
# ---------------------------------------------------------------------------

def obs_to_prompt(step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    lines = []

    for inter in obs.get("intersections", []):
        lane_parts = []
        for lane in inter.get("lanes", []):
            lane_parts.append(
                f"{lane['lane_id']}: queue={lane['queue_length']} "
                f"wait={lane['avg_wait_time']:.1f}s"
            )
        lanes_str = " | ".join(lane_parts)
        lines.append(
            f"Intersection {inter['intersection_id']} "
            f"[phase={inter['current_phase']}, elapsed={inter['phase_elapsed']:.0f}s]: "
            f"{lanes_str}"
        )

    lines.append("")
    lines.append(f"Total vehicles waiting : {obs.get('total_waiting_vehicles', 0)}")
    lines.append(f"Fleet avg wait time    : {obs.get('total_avg_wait', 0):.1f}s")
    lines.append(f"Vehicles cleared (last): {obs.get('throughput_last_step', 0)}")
    lines.append(f"Last step reward       : {last_reward:.3f}")
    lines.append(f"Current step           : {step}")

    if history:
        lines.append("")
        lines.append("Recent actions:")
        for h in history[-4:]:
            lines.append(f"  {h}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM policy — calls the model and parses the JSON action
# Falls back to a heuristic if the model returns unparseable output
# ---------------------------------------------------------------------------

def get_llm_action(
    client: OpenAI,
    step: int,
    obs: dict,
    last_reward: float,
    history: List[str],
    n_intersections: int,
) -> tuple[dict, Optional[str]]:
    """
    Returns (action_dict, error_string_or_None).
    action_dict always has keys: action_type, intersection_id.
    """
    prompt = obs_to_prompt(step, obs, last_reward, history)
    error  = None

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if model adds them despite instructions
        raw = raw.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw)

        # Validate action_type is one of the two allowed values
        if data.get("action_type") not in ("extend_green", "next_phase"):
            raise ValueError(f"Invalid action_type: {data.get('action_type')}")

        action = {
            "action_type":      data["action_type"],
            "intersection_id":  int(data.get("intersection_id", 0)),
        }
        return action, None

    except Exception as exc:
        error = str(exc)
        # Fallback: heuristic — find most congested intersection, extend or switch
        action = _heuristic_fallback(obs, n_intersections)
        return action, error


def _heuristic_fallback(obs: dict, n_intersections: int) -> dict:
    """Simple heuristic used when the LLM fails to respond correctly."""
    max_queue  = -1
    target_id  = 0
    for inter in obs.get("intersections", []):
        total_q = sum(l.get("queue_length", 0) for l in inter.get("lanes", []))
        if total_q > max_queue:
            max_queue = total_q
            target_id = inter["intersection_id"]

    # Count heavily loaded lanes at that intersection
    target     = next(
        (i for i in obs.get("intersections", []) if i["intersection_id"] == target_id),
        None,
    )
    heavy = 0
    if target:
        heavy = sum(1 for l in target.get("lanes", []) if l.get("queue_length", 0) > 5)

    action_type = "extend_green" if heavy >= 2 else "next_phase"
    return {"action_type": action_type, "intersection_id": target_id}


# ---------------------------------------------------------------------------
# Score normalisation
# Maps cumulative reward to a [0, 1] score using empirical baselines
# ---------------------------------------------------------------------------

# (random_baseline, oracle_baseline) measured over 10 seeds each
_BASELINES = {
    "easy":   (-12.0,  68.0),
    "medium": (-36.0, 116.0),
    "hard":   (-66.0, 138.0),
}

def normalise_score(task: str, cumulative_reward: float) -> float:
    lo, hi = _BASELINES.get(task, (-50.0, 100.0))
    raw = (cumulative_reward - lo) / (hi - lo + 1e-9)
    return round(min(max(raw, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------

async def main() -> None:
    # -- OpenAI client (used for all LLM calls) ------------------------------
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # -- Connect to the environment ------------------------------------------
    # Priority: LOCAL_IMAGE_NAME > ENV_BASE_URL > default localhost
    #
    # The sample template uses openenv-core's async client.
    # We use our own HTTP client (client.py) via requests since it works with
    # both local servers and HF Spaces without requiring the openenv SDK.
    #
    # To use the openenv-core async client instead, replace this block with:
    #   from traffic_env import TrafficEnvClient as OfficialClient
    #   env = await OfficialClient.from_docker_image(IMAGE_NAME)

    import requests as _requests

    if IMAGE_NAME:
        # Start Docker container and wait for it to be healthy
        import subprocess, time
        port = 7860
        container_name = "traffic-env-inference"
        subprocess.Popen([
            "docker", "run", "--rm", "-d",
            "--name", container_name,
            "-p", f"{port}:{port}",
            "-e", f"TASK_LEVEL={TASK_NAME}",
            IMAGE_NAME,
        ])
        base_url = f"http://localhost:{port}"
        # Wait for server to be ready (up to 30 seconds)
        for _ in range(30):
            try:
                r = _requests.get(f"{base_url}/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
    elif ENV_BASE_URL:
        base_url = ENV_BASE_URL.rstrip("/")
    else:
        base_url = "http://localhost:7860"

    # -- Episode state -------------------------------------------------------
    history: List[str] = []
    rewards: List[float] = []
    steps_taken  = 0
    cumulative_r = 0.0
    score        = 0.0
    success      = False
    n_intersections = 1   # updated after reset

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # reset() — start a fresh episode
        resp = _requests.post(f"{base_url}/reset", params={"seed": 42}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
        last_reward = 0.0
        n_intersections = len(obs.get("intersections", [{"intersection_id": 0}]))

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            # Ask the LLM what action to take
            action, error = get_llm_action(
                client, step, obs, last_reward, history, n_intersections
            )

            # Format action as a compact string for the [STEP] log line
            action_str = f"{action['action_type']}(id={action['intersection_id']})"

            # step() — apply action, get next observation
            resp = _requests.post(
                f"{base_url}/step",
                json=action,
                timeout=30,
            )
            resp.raise_for_status()
            obs = resp.json()

            reward       = float(obs.get("reward", 0.0))
            done         = bool(obs.get("done", False))
            last_reward  = reward
            cumulative_r += reward
            rewards.append(reward)
            steps_taken  = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"step {step}: {action_str} -> reward {reward:+.3f} "
                f"(waiting={obs.get('total_waiting_vehicles', 0)}, "
                f"throughput={obs.get('throughput_last_step', 0)})"
            )

            if done:
                break

        score   = normalise_score(TASK_NAME, cumulative_r)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        # Always emit [END] even on crash
        print(f"[DEBUG] Fatal error: {exc}", flush=True)

    finally:
        # Stop Docker container if we started one
        if IMAGE_NAME:
            try:
                import subprocess
                subprocess.run(
                    ["docker", "stop", "traffic-env-inference"],
                    capture_output=True,
                )
            except Exception as e:
                print(f"[DEBUG] docker stop error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())