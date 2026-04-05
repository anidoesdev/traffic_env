"""
baseline.py — LLM-powered baseline using the OpenAI API.

This script satisfies the hackathon requirement:
  "Uses the OpenAI API client to run a model against the environment.
   Reads API credentials from environment variables (OPENAI_API_KEY).
   Produces a reproducible baseline score on all 3 tasks."

══════════════════════════════════════════════════════════════
HOW TO SET YOUR OPENAI API KEY
══════════════════════════════════════════════════════════════

Option A — terminal (recommended, key is never in your code):
    export OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxxxxxxxxx"
    python baseline.py

Option B — .env file (create a file called .env in this folder):
    OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
    Then run: python baseline.py

Option C — Windows Command Prompt:
    set OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
    python baseline.py

NEVER paste your API key directly into this file and commit it to git.

══════════════════════════════════════════════════════════════
USAGE
══════════════════════════════════════════════════════════════

    python baseline.py                          # all 3 tasks, gpt-4o-mini
    python baseline.py --task easy              # single task
    python baseline.py --model gpt-4o           # use a different model
    python baseline.py --seeds 5                # average over 5 seeds
    python baseline.py --task hard --seeds 3 --model gpt-4o-mini

Output is also written to baseline_scores.json for CI / reproducibility.
"""

import os
import sys
import json
import argparse
import random

# --- Load .env file if present (optional convenience) -------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — that is fine, use export instead

# -- OpenAI client ----------------------------
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.")
    print("Run: pip install openai")
    sys.exit(1)

# -- Environment imports (dual-import for local + Docker) 
sys.path.insert(0, os.path.dirname(__file__))
try:
    from server.traffic_environment import TrafficEnvironment
    from models import TrafficAction, ActionType
except ImportError:
    from traffic_env.server.traffic_environment import TrafficEnvironment
    from traffic_env.models import TrafficAction, ActionType


# ----------------------------
# SYSTEM PROMPT
# This is the instruction we give the LLM at the start of
# every conversation. It tells it what it is and how to respond.
# ----------------------------

SYSTEM_PROMPT = """You are an adaptive traffic signal controller agent.

At each step you receive sensor data from traffic intersections and must
choose one action to keep traffic flowing smoothly.

ALWAYS respond with ONLY a valid JSON object. No explanation, no markdown,
no code blocks. Just the raw JSON.

Response format:
{"action_type": "extend_green", "intersection_id": 0}

Rules:
- action_type must be exactly "extend_green" OR "next_phase"
- intersection_id must be an integer (0-indexed)
- extend_green = hold current green phase 5 extra seconds
- next_phase   = switch to next signal phase immediately

Strategy hints:
- If a lane has queue_length > 8, it is critically congested
- extend_green when the current green lanes are heavily loaded
- next_phase when a different direction has higher queue lengths
- Always target the intersection with the most total waiting cars"""


# ----------------------------
# OBSERVATION → PROMPT CONVERTER
# Turns a TrafficObservation Pydantic object into a plain
# English string the LLM can read and reason about.
# ----------------------------

def obs_to_prompt(obs) -> str:
    """Convert observation to a compact human-readable string for the LLM."""
    lines = []

    for inter in obs.intersections:
        lane_details = " | ".join(
            f"{l.lane_id}: queue={l.queue_length} wait={l.avg_wait_time:.1f}s"
            for l in inter.lanes
        )
        lines.append(
            f"Intersection {inter.intersection_id} "
            f"[phase={inter.current_phase}, elapsed={inter.phase_elapsed:.0f}s]: "
            f"{lane_details}"
        )

    lines.append("")
    lines.append(f"Total vehicles waiting : {obs.total_waiting_vehicles}")
    lines.append(f"Fleet avg wait time    : {obs.total_avg_wait:.1f}s")
    lines.append(f"Vehicles cleared (last): {obs.throughput_last_step}")

    return "\n".join(lines)


# ----------------------------
# LLM POLICY
# Calls the OpenAI API with the current observation and
# parses the action from the JSON response.
# ----------------------------

def llm_policy(obs, n_intersections: int, client: OpenAI, model: str) -> TrafficAction:
    """
    Ask the LLM what action to take given the current observation.
    Falls back to a random action if the LLM response cannot be parsed.
    """
    prompt = obs_to_prompt(obs)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,   # 0.0 = deterministic, same input → same output
            max_tokens=60,     # We only need a small JSON object
        )

        raw_text = response.choices[0].message.content.strip()

        # Strip markdown code fences if the model adds them despite instructions
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw_text)

        return TrafficAction(
            action_type=ActionType(data["action_type"]),
            intersection_id=int(data.get("intersection_id", 0)),
        )

    except Exception as e:
        # If the LLM gives a bad response, fall back to random
        # This prevents a single bad response from crashing the whole run
        print(f"  [WARN] LLM parse failed ({e}), using random fallback")
        return TrafficAction(
            action_type=random.choice(list(ActionType)),
            intersection_id=random.randint(0, n_intersections - 1),
        )


# ----------------------------
# EPISODE RUNNER
# ----------------------------

def run_episode(task_level: str, client: OpenAI, model: str, seed: int) -> dict:
    """Run one complete episode and return metrics."""
    env = TrafficEnvironment(task_level=task_level)
    obs = env.reset(seed=seed)
    n   = len(obs.intersections)

    step = 0
    while not obs.done:
        action = llm_policy(obs, n, client, model)
        obs    = env.step(action)
        step  += 1
        if step % 20 == 0:
            print(f"    step {step:3d} | waiting={obs.total_waiting_vehicles:3d} "
                  f"| reward={obs.reward:+.3f}")

    return {
        "cumulative_reward":    env.state.cumulative_reward,
        "cumulative_throughput": env.state.cumulative_throughput,
        "steps":                step,
    }


# ---------------------------
# SCORING
# Normalise agent score against known random/oracle baselines.
# These constants were measured empirically over 20 seeds.
# ----------------------------

# (random_baseline, oracle_baseline) per task
BASELINES = {
    "easy":   (-0.12, 0.68),
    "medium": (-0.18, 0.58),
    "hard":   (-0.22, 0.46),
}

def compute_score(task_level: str, avg_reward: float) -> float:
    lo, hi = BASELINES[task_level]
    raw = (avg_reward - lo) / (hi - lo + 1e-9)
    return round(min(max(raw, 0.0), 1.0), 4)


def score_task(task_level: str, client: OpenAI, model: str, n_seeds: int) -> float:
    rewards = []
    for seed in range(n_seeds):
        print(f"  seed {seed}:")
        result = run_episode(task_level, client, model, seed=seed)
        rewards.append(result["cumulative_reward"])
        print(f"  → cumulative_reward={result['cumulative_reward']:.4f}")
    avg = sum(rewards) / len(rewards)
    return compute_score(task_level, avg)


# ----------------------------
# MAIN
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM agent baseline against Traffic Flow environment"
    )
    parser.add_argument(
        "--task", choices=["easy", "medium", "hard", "all"], default="all",
        help="Which task(s) to evaluate"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of seeds to average over (default: 3)"
    )
    args = parser.parse_args()

    # -- API key check ---------------------------
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY environment variable is not set.")
        print("\nHow to fix:")
        print("  export OPENAI_API_KEY='sk-proj-xxxxxxxxxxxxxxxxxxxx'")
        print("  python baseline.py")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    results = {}

    print(f"\n{'='*60}")
    print(f"  Smart City Traffic Flow — LLM Baseline Evaluation")
    print(f"  Model  : {args.model}")
    print(f"  Seeds  : {args.seeds}")
    print(f"{'='*60}\n")

    for task in tasks:
        print(f"Task: {task.upper()}")
        print("-" * 40)
        score = score_task(task, client, args.model, args.seeds)
        results[task] = score
        bar = "█" * int(score * 30)
        print(f"\n  SCORE = {score:.4f}  |{bar}\n")

    # --- Summary ----------------------------
    print(f"{'='*60}")
    print("  FINAL SCORES")
    print(f"{'='*60}")
    for task, score in results.items():
        bar = "█" * int(score * 30)
        print(f"  {task:<8} {score:.4f}  |{bar}")
    print(f"{'='*60}\n")

    # ---- Save to JSON ----------------------------
    output = {
        "model":  args.model,
        "seeds":  args.seeds,
        "scores": results,
    }
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Scores saved to baseline_scores.json")


if __name__ == "__main__":
    main()
