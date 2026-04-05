"""
server/graders.py — Agent graders for all 3 task levels.

Each grader runs multiple episodes with a given policy and returns
a normalised score between 0.0 and 1.0.

Score meaning:
  0.0 = performs the same as a purely random policy
  1.0 = matches the built-in heuristic oracle
  >1.0 = beats the heuristic (your LLM agent might do this!)

Scoring formula:
  score = clamp(
      (agent_metric - random_metric) / (oracle_metric - random_metric),
      0.0, 1.0
  )

Three metrics combined: throughput (60%) + wait time reduction (40%)
— same weights as the reward function.
"""

import random
from typing import Callable, Dict

#  Dual-import pattern ----------------------------
try:
    from .traffic_environment import TrafficEnvironment
    from ..models import TrafficAction, ActionType
except ImportError:
    from server.traffic_environment import TrafficEnvironment
    from models import TrafficAction, ActionType


# ----------------------------
# POLICIES
# A policy is just a function: (observation, n_intersections) → TrafficAction
# ----------------------------

def random_policy(obs, n_intersections: int) -> TrafficAction:
    """
    Baseline lower bound: picks a random action at every step.
    Score 0.0 is defined as matching this policy.
    """
    return TrafficAction(
        action_type=random.choice(list(ActionType)),
        intersection_id=random.randint(0, n_intersections - 1),
    )


def heuristic_policy(obs, n_intersections: int) -> TrafficAction:
    """
    Oracle upper bound: finds the most congested intersection and
    either extends its green (if heavily loaded) or switches phase.
    Score 1.0 is defined as matching this policy.
    """
    # Find intersection with longest total queue
    max_queue   = -1
    target_id   = 0
    for inter in obs.intersections:
        total_q = sum(l.queue_length for l in inter.lanes)
        if total_q > max_queue:
            max_queue = total_q
            target_id = inter.intersection_id

    # If 2+ lanes heavily loaded → extend green; otherwise switch phase
    target      = obs.intersections[target_id]
    heavy_lanes = sum(1 for l in target.lanes if l.queue_length > 5)
    action_type = ActionType.EXTEND_GREEN if heavy_lanes >= 2 else ActionType.NEXT_PHASE

    return TrafficAction(action_type=action_type, intersection_id=target_id)


# ----------------------------
# EPISODE RUNNER
# ----------------------------

def run_episode(task_level: str, policy: Callable, seed: int) -> Dict[str, float]:
    """
    Run one full episode and return performance metrics.
    seed is passed to reset() for full reproducibility.
    """
    env = TrafficEnvironment(task_level=task_level)
    obs = env.reset(seed=seed)
    n   = len(obs.intersections)

    total_throughput = 0
    total_wait_sum   = 0.0
    steps            = 0

    while not obs.done:
        action            = policy(obs, n)
        obs               = env.step(action)
        total_throughput += obs.throughput_last_step
        total_wait_sum   += obs.total_avg_wait
        steps            += 1

    return {
        "throughput":        total_throughput,
        "avg_wait":          total_wait_sum / max(steps, 1),
        "cumulative_reward": env.state.cumulative_reward,
    }


def _avg_over_seeds(task_level: str, policy: Callable, n_seeds: int) -> Dict[str, float]:
    """Average metrics over multiple seeds for stable scores."""
    results = [run_episode(task_level, policy, seed=i) for i in range(n_seeds)]
    return {k: sum(r[k] for r in results) / len(results) for k in results[0]}


# ----------------------------
# NORMALISED SCORING
# ----------------------------

def _normalise(agent_val: float, random_val: float, oracle_val: float) -> float:
    """Clamp score to [0, 1] relative to random=0 and oracle=1."""
    denom = oracle_val - random_val
    if abs(denom) < 1e-9:
        return 0.5
    raw = (agent_val - random_val) / denom
    return round(min(max(raw, 0.0), 1.0), 4)


def _score_results(agent: dict, rnd: dict, oracle: dict) -> float:
    """Combine throughput (60%) and wait-time reduction (40%) into final score."""
    tp_score   = _normalise(agent["throughput"], rnd["throughput"],  oracle["throughput"])
    # For wait time: lower is better, so we negate
    wait_score = _normalise(-agent["avg_wait"],  -rnd["avg_wait"],   -oracle["avg_wait"])
    return round(0.6 * tp_score + 0.4 * wait_score, 4)


# ----------------------------
# PUBLIC GRADERS  (called by baseline.py and the test suite)
# ----------------------------

def grade_easy(policy: Callable = None, n_seeds: int = 5) -> float:
    """
    Easy grader — single intersection.
    A score of 0.5+ means the agent is meaningfully better than random.
    """
    if policy is None:
        policy = heuristic_policy
    rnd    = _avg_over_seeds("easy", random_policy,    n_seeds)
    agent  = _avg_over_seeds("easy", policy,           n_seeds)
    oracle = _avg_over_seeds("easy", heuristic_policy, n_seeds)
    score  = _score_results(agent, rnd, oracle)
    print(f"[EASY]   throughput={agent['throughput']:.1f}  "
          f"avg_wait={agent['avg_wait']:.2f}s  score={score}")
    return score


def grade_medium(policy: Callable = None, n_seeds: int = 5) -> float:
    """
    Medium grader — 3-intersection corridor with rush-hour spike.
    A score < 0.4 means the agent cannot handle the surge.
    """
    if policy is None:
        policy = heuristic_policy
    rnd    = _avg_over_seeds("medium", random_policy,    n_seeds)
    agent  = _avg_over_seeds("medium", policy,           n_seeds)
    oracle = _avg_over_seeds("medium", heuristic_policy, n_seeds)
    score  = _score_results(agent, rnd, oracle)
    print(f"[MEDIUM] throughput={agent['throughput']:.1f}  "
          f"avg_wait={agent['avg_wait']:.2f}s  score={score}")
    return score


def grade_hard(policy: Callable = None, n_seeds: int = 5) -> float:
    """
    Hard grader — 9-intersection grid with random incidents.
    A score > 0.6 means the agent handles incidents gracefully.
    """
    if policy is None:
        policy = heuristic_policy
    rnd    = _avg_over_seeds("hard", random_policy,    n_seeds)
    agent  = _avg_over_seeds("hard", policy,           n_seeds)
    oracle = _avg_over_seeds("hard", heuristic_policy, n_seeds)
    score  = _score_results(agent, rnd, oracle)
    print(f"[HARD]   throughput={agent['throughput']:.1f}  "
          f"avg_wait={agent['avg_wait']:.2f}s  score={score}")
    return score


# Quick self-test ----------------------------
if __name__ == "__main__":
    print("Running all graders with heuristic policy...\n")
    e = grade_easy()
    m = grade_medium()
    h = grade_hard()
    print(f"\nFinal scores → easy={e}  medium={m}  hard={h}")
