"""
server/traffic_environment.py — Core simulation engine.

This file contains the actual "world" — the physics of the simulation.
It is completely independent of HTTP/FastAPI. You can import and use it
directly in Python without running any server.

KEY FIX vs original: every episode now uses an ISOLATED random.Random
instance seeded from reset(seed=N). This means the same seed ALWAYS
produces the exact same episode, no matter what else is running.
That is what "reproducible scores" means in the requirements.

Three task levels:
  easy   — 1 intersection,  4 lanes, steady arrivals,    100 steps
  medium — 3 intersections, 4 lanes, rush-hour spike,    200 steps
  hard   — 9 intersections, 4 lanes, incidents + spikes, 300 steps
"""

import uuid
import random as _random_module   # alias so we don't shadow the local rng variable
import math
from typing import List, Optional, Tuple

# -- Dual-import pattern (required by OpenEnv) ----------
# Relative imports work when running inside the repo (PYTHONPATH=src:envs)
# Bare imports work inside Docker (PYTHONPATH=/app)
# Both paths must be tried or one environment will always fail.
try:
    from ..models import (
        TrafficAction, TrafficObservation, TrafficState,
        LaneState, IntersectionState, ActionType,
    )
except ImportError:
    from models import (
        TrafficAction, TrafficObservation, TrafficState,
        LaneState, IntersectionState, ActionType,
    )


# SIMULATION CONSTANTS
# Change these to make the world harder or easier.

PHASE_COUNT        = 4     # How many signal phases exist (N green, N yellow, S green, S yellow)
DEFAULT_PHASE_TIME = 30.0  # Seconds before auto-advancing to next phase (static baseline)
EXTEND_SECONDS     = 5.0   # How many seconds extend_green adds
STEP_DURATION      = 5.0   # Each env step = 5 simulated seconds
MAX_QUEUE          = 50    # Maximum cars that can queue in one lane
ARRIVAL_BASE       = 2.0   # Mean cars arriving per lane per step (Poisson mean)
DISCHARGE_RATE     = 3.5   # Max cars that clear a green lane per step



# LANE SIMULATION
# Models one lane of traffic (e.g. "North lane at intersection 0")

class LaneSim:
    """
    Internal mutable state for one lane.
    The agent never sees this object directly — it only sees LaneState
    (the Pydantic model built from this in to_model()).
    """

    def __init__(self, lane_id: str, arrival_mean: float, rng: _random_module.Random):
        self.lane_id      = lane_id
        self.arrival_mean = arrival_mean
        self._rng         = rng          # ISOLATED rng — not the global random module
        # Start with a small random queue so the episode isn't trivially easy at step 0
        self.queue:      int   = self._rng.randint(0, 5)
        self.total_wait: float = 0.0     # Cumulative vehicle-seconds of waiting

    def arrive(self, multiplier: float = 1.0) -> int:
        """
        Simulate cars arriving this step (Poisson-distributed).
        multiplier > 1.0 = rush hour or incident — more cars than usual.
        """
        mean = max(self.arrival_mean * multiplier, 0.1)
        # expovariate(1/mean) gives exponential distribution; rounding gives Poisson approx
        n = int(self._rng.expovariate(1.0 / mean) + 0.5)
        n = max(0, min(n, MAX_QUEUE - self.queue))
        self.queue = min(self.queue + n, MAX_QUEUE)
        return n

    def discharge(self, is_green: bool) -> int:
        """Cars that cross the stop-line this step. Only happens on green."""
        if not is_green or self.queue == 0:
            return 0
        discharged = min(self.queue, int(DISCHARGE_RATE))
        self.queue -= discharged
        return discharged

    def accumulate_wait(self):
        """Every step, every waiting car adds STEP_DURATION seconds to total wait."""
        self.total_wait += self.queue * STEP_DURATION

    @property
    def avg_wait(self) -> float:
        """Average wait time per car currently in the queue."""
        return round(self.total_wait / max(self.queue, 1), 2)

    @property
    def flow_rate(self) -> float:
        """Theoretical max vehicles per minute if this lane were always green."""
        return round(DISCHARGE_RATE * (60.0 / STEP_DURATION), 2)



# INTERSECTION SIMULATION
# Models one full intersection (4 lanes + signal controller)


class IntersectionSim:
    """
    Internal mutable state for one intersection.
    Contains 4 LaneSim objects and manages the signal phase.
    """

    # Which lane indices are green in each phase
    # Phase 0 = N+S green, Phase 1 = N only (yellow transition)
    # Phase 2 = E+W green, Phase 3 = E only (yellow transition)
    PHASE_MAP = {0: [0, 1], 1: [0], 2: [2, 3], 3: [2]}

    def __init__(self, intersection_id: int, n_lanes: int,
                 arrival_mean: float, rng: _random_module.Random):
        self.intersection_id = intersection_id
        self.current_phase   = 0
        self.phase_elapsed   = 0.0
        self._rng            = rng
        self.lanes: List[LaneSim] = [
            LaneSim(f"I{intersection_id}_L{i}", arrival_mean, rng)
            for i in range(n_lanes)
        ]

    def _green_lanes(self) -> List[int]:
        return self.PHASE_MAP.get(self.current_phase % PHASE_COUNT, [0])

    def apply_action(self, action: TrafficAction):
        """Apply agent's chosen action to this intersection's signal."""
        if action.action_type == ActionType.EXTEND_GREEN:
            # Push back the auto-advance clock by EXTEND_SECONDS
            self.phase_elapsed = max(0.0, self.phase_elapsed - EXTEND_SECONDS)
        elif action.action_type == ActionType.NEXT_PHASE:
            # Force immediate phase switch
            self.current_phase = (self.current_phase + 1) % PHASE_COUNT
            self.phase_elapsed = 0.0

    def step_sim(self, arrival_multiplier: float = 1.0) -> Tuple[int, float]:
        """
        Advance the simulation by one 5-second step.
        Returns (throughput, total_wait_added) for reward calculation.
        """
        green = self._green_lanes()
        throughput = 0

        for i, lane in enumerate(self.lanes):
            lane.arrive(arrival_multiplier)          # new cars arrive
            if i in green:
                throughput += lane.discharge(True)   # green lanes discharge
            lane.accumulate_wait()                   # all waiting cars accumulate wait

        # Advance phase clock; auto-switch if time exceeded
        self.phase_elapsed += STEP_DURATION
        if self.phase_elapsed >= DEFAULT_PHASE_TIME:
            self.current_phase = (self.current_phase + 1) % PHASE_COUNT
            self.phase_elapsed = 0.0

        total_wait = sum(l.queue * STEP_DURATION for l in self.lanes)
        return throughput, total_wait

    def to_model(self) -> IntersectionState:
        """Convert internal state → Pydantic model for the agent to observe."""
        return IntersectionState(
            intersection_id=self.intersection_id,
            current_phase=self.current_phase,
            phase_elapsed=round(self.phase_elapsed, 1),
            lanes=[
                LaneState(
                    lane_id=l.lane_id,
                    queue_length=l.queue,
                    avg_wait_time=l.avg_wait,
                    flow_rate=l.flow_rate,
                )
                for l in self.lanes
            ],
        )



# TRAFFIC ENVIRONMENT  (the OpenEnv interface)
# This is the class that implements reset() / step() / state()

class TrafficEnvironment:
    """
    OpenEnv-compatible environment for adaptive traffic signal control.

    Usage:
        env = TrafficEnvironment(task_level="easy")
        obs = env.reset(seed=42)
        while not obs.done:
            action = agent.decide(obs)
            obs = env.step(action)
        print(env.state.cumulative_reward)
    """

    TASK_CONFIG = {
        "easy":   {"n_intersections": 1, "n_lanes": 4, "max_steps": 100,
                   "arrival_mean": ARRIVAL_BASE},
        "medium": {"n_intersections": 3, "n_lanes": 4, "max_steps": 200,
                   "arrival_mean": ARRIVAL_BASE * 1.5},
        "hard":   {"n_intersections": 9, "n_lanes": 4, "max_steps": 300,
                   "arrival_mean": ARRIVAL_BASE * 2.0},
    }

    def __init__(self, task_level: str = "easy"):
        assert task_level in self.TASK_CONFIG, \
            f"task_level must be one of {list(self.TASK_CONFIG)}"
        self.task_level          = task_level
        self._cfg                = self.TASK_CONFIG[task_level]
        self._intersections:     List[IntersectionSim] = []
        self._state              = TrafficState(task_level=task_level)
        self._step               = 0
        self._cumulative_reward  = 0.0
        self._cumulative_throughput = 0
        self._cumulative_wait    = 0.0
        self._rng                = _random_module.Random()  # will be re-seeded in reset()

    # -- OpenEnv required method 1 -------

    def reset(self, seed: Optional[int] = None) -> TrafficObservation:
        """
        Start a fresh episode.
        seed=N guarantees identical episode every time — required for reproducibility.
        """
        # Create a NEW isolated RNG for this episode
        # This means global random state is never touched → fully reproducible
        self._rng = _random_module.Random(seed)

        cfg = self._cfg
        self._intersections = [
            IntersectionSim(i, cfg["n_lanes"], cfg["arrival_mean"], self._rng)
            for i in range(cfg["n_intersections"])
        ]
        self._step                  = 0
        self._cumulative_reward     = 0.0
        self._cumulative_throughput = 0
        self._cumulative_wait       = 0.0
        self._state = TrafficState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_level=self.task_level,
        )
        return self._build_obs(total_throughput=0, total_wait=0.0, reward=0.0, done=False)

    #  OpenEnv required method --------

    def step(self, action: TrafficAction) -> TrafficObservation:
        """
        Apply action, advance simulation by 5 seconds, return new observation + reward.
        """
        self._step += 1
        arrival_mult = self._arrival_multiplier()

        # Apply action to the targeted intersection
        if 0 <= action.intersection_id < len(self._intersections):
            self._intersections[action.intersection_id].apply_action(action)

        # Advance ALL intersections (even uncontrolled ones keep ticking)
        total_throughput = 0
        total_wait       = 0.0
        for inter in self._intersections:
            tp, wt = inter.step_sim(arrival_mult)
            total_throughput += tp
            total_wait       += wt

        reward = self._compute_reward(total_throughput, total_wait)
        self._cumulative_reward     += reward
        self._cumulative_throughput += total_throughput
        self._cumulative_wait       += total_wait

        done = (self._step >= self._cfg["max_steps"])

        # Update episode metadata
        self._state = TrafficState(
            episode_id=self._state.episode_id,
            step_count=self._step,
            task_level=self.task_level,
            cumulative_reward=round(self._cumulative_reward, 4),
            cumulative_throughput=self._cumulative_throughput,
            cumulative_wait=round(self._cumulative_wait, 2),
        )
        return self._build_obs(total_throughput, total_wait, reward, done)

    # ---- OpenEnv required method 3 --------

    @property
    def state(self) -> TrafficState:
        """Episode-level metadata (step count, cumulative reward, etc.)"""
        return self._state

    # ---- Internal helpers -----------

    def _arrival_multiplier(self) -> float:
        """
        Rush-hour spike: arrival rate peaks mid-episode for medium/hard.
        Hard additionally has random incidents (5% chance per step).
        """
        if self.task_level == "easy":
            return 1.0
        peak  = 50 if self.task_level == "medium" else 80
        spike = 1.0 + 1.5 * math.exp(-((self._step - peak) ** 2) / (2 * 20 ** 2))
        if self.task_level == "hard" and self._rng.random() < 0.05:
            spike += self._rng.uniform(0.5, 2.0)
        return round(spike, 3)

    def _compute_reward(self, throughput: int, total_wait: float) -> float:
        """
        Dense reward fired every single step (not just at the end).

        Formula:
            reward = 0.6 × throughput_bonus + 0.4 × wait_penalty

        throughput_bonus ∈ [0, 1]  — reward clearing cars
        wait_penalty     ∈ [-1, 0] — penalise idle waiting

        Both terms are normalised by the theoretical maximum so the
        reward stays in roughly [-1, 1] regardless of task size.
        """
        n = len(self._intersections) * self._cfg["n_lanes"]
        throughput_bonus = throughput / (DISCHARGE_RATE * n + 1e-9)
        wait_penalty     = -total_wait / (MAX_QUEUE * STEP_DURATION * n + 1e-9)
        return round(0.6 * throughput_bonus + 0.4 * wait_penalty, 6)

    def _build_obs(self, total_throughput: int, total_wait: float,
                   reward: float, done: bool) -> TrafficObservation:
        """Assemble the Pydantic observation object from current simulation state."""
        total_waiting = sum(l.queue for inter in self._intersections for l in inter.lanes)
        avg_wait = total_wait / max(total_waiting * STEP_DURATION, 1e-9)
        return TrafficObservation(
            intersections=[i.to_model() for i in self._intersections],
            total_waiting_vehicles=total_waiting,
            total_avg_wait=round(avg_wait, 2),
            throughput_last_step=total_throughput,
            reward=reward,
            done=done,
            info={
                "step":               float(self._step),
                "arrival_multiplier": self._arrival_multiplier(),
                "cumulative_reward":  round(self._cumulative_reward, 4),
            },
        )
