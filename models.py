"""
models.py — Typed data contracts for the Traffic Flow environment.

Every piece of data the environment sends or receives is defined here
as a Pydantic model. This gives us:
  - Automatic validation (bad data raises an error immediately)
  - Auto-generated JSON schema (FastAPI uses this for /docs)
  - Clean type hints throughout the codebase

Three models you MUST have for OpenEnv spec compliance:
  TrafficAction      → what the agent sends IN  (input)
  TrafficObservation → what the environment sends OUT after each step
  TrafficState       → episode-level metadata returned by state()
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum


# ----------------------------
# ACTION SPACE
# What the agent can do. Kept deliberately small so the
# agent has fewer choices to learn — best practice for RL.
# ----------------------------

class ActionType(str, Enum):
    EXTEND_GREEN = "extend_green"  # Hold current green phase 5 extra seconds
    NEXT_PHASE   = "next_phase"    # Switch to next phase immediately


class TrafficAction(BaseModel):
    """
    One action the agent takes per step.

    action_type     : extend_green | next_phase
    intersection_id : which intersection to control (0-indexed)
                      easy=0 only, medium=0-2, hard=0-8
    """
    action_type: ActionType = Field(
        ...,
        description="extend_green = +5s on current green; next_phase = switch now",
    )
    intersection_id: int = Field(
        default=0,
        ge=0,
        description="Index of the intersection to control",
    )


# ----------------------------
# OBSERVATION SPACE
# What the agent sees. Mirrors real road-sensor output:
# queue lengths and wait times per lane — nothing more.
# ----------------------------

class LaneState(BaseModel):
    """Per-lane sensor reading (one road-camera / loop-detector)."""
    lane_id:       str
    queue_length:  int   = Field(..., ge=0,   description="Cars waiting at red")
    avg_wait_time: float = Field(..., ge=0.0, description="Average wait time (seconds)")
    flow_rate:     float = Field(..., ge=0.0, description="Max possible veh/min on green")


class IntersectionState(BaseModel):
    """All sensor readings for one intersection."""
    intersection_id: int
    current_phase:   int   = Field(..., description="Active signal phase (0–3)")
    phase_elapsed:   float = Field(..., description="Seconds spent in this phase so far")
    lanes:           List[LaneState]


class TrafficObservation(BaseModel):
    """
    Returned by reset() and step().
    Everything the agent needs to make its next decision.
    """
    intersections:          List[IntersectionState]
    total_waiting_vehicles: int
    total_avg_wait:         float = Field(..., description="Fleet-wide average wait (seconds)")
    throughput_last_step:   int   = Field(..., description="Vehicles cleared this step")
    reward:                 float = Field(..., description="Step reward signal")
    done:                   bool  = False
    info:                   Dict[str, float] = Field(default_factory=dict)


# ----------------------------
# EPISODE STATE
# Returned by state(). NOT the same as observation —
# this is bookkeeping metadata for the training framework.
# ----------------------------

class TrafficState(BaseModel):
    """Episode-level metadata returned by state()."""
    episode_id:              Optional[str] = None
    step_count:              int   = 0
    task_level:              str   = Field("easy", description="easy | medium | hard")
    cumulative_reward:       float = 0.0
    cumulative_throughput:   int   = 0
    cumulative_wait:         float = 0.0
