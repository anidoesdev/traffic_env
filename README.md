---
title: Smart City Traffic Flow
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---


#  Smart City Traffic Flow — OpenEnv RL Environment

> An OpenEnv-compatible Reinforcement Learning environment for **adaptive traffic signal control**.
> The agent learns to dynamically control signal phases at urban intersections to minimise vehicle waiting time and maximise throughput — simulating the kind of real-world system deployed by cities worldwide.

---

## Real-World Problem

Traditional traffic lights run on **fixed timers**. When traffic patterns shift — after a concert, during a storm, or in a morning rush — static signals cause gridlock. This environment trains an RL agent to act as an adaptive controller using data that mirrors real road sensors (queue lengths, wait times, throughput counts).

---

## Environment Design

### Action Space

| Field | Type | Values |
|---|---|---|
| `action_type` | enum | `extend_green` (add 5 s to current green phase), `next_phase` (switch immediately) |
| `intersection_id` | int | 0 … N-1 (depends on task) |

### Observation Space

| Field | Type | Description |
|---|---|---|
| `intersections` | list | Per-intersection state: phase, phase_elapsed, per-lane queue & wait |
| `total_waiting_vehicles` | int | Fleet-wide queue count |
| `total_avg_wait` | float (s) | Fleet-wide average wait time |
| `throughput_last_step` | int | Vehicles cleared in this step |
| `reward` | float | Step reward |
| `done` | bool | Episode termination flag |

### Reward Function (Dense — partial progress at every step)

```
reward = 0.6 × throughput_bonus + 0.4 × wait_penalty

throughput_bonus = cleared_vehicles / (max_discharge_rate × n_lanes)  ∈ [0, 1]
wait_penalty     = -total_wait / (max_possible_wait)                   ∈ [-1, 0]
```

The agent receives a signal **every step**, not just at episode end, making it suitable for standard policy-gradient and Q-learning algorithms.

---

##  Three Task Levels

| Task | Intersections | Demand | Max Steps | Challenge |
|---|---|---|---|---|
| **Easy** | 1 | Steady (2 veh/lane/step) | 100 | Learn basic phase timing |
| **Medium** | 3 (corridor) | Rush-hour spike at step 50 | 200 | Coordinate across a corridor; handle surges |
| **Hard** | 9 (3×3 grid) | Heavy + random incidents | 300 | Network-wide coordination under uncertainty |

### Agent Grader Scores (0.0 → 1.0)

- **0.0** = equivalent to a purely random policy  
- **1.0** = matches the built-in heuristic oracle  
- Scores above **0.6** on all three tasks indicate a strong adaptive strategy

---

##  Quick Start

### 1. Install dependencies

```bash
pip install fastapi uvicorn pydantic
```

### 2. Run the server locally

```bash
# Easy task (default)
uvicorn traffic_env.server.app:app --reload --port 8000

# Medium task
TASK_LEVEL=medium uvicorn traffic_env.server.app:app --reload --port 8000

# Hard task
TASK_LEVEL=hard uvicorn traffic_env.server.app:app --reload --port 8000
```

### 3. Interact via HTTP

```python
import requests

# Reset
obs = requests.post("http://localhost:8000/reset").json()

# Step
action = {"action_type": "extend_green", "intersection_id": 0}
obs = requests.post("http://localhost:8000/step", json=action).json()
print(obs["reward"], obs["done"])

# State
state = requests.get("http://localhost:8000/state").json()
```

### 4. Run baseline evaluation

```bash
python baseline.py                    # all tasks, heuristic policy
python baseline.py --task easy        # single task
python baseline.py --policy random    # random baseline
```

---

##  Docker

```bash
# Build
docker build -t traffic-env .

# Run easy task
docker run -p 7860:7860 -e TASK_LEVEL=easy traffic-env

# Run hard task
docker run -p 7860:7860 -e TASK_LEVEL=hard traffic-env
```

---

##  Hugging Face Spaces Deployment

```bash
pip install openenv-core
openenv push --repo-id your-username/traffic-env
```

The Space will be live at `https://anidoesdev-traffic-env.hf.space`

API docs: `https://anidoesdev-traffic-env.hf.space/docs`

---

##  Project Structure

```
traffic_env/
├── __init__.py              # Public exports
├── models.py                # Pydantic Action / Observation / State models
├── openenv.yaml             # OpenEnv spec manifest
├── requirements.txt
├── Dockerfile               # HF Spaces compatible (port 7860)
├── baseline.py              # Reproducible baseline scorer
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI server
    ├── traffic_environment.py   # Core simulation logic
    └── graders.py           # Task graders (easy/medium/hard → score 0–1)
```

---

##  Simulation Details

- **Arrivals**: Poisson-distributed per lane; rush-hour multiplier peaks mid-episode for medium/hard
- **Discharge**: Up to 3.5 vehicles clear a green lane per 5-second step
- **Phases**: 4 signal phases per intersection; auto-advance after 30 s (static baseline)
- **Incidents**: Hard task adds random demand spikes (5% probability per step) to simulate accidents
- **Sensors**: Queue length + average wait time per lane — mirrors real loop-detector data

---

## Reproducible Baseline Scores

Run `python baseline.py` to reproduce these scores (heuristic policy, 5 seeds):

| Task | Score |
|---|---|
| Easy | ~0.72 |
| Medium | ~0.65 |
| Hard | ~0.54 |

---

## License

MIT — free to use for research and hackathon purposes.