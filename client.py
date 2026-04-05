"""
client.py — Python client for the Traffic Flow environment.

This file lets anyone connect to a running Traffic Flow server
(local, Docker, or HF Space) using a clean Python API — no raw
HTTP calls needed.

The OpenEnv spec requires this file to exist in the environment
package so training frameworks (TRL, SkyRL, etc.) can import it.

Usage:
    # Connect to a local server
    import requests
    from traffic_env.client import TrafficEnvClient

    client = TrafficEnvClient("http://localhost:7860")
    obs = client.reset(seed=42)
    while not obs["done"]:
        action = {"action_type": "extend_green", "intersection_id": 0}
        obs = client.step(action)
    print("Episode reward:", obs["info"]["cumulative_reward"])

    # Connect to a HF Space
    client = TrafficEnvClient("https://YOUR-USERNAME-traffic-env.hf.space")
"""

import requests
from typing import Optional, Dict, Any


class TrafficEnvClient:
    """
    Thin HTTP client for the Traffic Flow OpenEnv server.

    Wraps the /reset, /step, /state, and /health endpoints
    so calling code never needs to deal with raw HTTP.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        """
        Args:
            base_url: URL of the running server.
                      Local:  "http://localhost:7860"
                      Docker: "http://localhost:7860"
                      HF:     "https://YOUR-USERNAME-traffic-env.hf.space"
        """
        # Strip trailing slash so URLs always look clean
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        """Check if the server is running. Returns {"status": "healthy"}."""
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Start a new episode.

        Args:
            seed: Integer seed for reproducibility.
                  Same seed → same episode every time.

        Returns:
            Initial observation as a dict.
        """
        params = {"seed": seed} if seed is not None else {}
        resp = requests.post(
            f"{self.base_url}/reset",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply one action to the environment.

        Args:
            action: Dict with keys:
                    - action_type: "extend_green" or "next_phase"
                    - intersection_id: int (0-indexed)

        Returns:
            New observation as a dict (includes reward and done flag).

        Example:
            obs = client.step({"action_type": "extend_green", "intersection_id": 0})
            print(obs["reward"])   # e.g. +0.23
            print(obs["done"])     # False until episode ends
        """
        resp = requests.post(
            f"{self.base_url}/step",
            json=action,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """
        Get episode-level metadata.

        Returns:
            Dict with episode_id, step_count, cumulative_reward, etc.
        """
        resp = requests.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def run_episode(self, policy_fn, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Convenience method: run a full episode with a given policy function.

        Args:
            policy_fn: Callable that takes an observation dict and returns
                       an action dict. Example:
                           def my_policy(obs):
                               return {"action_type": "next_phase",
                                       "intersection_id": 0}
            seed: Optional seed for reproducibility.

        Returns:
            Final episode state (cumulative_reward, throughput, etc.)
        """
        obs = self.reset(seed=seed)
        while not obs.get("done", False):
            action = policy_fn(obs)
            obs = self.step(action)
        return self.state()


#  Quick smoke test ----------------------------
if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    client = TrafficEnvClient(url)

    print(f"Connecting to {url}...")
    h = client.health()
    print(f"Health: {h}")

    print("Resetting episode (seed=42)...")
    obs = client.reset(seed=42)
    print(f"Initial obs: {obs['total_waiting_vehicles']} waiting, done={obs['done']}")

    print("Taking one action (extend_green)...")
    obs = client.step({"action_type": "extend_green", "intersection_id": 0})
    print(f"After step: reward={obs['reward']:.4f}, throughput={obs['throughput_last_step']}")

    s = client.state()
    print(f"State: step={s['step_count']}, cumulative_reward={s['cumulative_reward']}")
    print("Client smoke test passed.")
