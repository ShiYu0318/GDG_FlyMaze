"""Shared episode evaluation for Starter Kit Agents and PPO models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from flymaze.env import FlyMazeEnv


class PredictAgent(Protocol):
    def predict(self, observation: NDArray[np.float32]) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    map_id: str
    seed: int
    success: bool
    success_role: str | None
    fallen: bool
    truncated: bool
    steps: int
    total_reward: float
    final_route_distance_mm: float
    wall_contact_steps: int


def run_episode(env: FlyMazeEnv, agent: PredictAgent, seed: int) -> EpisodeResult:
    reset_method = getattr(agent, "reset", None)
    if callable(reset_method):
        reset_method()
    observation, reset_info = env.reset(seed=seed)
    total_reward = 0.0
    wall_contact_steps = 0
    terminated = False
    truncated = False
    info = reset_info
    while not (terminated or truncated):
        action = agent.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        wall_contact_steps += int(bool(info.get("wall_contact", False)))
    return EpisodeResult(
        map_id=str(info["map_id"]),
        seed=seed,
        success=bool(info.get("success", False)),
        success_role=(None if info.get("success_role") is None else str(info["success_role"])),
        fallen=bool(info.get("fallen", False)),
        truncated=truncated,
        steps=int(info["episode_steps"]),
        total_reward=float(total_reward),
        final_route_distance_mm=float(info["route_distance_mm"]),
        wall_contact_steps=wall_contact_steps,
    )


def evaluate_agent(
    map_id: str,
    agent: PredictAgent,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one seed is required")
    env = FlyMazeEnv(map_id)
    try:
        results = [run_episode(env, agent, seed) for seed in seeds]
    finally:
        env.close()
    return {
        "map_id": map_id,
        "seeds": list(seeds),
        "episode_count": len(results),
        "success_count": sum(result.success for result in results),
        "fallen_count": sum(result.fallen for result in results),
        "mean_steps": float(np.mean([result.steps for result in results])),
        "mean_total_reward": float(np.mean([result.total_reward for result in results])),
        "results": [asdict(result) for result in results],
    }
