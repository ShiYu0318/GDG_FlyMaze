"""Validated runtime and PPO configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APPROVED_DECISION_INTERVAL_SECONDS = 0.1
APPROVED_PHYSICS_TIMESTEP_SECONDS = 0.0002


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class PPOHyperparameters:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coefficient: float = 0.005
    n_steps: int = 128
    batch_size: int = 64
    network: tuple[int, ...] = (64, 64)
    device: str = "cpu"

    def __post_init__(self) -> None:
        _positive("learning_rate", self.learning_rate)
        _positive("n_steps", self.n_steps)
        _positive("batch_size", self.batch_size)
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in (0, 1]")
        if self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be non-negative")
        if not self.network or any(width <= 0 for width in self.network):
            raise ValueError("network widths must be positive")
        if self.device != "cpu":
            raise ValueError("FlyMaze supports the CPU PyTorch baseline only")


@dataclass(frozen=True, slots=True)
class ScratchTrainingConfig:
    total_timesteps: int = 4096
    seed: int = 0
    n_envs: int = 4
    evaluation_seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        _positive("total_timesteps", self.total_timesteps)
        _positive("n_envs", self.n_envs)
        if self.seed < 0 or not self.evaluation_seeds:
            raise ValueError("training and evaluation seeds are required")
        if any(seed < 0 for seed in self.evaluation_seeds):
            raise ValueError("seeds must be non-negative")


@dataclass(frozen=True, slots=True)
class MultitaskTrainingConfig:
    seed: int = 7
    demonstration_seeds: tuple[int, ...] = (0,)
    behavior_epochs: int = 2000
    post_ppo_behavior_epochs: int = 1000
    ppo_timesteps: int = 512
    n_envs: int = 6
    samples_per_demonstration: int = 128
    evaluation_seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        for name in (
            "behavior_epochs",
            "post_ppo_behavior_epochs",
            "ppo_timesteps",
            "n_envs",
            "samples_per_demonstration",
        ):
            _positive(name, getattr(self, name))
        seeds = (*self.demonstration_seeds, *self.evaluation_seeds)
        if self.seed < 0 or not self.demonstration_seeds or not self.evaluation_seeds:
            raise ValueError("training, demonstration, and evaluation seeds are required")
        if any(seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-negative")


@dataclass(frozen=True, slots=True)
class FlyMazeConfig:
    ppo: PPOHyperparameters
    scratch: ScratchTrainingConfig
    multitask: MultitaskTrainingConfig

    @classmethod
    def from_toml(cls, path: str | Path) -> FlyMazeConfig:
        with Path(path).open("rb") as stream:
            raw: dict[str, Any] = tomllib.load(stream)
        try:
            ppo_raw = raw["ppo"]
            scratch_raw = raw["training"]["scratch"]
            multitask_raw = raw["training"]["multitask"]
            ppo = PPOHyperparameters(
                learning_rate=float(ppo_raw["learning_rate"]),
                gamma=float(ppo_raw["gamma"]),
                gae_lambda=float(ppo_raw["gae_lambda"]),
                entropy_coefficient=float(ppo_raw["entropy_coefficient"]),
                n_steps=int(ppo_raw["n_steps"]),
                batch_size=int(ppo_raw["batch_size"]),
                network=tuple(int(value) for value in ppo_raw["network"]),
                device=str(ppo_raw["device"]),
            )
            scratch = ScratchTrainingConfig(
                total_timesteps=int(scratch_raw["total_timesteps"]),
                seed=int(scratch_raw["seed"]),
                n_envs=int(scratch_raw["n_envs"]),
                evaluation_seeds=tuple(int(value) for value in scratch_raw["evaluation_seeds"]),
            )
            multitask = MultitaskTrainingConfig(
                seed=int(multitask_raw["seed"]),
                demonstration_seeds=tuple(
                    int(value) for value in multitask_raw["demonstration_seeds"]
                ),
                behavior_epochs=int(multitask_raw["behavior_epochs"]),
                post_ppo_behavior_epochs=int(multitask_raw["post_ppo_behavior_epochs"]),
                ppo_timesteps=int(multitask_raw["ppo_timesteps"]),
                n_envs=int(multitask_raw["n_envs"]),
                samples_per_demonstration=int(multitask_raw["samples_per_demonstration"]),
                evaluation_seeds=tuple(int(value) for value in multitask_raw["evaluation_seeds"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid FlyMaze config: {path}") from exc
        return cls(ppo=ppo, scratch=scratch, multitask=multitask)
