"""Stable-Baselines3 PPO training, persistence, and inference utilities."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from gymnasium import ActionWrapper, Env, spaces
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from flymaze.config import PPOHyperparameters, ScratchTrainingConfig
from flymaze.env import FlyMazeEnv
from flymaze.evaluation import evaluate_agent


class SB3AgentAdapter:
    """Adapt an SB3 model to the common FlyMaze evaluation protocol."""

    def __init__(self, model: PPO, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def predict(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        values = np.asarray(action, dtype=np.float32)
        if (
            isinstance(self.model.action_space, spaces.Box)
            and float(self.model.action_space.low[0]) < 0.0
        ):
            values = values.copy()
            values[0] = (values[0] + 1.0) / 2.0
        return values


class SymmetricPPOActionWrapper(ActionWrapper[NDArray[np.float32], NDArray[np.float32]]):
    """Expose symmetric policy actions while preserving FlyMaze's public contract."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        values = np.asarray(action, dtype=np.float32).copy()
        if values.shape != (2,):
            raise ValueError("symmetric PPO action must have shape (2,)")
        values[0] = (values[0] + 1.0) / 2.0
        return values


def _make_training_env() -> SymmetricPPOActionWrapper:
    return SymmetricPPOActionWrapper(FlyMazeEnv("train_l_left"))


def train_ppo(
    hyperparameters: PPOHyperparameters,
    training: ScratchTrainingConfig,
    output_directory: Path,
) -> dict[str, object]:
    """Train PPO from random weights, save it, reload it, and evaluate it."""

    rollout_size = training.n_envs * hyperparameters.n_steps
    if rollout_size % hyperparameters.batch_size != 0:
        raise ValueError("batch_size must divide n_envs * n_steps")
    output_directory.mkdir(parents=True, exist_ok=True)
    vec_env = make_vec_env(
        _make_training_env,
        n_envs=training.n_envs,
        seed=training.seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "forkserver"},
    )
    model_path = output_directory / "ppo_flymaze"
    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=hyperparameters.learning_rate,
            n_steps=hyperparameters.n_steps,
            batch_size=hyperparameters.batch_size,
            gamma=hyperparameters.gamma,
            gae_lambda=hyperparameters.gae_lambda,
            ent_coef=hyperparameters.entropy_coefficient,
            seed=training.seed,
            verbose=1,
            tensorboard_log=str(output_directory / "tensorboard"),
            policy_kwargs={"net_arch": list(hyperparameters.network)},
            device=hyperparameters.device,
        )
        model.learn(total_timesteps=training.total_timesteps, progress_bar=False)
        model.save(model_path)
        actual_transitions = model.num_timesteps
    finally:
        vec_env.close()

    reloaded = PPO.load(model_path, device=hyperparameters.device)
    adapter = SB3AgentAdapter(reloaded)
    return {
        "hyperparameters": asdict(hyperparameters),
        "training": asdict(training),
        "actual_transitions": actual_transitions,
        "model_path": str(model_path.with_suffix(".zip")),
        "train_evaluation": evaluate_agent("train_l_left", adapter, training.evaluation_seeds),
        "held_out_evaluation": evaluate_agent(
            "test_straight_unknown", adapter, training.evaluation_seeds
        ),
    }
