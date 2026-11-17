from pathlib import Path

import pytest

from flymaze.config import (
    APPROVED_DECISION_INTERVAL_SECONDS,
    APPROVED_PHYSICS_TIMESTEP_SECONDS,
    FlyMazeConfig,
    PPOHyperparameters,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_release_config_matches_selected_ppo_training() -> None:
    config = FlyMazeConfig.from_toml(PROJECT_ROOT / "configs/ppo.toml")
    assert config.ppo.network == (64, 64)
    assert config.ppo.device == "cpu"
    assert config.scratch.n_envs == 4
    assert config.multitask.seed == 7
    assert config.multitask.demonstration_seeds == (0,)
    assert config.multitask.behavior_epochs == 2000
    assert config.multitask.post_ppo_behavior_epochs == 1000
    assert config.multitask.ppo_timesteps == 512
    assert config.multitask.n_envs == 6
    assert config.multitask.samples_per_demonstration == 128


def test_runtime_timing_contract_is_fixed() -> None:
    assert APPROVED_DECISION_INTERVAL_SECONDS == 0.1
    assert APPROVED_PHYSICS_TIMESTEP_SECONDS == 0.0002


def test_non_cpu_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="CPU"):
        PPOHyperparameters(device="cuda")
