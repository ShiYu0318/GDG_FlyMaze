import numpy as np
import pytest

from flymaze.env import FlyMazeEnv


@pytest.mark.integration
def test_environment_reset_step_and_reuse() -> None:
    env = FlyMazeEnv("test_straight_unknown")
    try:
        observation, info = env.reset(seed=0)
        assert observation.shape == (7,)
        assert env.observation_space.contains(observation)
        assert info["map_id"] == "test_straight_unknown"
        next_observation, reward, terminated, truncated, step_info = env.step(
            np.asarray([1.0, 0.0], dtype=np.float32)
        )
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert not (terminated and truncated)
        assert step_info["episode_steps"] == 1
        reset_observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(reset_observation)
    finally:
        env.close()
