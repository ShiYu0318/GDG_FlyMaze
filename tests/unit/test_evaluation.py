import numpy as np

from flymaze.evaluation import EpisodeResult


def test_episode_result_fields_are_numeric_and_serializable() -> None:
    result = EpisodeResult("map", 0, False, None, False, True, 10, 1.0, 2.0, 0)
    assert result.steps == 10
    assert np.isfinite(result.total_reward)
