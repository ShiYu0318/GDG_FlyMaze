import math

import numpy as np
import pytest

from flymaze.observation import OBSERVATION_NAMES, SevenDimObservation


def test_observation_has_exactly_seven_values_in_stable_order() -> None:
    observation = SevenDimObservation(0.1, 0.2, 0.3, 0.4, 0.5, -0.6, 0.7)
    assert len(OBSERVATION_NAMES) == 7
    assert observation.as_array().dtype == np.float32
    np.testing.assert_allclose(observation.as_array(), [0.1, 0.2, 0.3, 0.4, 0.5, -0.6, 0.7])


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf])
def test_distance_and_odor_values_are_bounded(value: float) -> None:
    with pytest.raises(ValueError):
        SevenDimObservation(value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("value", [-1.01, 1.01, math.nan, math.inf])
def test_motion_values_are_bounded(value: float) -> None:
    with pytest.raises(ValueError):
        SevenDimObservation(0.0, 0.0, 0.0, 0.0, 0.0, value, 0.0)
