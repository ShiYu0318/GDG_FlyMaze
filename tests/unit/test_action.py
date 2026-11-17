import math

import numpy as np
import pytest

from flymaze.action import (
    TURN_SIGN_CONVENTION,
    FlyMazeActionAdapter,
    HighLevelAction,
)


@pytest.mark.parametrize(
    ("forward", "turn"),
    [(0.0, -1.0), (0.0, 0.0), (0.5, 0.25), (1.0, 1.0)],
)
def test_high_level_action_accepts_contract_boundaries(forward: float, turn: float) -> None:
    action = HighLevelAction(forward=forward, turn=turn)
    np.testing.assert_array_equal(action.as_array(), [forward, turn])


@pytest.mark.parametrize(
    ("forward", "turn"),
    [(-0.01, 0.0), (1.01, 0.0), (0.5, -1.01), (0.5, 1.01)],
)
def test_high_level_action_rejects_out_of_range_values(forward: float, turn: float) -> None:
    with pytest.raises(ValueError):
        HighLevelAction(forward=forward, turn=turn)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_high_level_action_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError):
        HighLevelAction(forward=value, turn=0.0)


def test_turn_sign_convention_is_explicit() -> None:
    assert TURN_SIGN_CONVENTION == "positive-left-positive-yaw"


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (HighLevelAction(0.0, 0.0), [0.0, 0.0]),
        (HighLevelAction(1.0, 0.0), [0.8, 0.8]),
        (HighLevelAction(1.0, 1.0), [0.4, 1.2]),
        (HighLevelAction(1.0, -1.0), [1.2, 0.4]),
        (HighLevelAction(0.0, 1.0), [0.0, 0.4]),
        (HighLevelAction(0.0, -1.0), [0.4, 0.0]),
    ],
)
def test_action_mapping_matches_approved_probe_points(
    action: HighLevelAction, expected: list[float]
) -> None:
    signal = FlyMazeActionAdapter().to_descending_signal(action)
    np.testing.assert_allclose(signal.as_array(), expected)


def test_action_mapping_never_emits_negative_signal() -> None:
    adapter = FlyMazeActionAdapter()
    for forward in np.linspace(0.0, 1.0, 11):
        for turn in np.linspace(-1.0, 1.0, 21):
            signal = adapter.to_descending_signal(HighLevelAction(forward, turn))
            assert signal.left >= 0.0
            assert signal.right >= 0.0


def test_action_mapping_rejects_negative_signal_floor() -> None:
    with pytest.raises(ValueError, match="forbids negative"):
        FlyMazeActionAdapter(minimum_signal=-0.1)
