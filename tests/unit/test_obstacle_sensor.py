import pytest

from flymaze.obstacle_sensor import (
    APPROVED_OBSTACLE_MAX_DISTANCE_MM,
    ObstacleRaySensor,
    normalize_obstacle_distance,
)


@pytest.mark.parametrize(
    ("distance", "maximum", "expected"),
    [
        (None, 5.0, 1.0),
        (-1.0, 5.0, 1.0),
        (0.0, 5.0, 0.0),
        (2.5, 5.0, 0.5),
        (5.0, 5.0, 1.0),
        (7.0, 5.0, 1.0),
    ],
)
def test_obstacle_distance_normalization(
    distance: float | None, maximum: float, expected: float
) -> None:
    assert normalize_obstacle_distance(distance, maximum) == pytest.approx(expected)


def test_default_obstacle_range_is_approved_5mm() -> None:
    assert APPROVED_OBSTACLE_MAX_DISTANCE_MM == 5.0
    assert ObstacleRaySensor().max_distance_mm == 5.0
