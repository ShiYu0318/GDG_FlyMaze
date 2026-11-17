from flymaze.geometry import (
    CORNER_WALL_NAMES,
    RIGHT_CORNER_WALL_NAMES,
    S_TURN_WALL_NAMES,
    LeftCornerFixtureSpec,
    STurnFixtureSpec,
)


def test_corner_fixture_uses_fixed_corridor_width() -> None:
    spec = LeftCornerFixtureSpec()
    assert spec.corridor_width_mm == 10.0
    assert len(CORNER_WALL_NAMES) == 4


def test_s_turn_fixture_uses_fixed_width_and_seven_walls() -> None:
    assert STurnFixtureSpec().corridor_width_mm == 10.0
    assert len(S_TURN_WALL_NAMES) == 7


def test_right_corner_has_four_mirrored_walls() -> None:
    assert len(RIGHT_CORNER_WALL_NAMES) == 4
