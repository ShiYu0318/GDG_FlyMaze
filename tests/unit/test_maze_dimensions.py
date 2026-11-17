import pytest

from flymaze.maze_dimensions import STANDARD_CORRIDOR_WIDTH_MM, validate_corridor_width


def test_standard_corridor_width_is_fixed_at_10mm() -> None:
    assert STANDARD_CORRIDOR_WIDTH_MM == 10.0
    validate_corridor_width(10.0)


@pytest.mark.parametrize("width", [3.1, 9.0, 10.1, 12.0])
def test_nonstandard_corridor_width_is_rejected(width: float) -> None:
    with pytest.raises(ValueError, match=r"exactly 10\.0 mm"):
        validate_corridor_width(width)
