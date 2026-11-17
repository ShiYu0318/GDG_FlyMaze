import pytest

from flymaze.corridor import CorridorSpec


def test_corridor_spec_accepts_production_dimensions() -> None:
    spec = CorridorSpec(width_mm=10.0)
    assert spec.length_mm == 30.0
    assert spec.wall_height_mm == 3.0


def test_corridor_spec_rejects_nonpositive_width() -> None:
    with pytest.raises(ValueError, match="width"):
        CorridorSpec(width_mm=0.0)
