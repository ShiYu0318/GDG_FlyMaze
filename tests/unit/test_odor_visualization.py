import pytest

from flymaze.odor_visualization import ODOR_VISUAL_GEOM_GROUP, odor_color


def test_odor_heatmap_uses_dedicated_geom_group() -> None:
    assert ODOR_VISUAL_GEOM_GROUP == 5


def test_odor_colors_progress_from_blue_to_red() -> None:
    low = odor_color(0.0, 0.35)
    middle = odor_color(0.5, 0.35)
    high = odor_color(1.0, 0.35)
    assert low[2] > low[0]
    assert middle[1] > middle[0]
    assert high[0] > high[2]
    assert low[3] == middle[3] == high[3] == pytest.approx(0.35)


def test_odor_alpha_must_be_valid() -> None:
    with pytest.raises(ValueError, match="alpha"):
        odor_color(0.5, 0.0)
