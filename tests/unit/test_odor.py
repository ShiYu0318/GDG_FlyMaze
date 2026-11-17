import numpy as np
import pytest

from flymaze.odor import (
    APPROVED_ODOR_CELL_SIZE_MM,
    APPROVED_ODOR_DIFFUSION_LENGTH_MM,
    APPROVED_ODOR_PARAMETERS,
    OdorDiffusionParameters,
    solve_static_odor_field,
)

PARAMETERS = OdorDiffusionParameters(
    diffusion_coefficient=1.0,
    decay_rate=0.2,
    source_strength=1.0,
)


def test_source_is_normalized_to_one_and_field_is_symmetric() -> None:
    field = solve_static_odor_field(
        free_mask=np.ones((5, 5), dtype=bool),
        source_cell=(2, 2),
        origin_x_mm=0.0,
        origin_y_mm=0.0,
        cell_size_mm=1.0,
        parameters=PARAMETERS,
    )
    assert field.normalized_concentration[2, 2] == pytest.approx(1.0)
    assert field.normalized_concentration[2, 1] == pytest.approx(
        field.normalized_concentration[2, 3]
    )
    assert field.normalized_concentration[1, 2] == pytest.approx(
        field.normalized_concentration[3, 2]
    )


def test_wall_disconnects_source_from_other_component() -> None:
    mask = np.ones((5, 7), dtype=bool)
    mask[:, 3] = False
    field = solve_static_odor_field(
        free_mask=mask,
        source_cell=(2, 1),
        origin_x_mm=0.0,
        origin_y_mm=0.0,
        cell_size_mm=1.0,
        parameters=PARAMETERS,
    )
    assert np.max(field.normalized_concentration[:, 4:]) == pytest.approx(0.0)


def test_sampling_renormalizes_around_wall_cells() -> None:
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    field = solve_static_odor_field(
        free_mask=mask,
        source_cell=(0, 0),
        origin_x_mm=0.0,
        origin_y_mm=0.0,
        cell_size_mm=1.0,
        parameters=PARAMETERS,
    )
    assert 0.0 <= field.sample(0.5, 0.5) <= 1.0


def test_source_must_be_free_space() -> None:
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    with pytest.raises(ValueError, match="free space"):
        solve_static_odor_field(
            free_mask=mask,
            source_cell=(1, 1),
            origin_x_mm=0.0,
            origin_y_mm=0.0,
            cell_size_mm=1.0,
            parameters=PARAMETERS,
        )


def test_approved_odor_parameters_are_fixed() -> None:
    assert APPROVED_ODOR_CELL_SIZE_MM == 0.25
    assert APPROVED_ODOR_DIFFUSION_LENGTH_MM == 10.0
    assert APPROVED_ODOR_PARAMETERS.diffusion_coefficient == 100.0
    assert APPROVED_ODOR_PARAMETERS.decay_rate == 1.0
    assert APPROVED_ODOR_PARAMETERS.source_strength == 1.0
