import pytest

from flymaze.maps import (
    MAPS,
    TEST_STRAIGHT_UNKNOWN,
    TRAIN_L_LEFT,
    TRAIN_L_RIGHT,
    VALIDATION_S_TURN_COMPLEX,
)


def test_training_and_held_out_maps_have_distinct_topology() -> None:
    assert TRAIN_L_LEFT.topology.digest() != TEST_STRAIGHT_UNKNOWN.topology.digest()
    assert TRAIN_L_LEFT.split == "train"
    assert TEST_STRAIGHT_UNKNOWN.split == "held_out_test"
    assert set(MAPS) == {
        "train_l_left",
        "train_l_right",
        "test_straight_unknown",
        "validation_s_turn_complex",
    }
    assert TRAIN_L_LEFT.timeout_seconds == 10.0
    assert TEST_STRAIGHT_UNKNOWN.timeout_seconds == 8.0


def test_route_remaining_distance() -> None:
    route = TRAIN_L_LEFT.route
    assert route.total_length_mm == pytest.approx(28.0)
    assert route.remaining_distance_mm((0.0, 0.0)) == pytest.approx(28.0)
    assert route.remaining_distance_mm((10.0, 0.0)) == pytest.approx(18.0)
    assert route.remaining_distance_mm((10.0, 18.0)) == pytest.approx(0.0)


def test_map_odor_fields_have_source_at_one() -> None:
    for definition in MAPS.values():
        field = definition.build_odor_field()
        assert field.sample(*definition.food_xy_mm) == pytest.approx(1.0)


def test_s_turn_has_two_direction_changes_and_40mm_route() -> None:
    assert VALIDATION_S_TURN_COMPLEX.split == "train_auxiliary"
    assert VALIDATION_S_TURN_COMPLEX.route.total_length_mm == pytest.approx(40.0)
    assert VALIDATION_S_TURN_COMPLEX.timeout_seconds == 14.0


def test_right_training_map_is_mirrored_and_versioned() -> None:
    assert TRAIN_L_RIGHT.split == "train"
    assert TRAIN_L_RIGHT.food_xy_mm == (10.0, -18.0)
    assert TRAIN_L_RIGHT.route.total_length_mm == pytest.approx(28.0)
    assert TRAIN_L_RIGHT.topology.digest() != TRAIN_L_LEFT.topology.digest()
