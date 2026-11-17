from flymaze.wall_collision import (
    EXCLUDED_WALL_COLLISION_SEGMENTS,
    LEG_WALL_COLLISION_SEGMENTS,
    WALL_COLLISION_SEGMENTS,
)


def test_wall_collision_includes_every_leg_segment() -> None:
    assert len(LEG_WALL_COLLISION_SEGMENTS) == 48
    assert "lf_coxa" in WALL_COLLISION_SEGMENTS
    assert "rf_tarsus5" in WALL_COLLISION_SEGMENTS
    assert "rh_tibia" in WALL_COLLISION_SEGMENTS


def test_wall_collision_excludes_sensory_and_flight_appendages() -> None:
    assert set(WALL_COLLISION_SEGMENTS).isdisjoint(EXCLUDED_WALL_COLLISION_SEGMENTS)
    assert "l_arista" in EXCLUDED_WALL_COLLISION_SEGMENTS
    assert "r_wing" in EXCLUDED_WALL_COLLISION_SEGMENTS
    assert "l_haltere" in EXCLUDED_WALL_COLLISION_SEGMENTS
