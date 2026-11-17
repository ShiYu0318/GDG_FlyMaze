from flymaze.food import FOOD_GEOM_NAME, FOOD_RADIUS_MM


def test_food_fixture_is_fixed_ground_sphere() -> None:
    assert FOOD_GEOM_NAME == "flymaze_food"
    assert FOOD_RADIUS_MM == 0.5
