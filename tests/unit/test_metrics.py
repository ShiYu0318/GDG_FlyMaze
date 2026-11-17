import numpy as np
import pytest

from flymaze.metrics import quaternion_wxyz_to_up_z, quaternion_wxyz_to_yaw


def test_quaternion_to_yaw_uses_wxyz_order() -> None:
    assert quaternion_wxyz_to_yaw(np.asarray([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0)
    half_angle = np.pi / 4.0
    quaternion = np.asarray([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
    assert quaternion_wxyz_to_yaw(quaternion) == pytest.approx(np.pi / 2.0)


def test_quaternion_helpers_reject_zero_norm() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        quaternion_wxyz_to_yaw(np.zeros(4))


def test_quaternion_up_z_detects_upright_and_upside_down() -> None:
    assert quaternion_wxyz_to_up_z(np.asarray([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert quaternion_wxyz_to_up_z(np.asarray([0.0, 1.0, 0.0, 0.0])) == pytest.approx(-1.0)
