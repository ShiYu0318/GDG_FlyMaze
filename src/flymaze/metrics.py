"""Quaternion helpers used by the production environment."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _normalized_quaternion(quaternion: FloatArray) -> FloatArray:
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    if not np.isfinite(values).all():
        raise ValueError("quaternion must be finite")
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("quaternion norm must be non-zero")
    return values / norm


def quaternion_wxyz_to_yaw(quaternion: FloatArray) -> float:
    """Convert one MuJoCo/FlyGym ``(w, x, y, z)`` quaternion to yaw."""

    w, x, y, z = _normalized_quaternion(quaternion)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(sin_yaw, cos_yaw))


def quaternion_wxyz_to_up_z(quaternion: FloatArray) -> float:
    """Return the world-z component of the body's local up axis."""

    _, x, y, _ = _normalized_quaternion(quaternion)
    return float(1.0 - 2.0 * (x * x + y * y))
