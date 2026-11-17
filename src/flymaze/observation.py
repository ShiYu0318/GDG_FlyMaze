"""The fixed seven-dimensional FlyMaze Agent observation contract."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

OBSERVATION_NAMES = (
    "obstacle_front",
    "obstacle_front_left",
    "obstacle_front_right",
    "odor_left",
    "odor_right",
    "forward_velocity",
    "yaw_rate",
)


def _bounded(name: str, value: float, minimum: float, maximum: float) -> float:
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if not minimum <= numeric <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {numeric}")
    return numeric


@dataclass(frozen=True, slots=True)
class SevenDimObservation:
    """Normalized observation exposed to every standard Agent."""

    obstacle_front: float
    obstacle_front_left: float
    obstacle_front_right: float
    odor_left: float
    odor_right: float
    forward_velocity: float
    yaw_rate: float

    def __post_init__(self) -> None:
        for name in OBSERVATION_NAMES[:5]:
            object.__setattr__(self, name, _bounded(name, getattr(self, name), 0.0, 1.0))
        for name in OBSERVATION_NAMES[5:]:
            object.__setattr__(self, name, _bounded(name, getattr(self, name), -1.0, 1.0))

    def as_array(self) -> NDArray[np.float32]:
        """Return the stable observation order as float32 for Gymnasium/PPO."""

        return np.asarray([getattr(self, name) for name in OBSERVATION_NAMES], dtype=np.float32)
