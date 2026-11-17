"""High- and low-level action contracts for FlyMaze."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

TURN_SIGN_CONVENTION = "positive-left-positive-yaw"


def _require_finite(name: str, value: float) -> float:
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numeric


@dataclass(frozen=True, slots=True)
class HighLevelAction:
    """The public action contract; positive `turn` means left/positive yaw."""

    forward: float
    turn: float

    def __post_init__(self) -> None:
        forward = _require_finite("forward", self.forward)
        turn = _require_finite("turn", self.turn)
        if not 0.0 <= forward <= 1.0:
            raise ValueError(f"forward must be in [0, 1], got {forward}")
        if not -1.0 <= turn <= 1.0:
            raise ValueError(f"turn must be in [-1, 1], got {turn}")
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "turn", turn)

    def as_array(self) -> NDArray[np.float64]:
        """Return the stable `[forward, turn]` representation."""

        return np.asarray([self.forward, self.turn], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DescendingSignal:
    """Low-level left/right signal accepted by FlyGym's turning controller."""

    left: float
    right: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", _require_finite("left", self.left))
        object.__setattr__(self, "right", _require_finite("right", self.right))

    def as_array(self) -> NDArray[np.float64]:
        """Return the controller's stable `[left, right]` representation."""

        return np.asarray([self.left, self.right], dtype=np.float64)


class ActionAdapter(Protocol):
    """Boundary between FlyMaze actions and FlyGym controller signals."""

    def to_descending_signal(self, action: HighLevelAction) -> DescendingSignal:
        """Map one public action to a controller signal."""


@dataclass(frozen=True, slots=True)
class FlyMazeActionAdapter:
    """Standard FlyMaze MVP mapping from public actions to FlyGym controller signals."""

    forward_gain: float = 0.8
    turn_gain: float = 0.4
    minimum_signal: float = 0.0
    maximum_signal: float = 1.2

    def __post_init__(self) -> None:
        for name in ("forward_gain", "turn_gain", "minimum_signal", "maximum_signal"):
            object.__setattr__(self, name, _require_finite(name, getattr(self, name)))
        if self.forward_gain < 0 or self.turn_gain < 0:
            raise ValueError("forward_gain and turn_gain must be non-negative")
        if self.minimum_signal < 0:
            raise ValueError("FlyMaze MVP mapping forbids negative descending signals")
        if self.maximum_signal <= self.minimum_signal:
            raise ValueError("maximum_signal must be greater than minimum_signal")

    def to_descending_signal(self, action: HighLevelAction) -> DescendingSignal:
        """Map an action using the approved, non-negative MVP formula."""

        base = self.forward_gain * action.forward
        differential = self.turn_gain * action.turn
        left = float(np.clip(base - differential, self.minimum_signal, self.maximum_signal))
        right = float(np.clip(base + differential, self.minimum_signal, self.maximum_signal))
        return DescendingSignal(left=left, right=right)
