"""Memoryless expert used to warm-start the released PPO workflow."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class StatelessRuleAgent:
    """Obstacle/odor expert that consumes only the standard seven observations."""

    def __init__(
        self,
        odor_turn_gain: float = 40.0,
        obstacle_odor_direction_threshold: float = 5e-4,
    ) -> None:
        self.odor_turn_gain = odor_turn_gain
        self.obstacle_odor_direction_threshold = obstacle_odor_direction_threshold

    def predict(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (7,):
            raise ValueError("StatelessRuleAgent requires the seven-dimensional observation")
        front, left_obstacle, right_obstacle, odor_left, odor_right, _, _ = values
        if min(front, left_obstacle, right_obstacle) < 1.0:
            odor_delta = odor_left - odor_right
            if abs(odor_delta) >= self.obstacle_odor_direction_threshold:
                turn = 1.0 if odor_delta > 0.0 else -1.0
            else:
                turn = -1.0 if left_obstacle < right_obstacle else 1.0
            return np.asarray([0.5, turn], dtype=np.float32)
        turn = float(np.clip(self.odor_turn_gain * (odor_left - odor_right), -1.0, 1.0))
        return np.asarray([1.0 - 0.5 * abs(turn), turn], dtype=np.float32)
