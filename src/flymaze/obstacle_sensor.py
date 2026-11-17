"""Three-direction MuJoCo wall-ray sensor for the standard observation."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import mujoco
import numpy as np
from flygym import Simulation
from numpy.typing import NDArray

from flymaze.wall_collision import MAZE_WALL_GEOM_GROUP

RAY_ANGLES_RADIANS = (0.0, pi / 4.0, -pi / 4.0)
APPROVED_OBSTACLE_MAX_DISTANCE_MM = 5.0


@dataclass(frozen=True, slots=True)
class ObstacleRayReading:
    """Raw and normalized distances in front, front-left, front-right order."""

    raw_distances_mm: tuple[float | None, float | None, float | None]
    normalized: tuple[float, float, float]
    hit_geom_ids: tuple[int | None, int | None, int | None]


def normalize_obstacle_distance(distance_mm: float | None, max_distance_mm: float) -> float:
    """Map contact proximity to 0 and no hit within range to 1."""

    if max_distance_mm <= 0:
        raise ValueError("max_distance_mm must be positive")
    if distance_mm is None or distance_mm < 0 or distance_mm >= max_distance_mm:
        return 1.0
    return float(np.clip(distance_mm / max_distance_mm, 0.0, 1.0))


def _rotate_xy(vector: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    rotation = np.asarray([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]], dtype=np.float64)
    rotated_xy = rotation @ vector[:2]
    return np.asarray([rotated_xy[0], rotated_xy[1], 0.0], dtype=np.float64)


class ObstacleRaySensor:
    """Raycast static maze walls from the actual head position."""

    def __init__(self, max_distance_mm: float = APPROVED_OBSTACLE_MAX_DISTANCE_MM) -> None:
        if max_distance_mm <= 0:
            raise ValueError("max_distance_mm must be positive")
        self.max_distance_mm = float(max_distance_mm)
        self._geom_group_mask = np.zeros(6, dtype=np.uint8)
        self._geom_group_mask[MAZE_WALL_GEOM_GROUP] = 1

    def sense(self, simulation: Simulation, fly_name: str) -> ObstacleRayReading:
        model = simulation.mj_model
        data = simulation.mj_data
        head_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{fly_name}/c_head")
        abdomen_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{fly_name}/c_abdomen12"
        )
        if head_geom_id < 0 or abdomen_geom_id < 0:
            raise RuntimeError("head or abdomen geom not found for obstacle sensing")
        origin = np.asarray(data.geom_xpos[head_geom_id], dtype=np.float64).copy()
        heading = anatomical_heading(simulation, fly_name)

        raw: list[float | None] = []
        normalized: list[float] = []
        hits: list[int | None] = []
        for angle in RAY_ANGLES_RADIANS:
            direction = _rotate_xy(heading, angle)
            geom_id = np.asarray([-1], dtype=np.int32)
            distance = float(
                mujoco.mj_ray(
                    model,
                    data,
                    origin,
                    direction,
                    self._geom_group_mask,
                    1,
                    -1,
                    geom_id,
                )
            )
            raw_distance = None if distance < 0 else distance
            raw.append(raw_distance)
            normalized.append(normalize_obstacle_distance(raw_distance, self.max_distance_mm))
            hits.append(None if geom_id[0] < 0 else int(geom_id[0]))
        return ObstacleRayReading(
            raw_distances_mm=(raw[0], raw[1], raw[2]),
            normalized=(normalized[0], normalized[1], normalized[2]),
            hit_geom_ids=(hits[0], hits[1], hits[2]),
        )


def anatomical_heading(simulation: Simulation, fly_name: str) -> NDArray[np.float64]:
    """Return the normalized horizontal abdomen-to-head direction."""

    model = simulation.mj_model
    data = simulation.mj_data
    head_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{fly_name}/c_head")
    abdomen_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{fly_name}/c_abdomen12")
    if head_geom_id < 0 or abdomen_geom_id < 0:
        raise RuntimeError("head or abdomen geom not found for anatomical heading")
    heading = np.asarray(
        data.geom_xpos[head_geom_id] - data.geom_xpos[abdomen_geom_id], dtype=np.float64
    )
    heading[2] = 0.0
    norm = float(np.linalg.norm(heading))
    if norm == 0.0:
        raise RuntimeError("head-to-abdomen heading has zero horizontal norm")
    return heading / norm
