"""Straight-corridor geometry and wall-contact telemetry used by FlyMaze maps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
from flygym import Simulation
from flygym.compose import FlatGroundWorld, NeuroMechFly

from flymaze.wall_collision import (
    MAZE_WALL_GEOM_GROUP,
    WALL_COLLISION_SEGMENTS,
    add_explicit_wall_pairs,
)


@dataclass(frozen=True, slots=True)
class CorridorSpec:
    """Physical dimensions for the straight held-out corridor."""

    width_mm: float
    length_mm: float = 30.0
    wall_thickness_mm: float = 1.0
    wall_height_mm: float = 3.0
    center_x_mm: float = 12.0

    def __post_init__(self) -> None:
        if self.width_mm <= 0:
            raise ValueError("corridor width must be positive")
        if self.length_mm <= 0 or self.wall_thickness_mm <= 0 or self.wall_height_mm <= 0:
            raise ValueError("corridor wall dimensions must be positive")


def corridor_world_configurator(
    spec: CorridorSpec,
) -> Callable[[FlatGroundWorld, NeuroMechFly], None]:
    """Create two box walls and explicit pairs for the walking envelope."""

    def configure(world: FlatGroundWorld, fly: NeuroMechFly) -> None:
        wall_y = spec.width_mm / 2.0 + spec.wall_thickness_mm / 2.0
        walls = [
            world.mjcf_root.worldbody.add_geom(
                name="corridor_left_wall",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[
                    spec.length_mm / 2.0,
                    spec.wall_thickness_mm / 2.0,
                    spec.wall_height_mm / 2.0,
                ],
                pos=[spec.center_x_mm, wall_y, spec.wall_height_mm / 2.0],
                contype=0,
                conaffinity=0,
                group=MAZE_WALL_GEOM_GROUP,
                rgba=[0.4, 0.4, 0.45, 1.0],
            ),
            world.mjcf_root.worldbody.add_geom(
                name="corridor_right_wall",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[
                    spec.length_mm / 2.0,
                    spec.wall_thickness_mm / 2.0,
                    spec.wall_height_mm / 2.0,
                ],
                pos=[spec.center_x_mm, -wall_y, spec.wall_height_mm / 2.0],
                contype=0,
                conaffinity=0,
                group=MAZE_WALL_GEOM_GROUP,
                rgba=[0.4, 0.4, 0.45, 1.0],
            ),
        ]
        add_explicit_wall_pairs(world, fly, walls)

    return configure


@dataclass(slots=True)
class WallContactRecorder:
    """Count active wall constraints at every physics substep."""

    wall_geom_ids: set[int]
    fly_geom_ids: set[int]
    wall_geom_ids_by_side: dict[str, int]
    physics_substeps: int = 0
    contact_substeps: int = 0
    active_constraint_count: int = 0
    max_simultaneous_constraints: int = 0
    minimum_contact_distance: float | None = None
    first_contact_time_seconds: float | None = None
    last_contact_time_seconds: float | None = None
    first_contact_side: str | None = None

    @classmethod
    def from_simulation(
        cls,
        simulation: Simulation,
        wall_geom_names_by_side: dict[str, str] | None = None,
    ) -> WallContactRecorder:
        model = simulation.mj_model
        names = (
            {"left": "corridor_left_wall", "right": "corridor_right_wall"}
            if wall_geom_names_by_side is None
            else wall_geom_names_by_side
        )
        wall_ids_by_side = {
            side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            for side, geom_name in names.items()
        }
        wall_ids = set(wall_ids_by_side.values())
        if any(geom_id < 0 for geom_id in wall_ids):
            raise RuntimeError("compiled corridor wall geom not found")
        fly_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"flymaze/{segment}")
            for segment in WALL_COLLISION_SEGMENTS
        }
        if any(geom_id < 0 for geom_id in fly_ids):
            raise RuntimeError("one or more approved wall-collision geoms were not compiled")
        return cls(wall_ids, fly_ids, wall_ids_by_side)

    def __call__(self, simulation: Simulation) -> None:
        self.physics_substeps += 1
        active_this_step = 0
        for contact_index in range(simulation.mj_data.ncon):
            contact = simulation.mj_data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not (pair & self.wall_geom_ids and pair & self.fly_geom_ids):
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(simulation.mj_model, simulation.mj_data, contact_index, force)
            if float(np.linalg.norm(force[:3])) <= 0.0:
                continue
            active_this_step += 1
            if self.first_contact_side is None:
                wall_id = next(iter(pair & self.wall_geom_ids))
                self.first_contact_side = next(
                    side
                    for side, candidate_id in self.wall_geom_ids_by_side.items()
                    if candidate_id == wall_id
                )
            distance = float(contact.dist)
            if self.minimum_contact_distance is None or distance < self.minimum_contact_distance:
                self.minimum_contact_distance = distance
        if active_this_step > 0:
            if self.first_contact_time_seconds is None:
                self.first_contact_time_seconds = float(simulation.time)
            self.last_contact_time_seconds = float(simulation.time)
            self.contact_substeps += 1
            self.active_constraint_count += active_this_step
            self.max_simultaneous_constraints = max(
                self.max_simultaneous_constraints, active_this_step
            )
