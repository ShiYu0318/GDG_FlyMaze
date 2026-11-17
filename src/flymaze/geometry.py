"""MuJoCo geometry fixtures derived from validated maze topology."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mujoco
from flygym.compose import FlatGroundWorld, NeuroMechFly

from flymaze.maze_dimensions import STANDARD_CORRIDOR_WIDTH_MM, validate_corridor_width
from flymaze.wall_collision import MAZE_WALL_GEOM_GROUP, add_explicit_wall_pairs

CORNER_WALL_NAMES = {
    "outer_bottom": "corner_outer_bottom",
    "inner_horizontal": "corner_inner_horizontal",
    "inner_vertical": "corner_inner_vertical",
    "outer_vertical": "corner_outer_vertical",
}

RIGHT_CORNER_WALL_NAMES = {
    "outer_top": "right_corner_outer_top",
    "inner_horizontal": "right_corner_inner_horizontal",
    "inner_vertical": "right_corner_inner_vertical",
    "outer_vertical": "right_corner_outer_vertical",
}

S_TURN_WALL_NAMES = {
    "lower_outer": "s_turn_lower_outer",
    "lower_inner": "s_turn_lower_inner",
    "left_vertical": "s_turn_left_vertical",
    "right_vertical_lower": "s_turn_right_vertical_lower",
    "upper_inner": "s_turn_upper_inner",
    "upper_outer": "s_turn_upper_outer",
    "exit_end": "s_turn_exit_end",
}


@dataclass(frozen=True, slots=True)
class LeftCornerFixtureSpec:
    """A fixed-width L-corner test fixture, not an official training map."""

    corridor_width_mm: float = STANDARD_CORRIDOR_WIDTH_MM
    wall_thickness_mm: float = 1.0
    wall_height_mm: float = 3.0
    corner_inner_x_mm: float = 5.0
    entrance_start_x_mm: float = -5.0
    exit_end_y_mm: float = 20.0

    def __post_init__(self) -> None:
        validate_corridor_width(self.corridor_width_mm)
        if self.wall_thickness_mm <= 0 or self.wall_height_mm <= 0:
            raise ValueError("corner wall dimensions must be positive")


def left_corner_world_configurator(
    spec: LeftCornerFixtureSpec,
) -> Callable[[FlatGroundWorld, NeuroMechFly], None]:
    """Create a 90-degree left turn with fixed-width entrance and exit corridors."""

    def configure(world: FlatGroundWorld, fly: NeuroMechFly) -> None:
        half_width = spec.corridor_width_mm / 2.0
        half_thickness = spec.wall_thickness_mm / 2.0
        half_height = spec.wall_height_mm / 2.0
        outer_x = spec.corner_inner_x_mm + spec.corridor_width_mm

        bottom_start_x = spec.entrance_start_x_mm
        bottom_end_x = outer_x
        bottom_length = bottom_end_x - bottom_start_x
        inner_horizontal_length = spec.corner_inner_x_mm - spec.entrance_start_x_mm
        inner_vertical_length = spec.exit_end_y_mm - half_width
        outer_vertical_length = spec.exit_end_y_mm + half_width

        walls = [
            world.mjcf_root.worldbody.add_geom(
                name=CORNER_WALL_NAMES["outer_bottom"],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[bottom_length / 2.0, half_thickness, half_height],
                pos=[
                    (bottom_start_x + bottom_end_x) / 2.0,
                    -half_width - half_thickness,
                    half_height,
                ],
                contype=0,
                conaffinity=0,
                group=MAZE_WALL_GEOM_GROUP,
            ),
            world.mjcf_root.worldbody.add_geom(
                name=CORNER_WALL_NAMES["inner_horizontal"],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[inner_horizontal_length / 2.0, half_thickness, half_height],
                pos=[
                    (spec.entrance_start_x_mm + spec.corner_inner_x_mm) / 2.0,
                    half_width + half_thickness,
                    half_height,
                ],
                contype=0,
                conaffinity=0,
                group=MAZE_WALL_GEOM_GROUP,
            ),
            world.mjcf_root.worldbody.add_geom(
                name=CORNER_WALL_NAMES["inner_vertical"],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_thickness, inner_vertical_length / 2.0, half_height],
                pos=[
                    spec.corner_inner_x_mm - half_thickness,
                    (half_width + spec.exit_end_y_mm) / 2.0,
                    half_height,
                ],
                contype=0,
                conaffinity=0,
                group=MAZE_WALL_GEOM_GROUP,
            ),
            world.mjcf_root.worldbody.add_geom(
                name=CORNER_WALL_NAMES["outer_vertical"],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_thickness, outer_vertical_length / 2.0, half_height],
                pos=[
                    outer_x + half_thickness,
                    (-half_width + spec.exit_end_y_mm) / 2.0,
                    half_height,
                ],
                contype=0,
                conaffinity=0,
                group=MAZE_WALL_GEOM_GROUP,
            ),
        ]
        add_explicit_wall_pairs(world, fly, walls)

    return configure


def right_corner_world_configurator(
    spec: LeftCornerFixtureSpec,
) -> Callable[[FlatGroundWorld, NeuroMechFly], None]:
    """Mirror the validated fixed-width left corner across the world X axis."""

    def configure(world: FlatGroundWorld, fly: NeuroMechFly) -> None:
        half_width = spec.corridor_width_mm / 2.0
        half_thickness = spec.wall_thickness_mm / 2.0
        half_height = spec.wall_height_mm / 2.0
        outer_x = spec.corner_inner_x_mm + spec.corridor_width_mm
        bottom_start_x = spec.entrance_start_x_mm
        bottom_end_x = outer_x
        bottom_length = bottom_end_x - bottom_start_x
        inner_horizontal_length = spec.corner_inner_x_mm - spec.entrance_start_x_mm
        exit_end_y = -spec.exit_end_y_mm
        inner_vertical_length = -half_width - exit_end_y
        outer_vertical_length = half_width - exit_end_y
        walls = [
            world.mjcf_root.worldbody.add_geom(
                name=RIGHT_CORNER_WALL_NAMES["outer_top"],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[bottom_length / 2.0, half_thickness, half_height],
                pos=[
                    (bottom_start_x + bottom_end_x) / 2.0,
                    half_width + half_thickness,
                    half_height,
                ],
                contype=0,
                conaffinity=0,
                group=MAZE_WALL_GEOM_GROUP,
            ),
            world.mjcf_root.worldbody.add_geom(
                name=RIGHT_CORNER_WALL_NAMES["inner_horizontal"],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[inner_horizontal_length / 2.0, half_thickness, half_height],
                pos=[
                    (spec.entrance_start_x_mm + spec.corner_inner_x_mm) / 2.0,
                    -half_width - half_thickness,
                    half_height,
                ],
                contype=0,
                conaffinity=0,
                group=MAZE_WALL_GEOM_GROUP,
            ),
            world.mjcf_root.worldbody.add_geom(
                name=RIGHT_CORNER_WALL_NAMES["inner_vertical"],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_thickness, inner_vertical_length / 2.0, half_height],
                pos=[
                    spec.corner_inner_x_mm - half_thickness,
                    (-half_width + exit_end_y) / 2.0,
                    half_height,
                ],
                contype=0,
                conaffinity=0,
                group=MAZE_WALL_GEOM_GROUP,
            ),
            world.mjcf_root.worldbody.add_geom(
                name=RIGHT_CORNER_WALL_NAMES["outer_vertical"],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_thickness, outer_vertical_length / 2.0, half_height],
                pos=[
                    outer_x + half_thickness,
                    (half_width + exit_end_y) / 2.0,
                    half_height,
                ],
                contype=0,
                conaffinity=0,
                group=MAZE_WALL_GEOM_GROUP,
            ),
        ]
        add_explicit_wall_pairs(world, fly, walls)

    return configure


@dataclass(frozen=True, slots=True)
class STurnFixtureSpec:
    """Two-turn fixed-width S-shaped validation fixture."""

    corridor_width_mm: float = STANDARD_CORRIDOR_WIDTH_MM
    wall_thickness_mm: float = 1.0
    wall_height_mm: float = 3.0

    def __post_init__(self) -> None:
        validate_corridor_width(self.corridor_width_mm)
        if self.wall_thickness_mm <= 0 or self.wall_height_mm <= 0:
            raise ValueError("S-turn wall dimensions must be positive")


def s_turn_world_configurator(
    spec: STurnFixtureSpec,
) -> Callable[[FlatGroundWorld, NeuroMechFly], None]:
    """Create east-to-north-to-east connected 10 mm corridors."""

    def configure(world: FlatGroundWorld, fly: NeuroMechFly) -> None:
        half_thickness = spec.wall_thickness_mm / 2.0
        half_height = spec.wall_height_mm / 2.0

        def wall(
            name: str,
            half_x: float,
            half_y: float,
            center_x: float,
            center_y: float,
        ) -> Any:
            return world.mjcf_root.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_x, half_y, half_height],
                pos=[center_x, center_y, half_height],
                contype=0,
                conaffinity=0,
                group=MAZE_WALL_GEOM_GROUP,
            )

        walls = [
            wall(S_TURN_WALL_NAMES["lower_outer"], 10.0, half_thickness, 5.0, -5.5),
            wall(S_TURN_WALL_NAMES["lower_inner"], 5.0, half_thickness, 0.0, 5.5),
            wall(S_TURN_WALL_NAMES["left_vertical"], half_thickness, 7.5, 4.5, 12.5),
            wall(
                S_TURN_WALL_NAMES["right_vertical_lower"],
                half_thickness,
                7.5,
                15.5,
                2.5,
            ),
            wall(S_TURN_WALL_NAMES["upper_inner"], 7.5, half_thickness, 22.5, 9.5),
            wall(S_TURN_WALL_NAMES["upper_outer"], 12.5, half_thickness, 17.5, 20.5),
            wall(S_TURN_WALL_NAMES["exit_end"], half_thickness, 5.0, 30.5, 15.0),
        ]
        add_explicit_wall_pairs(world, fly, walls)

    return configure
