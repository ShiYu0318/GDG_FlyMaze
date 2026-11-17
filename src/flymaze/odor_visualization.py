"""Non-colliding MuJoCo heatmap tiles for manual review of the static odor field."""

from __future__ import annotations

import mujoco
import numpy as np
from flygym.compose import FlatGroundWorld

from flymaze.odor import StaticOdorField

ODOR_VISUAL_GEOM_GROUP = 5


def odor_color(concentration: float, alpha: float) -> tuple[float, float, float, float]:
    """Map normalized odor to blue-cyan-green-yellow-red review colors."""

    value = float(np.clip(concentration, 0.0, 1.0))
    if not 0.0 < alpha <= 1.0:
        raise ValueError("odor visualization alpha must be in (0,1]")
    if value < 0.5:
        fraction = value * 2.0
        red = 0.0
        green = fraction
        blue = 1.0 - 0.5 * fraction
    else:
        fraction = (value - 0.5) * 2.0
        red = fraction
        green = 1.0 - 0.2 * fraction
        blue = 0.5 * (1.0 - fraction)
    return red, green, blue, alpha


def add_odor_field_overlay(
    world: FlatGroundWorld,
    field: StaticOdorField,
    *,
    stride: int = 4,
    alpha: float = 0.35,
) -> int:
    """Add downsampled, non-colliding floor tiles and return the tile count."""

    if stride < 1:
        raise ValueError("odor visualization stride must be at least one")
    spacing = field.cell_size_mm * stride
    half_tile = 0.46 * spacing
    tile_count = 0
    for row in range(0, field.free_mask.shape[0], stride):
        for column in range(0, field.free_mask.shape[1], stride):
            if not field.free_mask[row, column]:
                continue
            concentration = float(field.normalized_concentration[row, column])
            x_position = field.origin_x_mm + column * field.cell_size_mm
            y_position = field.origin_y_mm + row * field.cell_size_mm
            world.mjcf_root.worldbody.add_geom(
                name=f"odor_tile_{row}_{column}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[half_tile, half_tile, 0.005],
                pos=[x_position, y_position, 0.012],
                rgba=list(odor_color(concentration, alpha)),
                contype=0,
                conaffinity=0,
                group=ODOR_VISUAL_GEOM_GROUP,
            )
            tile_count += 1
    return tile_count
