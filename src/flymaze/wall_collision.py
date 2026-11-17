"""Approved anatomical scope for physical maze-wall contact."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flygym.anatomy import BodySegment
from flygym.compose import ContactParams, FlatGroundWorld, NeuroMechFly

MAZE_WALL_GEOM_GROUP = 3


CORE_WALL_COLLISION_SEGMENTS = (
    "c_thorax",
    "c_head",
    "c_rostrum",
    "c_haustellum",
    "l_eye",
    "r_eye",
    "c_abdomen12",
    "c_abdomen3",
    "c_abdomen4",
    "c_abdomen5",
    "c_abdomen6",
)
LEG_POSITIONS = ("lf", "lm", "lh", "rf", "rm", "rh")
LEG_WALL_COLLISION_LINKS = (
    "coxa",
    "trochanterfemur",
    "tibia",
    "tarsus1",
    "tarsus2",
    "tarsus3",
    "tarsus4",
    "tarsus5",
)
LEG_WALL_COLLISION_SEGMENTS = tuple(
    f"{leg}_{link}" for leg in LEG_POSITIONS for link in LEG_WALL_COLLISION_LINKS
)
WALL_COLLISION_SEGMENTS = CORE_WALL_COLLISION_SEGMENTS + LEG_WALL_COLLISION_SEGMENTS

EXCLUDED_WALL_COLLISION_SEGMENTS = (
    "l_pedicel",
    "l_funiculus",
    "l_arista",
    "r_pedicel",
    "r_funiculus",
    "r_arista",
    "l_wing",
    "r_wing",
    "l_haltere",
    "r_haltere",
)


def add_explicit_wall_pairs(
    world: FlatGroundWorld,
    fly: NeuroMechFly,
    wall_geoms: Sequence[Any],
    contact_params: ContactParams | None = None,
) -> None:
    """Pair every approved walking-envelope geom with each supplied wall geom."""

    params = ContactParams() if contact_params is None else contact_params
    for segment_name in WALL_COLLISION_SEGMENTS:
        segment_geoms = fly.bodyseg_to_mjcfgeom[BodySegment(segment_name)]
        for body_geom in segment_geoms:
            for wall in wall_geoms:
                world.mjcf_root.add_pair(
                    geomname1=body_geom.name,
                    geomname2=wall.name,
                    name=f"{body_geom.name}-{wall.name}-maze",
                    friction=params.get_friction_tuple(),
                    solref=params.get_solref_tuple(),
                    solimp=params.get_solimp_tuple(),
                    margin=params.margin,
                )
