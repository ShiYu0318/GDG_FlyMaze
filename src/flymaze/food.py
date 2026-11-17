"""Fixed food object, explicit target pairs, and physics-substep success latch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment
from flygym.compose import ContactParams, FlatGroundWorld, NeuroMechFly

from flymaze.contact import CONTACT_SEGMENT_WHITELIST, resolve_contact_whitelist

FOOD_GEOM_NAME = "flymaze_food"
FOOD_RADIUS_MM = 0.5
FOOD_GEOM_GROUP = 4


def food_world_configurator(
    food_xy_mm: tuple[float, float],
) -> Callable[[FlatGroundWorld, NeuroMechFly], None]:
    """Add one fixed ground food sphere and explicit pairs to approved target meshes."""

    def configure(world: FlatGroundWorld, fly: NeuroMechFly) -> None:
        food = world.mjcf_root.worldbody.add_geom(
            name=FOOD_GEOM_NAME,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[FOOD_RADIUS_MM, 0.0, 0.0],
            pos=[food_xy_mm[0], food_xy_mm[1], FOOD_RADIUS_MM],
            contype=0,
            conaffinity=0,
            group=FOOD_GEOM_GROUP,
            rgba=[0.2, 0.8, 0.2, 1.0],
        )
        params = ContactParams(margin=0.0)
        for role, segment_name in CONTACT_SEGMENT_WHITELIST.items():
            target_geoms = fly.bodyseg_to_mjcfgeom[BodySegment(segment_name)]
            if len(target_geoms) != 1:
                raise RuntimeError(f"expected one target geom for {role}")
            world.mjcf_root.add_pair(
                geomname1=target_geoms[0].name,
                geomname2=food.name,
                name=f"{target_geoms[0].name}-{food.name}-success",
                friction=params.get_friction_tuple(),
                solref=params.get_solref_tuple(),
                solimp=params.get_solimp_tuple(),
                margin=params.margin,
            )

    return configure


@dataclass(slots=True)
class FoodContactLatch:
    """Latch the first actual approved food contact seen on any physics substep."""

    food_geom_id: int
    target_role_by_geom_id: dict[int, str]
    success: bool = False
    role: str | None = None
    simulation_time: float | None = None
    distance: float | None = None
    force_norm: float | None = None

    @classmethod
    def from_simulation(cls, simulation: Simulation, fly_name: str) -> FoodContactLatch:
        food_id = mujoco.mj_name2id(simulation.mj_model, mujoco.mjtObj.mjOBJ_GEOM, FOOD_GEOM_NAME)
        if food_id < 0:
            raise RuntimeError("food geom not found")
        resolved = resolve_contact_whitelist(simulation.mj_model, fly_name)
        return cls(food_id, {geom_id: role for role, geom_id in resolved.items()})

    def reset(self) -> None:
        self.success = False
        self.role = None
        self.simulation_time = None
        self.distance = None
        self.force_norm = None

    def __call__(self, simulation: Simulation) -> None:
        if self.success:
            return
        for contact_index in range(simulation.mj_data.ncon):
            contact = simulation.mj_data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.food_geom_id not in pair:
                continue
            target_ids = pair & self.target_role_by_geom_id.keys()
            if not target_ids or float(contact.dist) > 0.0:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(simulation.mj_model, simulation.mj_data, contact_index, force)
            force_norm = float(np.linalg.norm(force[:3]))
            if force_norm <= 0.0:
                continue
            target_id = next(iter(target_ids))
            self.success = True
            self.role = self.target_role_by_geom_id[target_id]
            self.simulation_time = float(simulation.time)
            self.distance = float(contact.dist)
            self.force_norm = force_norm
            return
