"""Thin, headless FlyGym v2.1.1 CPU simulation wrapper."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import cos, sin

import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld, NeuroMechFly
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
)
from numpy.typing import NDArray

from flymaze.action import DescendingSignal

WorldConfigurator = Callable[[FlatGroundWorld, NeuroMechFly], None]
SubstepCallback = Callable[[Simulation], None]


@dataclass(frozen=True, slots=True)
class FlyState:
    """Minimal sampled state consumed by the navigation environment."""

    simulation_time: float
    thorax_position: NDArray[np.float64]
    thorax_quaternion: NDArray[np.float64]


class FlySimulation:
    """Own a FlyGym CPU simulation and official turning controller."""

    def __init__(
        self,
        physics_timestep_seconds: float,
        seed: int = 0,
        world_configurator: WorldConfigurator | None = None,
        spawn_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.8),
        spawn_yaw_radians: float = 0.0,
    ) -> None:
        if physics_timestep_seconds <= 0:
            raise ValueError("physics_timestep_seconds must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self._physics_timestep_seconds = float(physics_timestep_seconds)
        self._seed = seed
        self._world_configurator = world_configurator
        self._spawn_position_mm = spawn_position_mm
        self._spawn_yaw_radians = float(spawn_yaw_radians)
        self._simulation: Simulation | None = None
        self._fly = None
        self._controller: HybridTurningController | None = None
        self._thorax_index: int | None = None
        self._initial_joint_angles: NDArray[np.float64] | None = None

    def __enter__(self) -> FlySimulation:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def simulation(self) -> Simulation:
        if self._simulation is None:
            raise RuntimeError("simulation is not open")
        return self._simulation

    @property
    def controller(self) -> HybridTurningController:
        if self._controller is None:
            raise RuntimeError("simulation is not open")
        return self._controller

    def open(self) -> None:
        """Build and warm up the official CPU locomotion simulation."""

        if self._simulation is not None:
            return
        fly = make_locomotion_fly(name="flymaze", add_adhesion=True, colorize=False)
        world = FlatGroundWorld()
        world.add_fly(
            fly,
            list(self._spawn_position_mm),
            Rotation3D(
                "quat",
                [
                    cos(self._spawn_yaw_radians / 2.0),
                    0.0,
                    0.0,
                    sin(self._spawn_yaw_radians / 2.0),
                ],
            ),
            bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
            add_ground_contact_sensors=False,
        )
        if self._world_configurator is not None:
            self._world_configurator(world, fly)
        simulation = Simulation(world, timestep=self._physics_timestep_seconds)
        steps = PreprogrammedSteps()
        dof_order = fly.get_actuated_jointdofs_order("position")
        controller = HybridTurningController(
            timestep=simulation.timestep,
            preprogrammed_steps=steps,
            output_dof_order=dof_order,
        )
        try:
            simulation.reset()
            controller.reset(seed=self._seed)
            initial_joint_angles = np.asarray(
                steps.default_pose_by_dof_order(dof_order), dtype=np.float64
            )
            initial_action = LocomotionAction(
                joint_angles=initial_joint_angles,
                adhesion_onoff=np.ones(6, dtype=bool),
            )
            apply_locomotion_action(simulation, fly.name, initial_action)
            simulation.warmup()
        except BaseException:
            simulation.close()
            raise

        self._fly = fly
        self._simulation = simulation
        self._controller = controller
        self._thorax_index = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
        self._initial_joint_angles = initial_joint_angles

    def reset(self, seed: int | None = None) -> FlyState:
        """Reset state and controller without recompiling the static world."""

        if self._fly is None or self._initial_joint_angles is None:
            raise RuntimeError("simulation is not open")
        reset_seed = self._seed if seed is None else seed
        if reset_seed < 0:
            raise ValueError("seed must be non-negative")
        self.simulation.reset()
        self.controller.reset(seed=reset_seed)
        initial_action = LocomotionAction(
            joint_angles=self._initial_joint_angles.copy(),
            adhesion_onoff=np.ones(6, dtype=bool),
        )
        apply_locomotion_action(self.simulation, self._fly.name, initial_action)
        self.simulation.warmup()
        return self.sample_state()

    def sample_state(self) -> FlyState:
        """Read the current thorax pose through FlyGym's public API."""

        if self._fly is None or self._thorax_index is None:
            raise RuntimeError("simulation is not open")
        positions = self.simulation.get_body_positions(self._fly.name)
        rotations = self.simulation.get_body_rotations(self._fly.name)
        return FlyState(
            simulation_time=float(self.simulation.time),
            thorax_position=np.asarray(positions[self._thorax_index], dtype=np.float64).copy(),
            thorax_quaternion=np.asarray(rotations[self._thorax_index], dtype=np.float64).copy(),
        )

    def step_controller(
        self,
        signal: DescendingSignal,
        substeps: int = 1,
        substep_callback: SubstepCallback | None = None,
    ) -> FlyState:
        """Advance the official controller each physics substep using one low-level probe signal."""

        if substeps < 1:
            raise ValueError("substeps must be at least one")
        if self._fly is None:
            raise RuntimeError("simulation is not open")
        values = signal.as_array()
        for _ in range(substeps):
            observation = HybridControllerObservation.from_sim(self.simulation, self._fly.name)
            action = self.controller.step(values, observation)
            apply_locomotion_action(self.simulation, self._fly.name, action)
            self.simulation.step()
            if substep_callback is not None:
                substep_callback(self.simulation)
        return self.sample_state()

    def close(self) -> None:
        """Release MuJoCo resources; safe to call more than once."""

        if self._simulation is not None:
            self._simulation.close()
        self._simulation = None
        self._fly = None
        self._controller = None
        self._thorax_index = None
        self._initial_joint_angles = None
