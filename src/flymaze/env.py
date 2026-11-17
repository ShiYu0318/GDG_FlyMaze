"""Gymnasium-compatible FlyMaze environment for PPO training and evaluation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from flymaze.action import FlyMazeActionAdapter, HighLevelAction
from flymaze.config import (
    APPROVED_DECISION_INTERVAL_SECONDS,
    APPROVED_PHYSICS_TIMESTEP_SECONDS,
)
from flymaze.corridor import WallContactRecorder
from flymaze.food import FoodContactLatch
from flymaze.maps import MAPS, MazeMapDefinition
from flymaze.metrics import quaternion_wxyz_to_up_z, quaternion_wxyz_to_yaw
from flymaze.observation import SevenDimObservation
from flymaze.obstacle_sensor import ObstacleRaySensor, anatomical_heading
from flymaze.odor import AntennaOdorSampler
from flymaze.odor_visualization import add_odor_field_overlay
from flymaze.simulation import FlySimulation, FlyState

FORWARD_VELOCITY_SCALE_MM_PER_S = 15.0
YAW_RATE_SCALE_RAD_PER_S = 3.0
SUCCESS_REWARD = 100.0
TIME_PENALTY = 0.01
COLLISION_PENALTY = 0.05
SHAPING_GAMMA = 0.99
SHAPING_SCALE = 2.0
FALLEN_REQUIRED_STEPS = 2


class FlyMazeEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """One-fly physical maze environment with the fixed seven-dimensional observation."""

    def __init__(
        self,
        map_id: str = "train_l_left",
        *,
        spawn_position_mm: tuple[float, float, float] | None = None,
        spawn_yaw_radians: float = 0.0,
        visualize_odor: bool = False,
        odor_visual_stride: int = 4,
        odor_visual_alpha: float = 0.35,
    ) -> None:
        super().__init__()
        self.metadata = {"render_modes": []}
        try:
            self.map_definition: MazeMapDefinition = MAPS[map_id]
        except KeyError as exc:
            raise ValueError(f"unknown FlyMaze map: {map_id}") from exc
        self.action_space = spaces.Box(
            low=np.asarray([0.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )
        self._adapter = FlyMazeActionAdapter()
        self._obstacle_sensor = ObstacleRaySensor()
        self._odor_sampler = AntennaOdorSampler()
        self._odor_field = self.map_definition.build_odor_field()
        self._simulation: FlySimulation | None = None
        self._food_latch: FoodContactLatch | None = None
        self._wall_recorder: WallContactRecorder | None = None
        self._previous_state: FlyState | None = None
        self._previous_route_distance = 0.0
        self._episode_steps = 0
        self._fallen_steps = 0
        self._seed = 0
        self._spawn_position_mm = (
            (0.0, 0.0, 0.8) if spawn_position_mm is None else spawn_position_mm
        )
        self._spawn_yaw_radians = spawn_yaw_radians
        self._visualize_odor = visualize_odor
        self._odor_visual_stride = odor_visual_stride
        self._odor_visual_alpha = odor_visual_alpha

    @property
    def simulation(self) -> FlySimulation:
        if self._simulation is None:
            raise RuntimeError("environment has not been reset")
        return self._simulation

    def _ensure_open(self) -> None:
        if self._simulation is not None:
            return

        def configure_world(world: Any, fly: Any) -> None:
            self.map_definition.configure_world(world, fly)
            if self._visualize_odor:
                add_odor_field_overlay(
                    world,
                    self._odor_field,
                    stride=self._odor_visual_stride,
                    alpha=self._odor_visual_alpha,
                )

        self._simulation = FlySimulation(
            APPROVED_PHYSICS_TIMESTEP_SECONDS,
            seed=self._seed,
            world_configurator=configure_world,
            spawn_position_mm=self._spawn_position_mm,
            spawn_yaw_radians=self._spawn_yaw_radians,
        )
        self._simulation.open()

    def _route_distance(self, state: FlyState) -> float:
        return self.map_definition.route.remaining_distance_mm(
            (float(state.thorax_position[0]), float(state.thorax_position[1]))
        )

    def _observation(self, state: FlyState, previous: FlyState | None) -> NDArray[np.float32]:
        obstacle = self._obstacle_sensor.sense(self.simulation.simulation, "flymaze")
        odor_left, odor_right = self._odor_sampler.sample(
            self.simulation.simulation, "flymaze", self._odor_field
        )
        if previous is None:
            forward_velocity = 0.0
            yaw_rate = 0.0
        else:
            displacement = state.thorax_position - previous.thorax_position
            heading = anatomical_heading(self.simulation.simulation, "flymaze")
            velocity = displacement / APPROVED_DECISION_INTERVAL_SECONDS
            forward_velocity = float(np.dot(velocity[:2], heading[:2]))
            previous_yaw = quaternion_wxyz_to_yaw(previous.thorax_quaternion)
            current_yaw = quaternion_wxyz_to_yaw(state.thorax_quaternion)
            yaw_delta = float(np.unwrap([previous_yaw, current_yaw])[1] - previous_yaw)
            yaw_rate = yaw_delta / APPROVED_DECISION_INTERVAL_SECONDS
        observation = SevenDimObservation(
            obstacle_front=obstacle.normalized[0],
            obstacle_front_left=obstacle.normalized[1],
            obstacle_front_right=obstacle.normalized[2],
            odor_left=odor_left,
            odor_right=odor_right,
            forward_velocity=float(
                np.clip(forward_velocity / FORWARD_VELOCITY_SCALE_MM_PER_S, -1, 1)
            ),
            yaw_rate=float(np.clip(yaw_rate / YAW_RATE_SCALE_RAD_PER_S, -1, 1)),
        )
        return observation.as_array()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        if seed is not None:
            self._seed = seed
        self._ensure_open()
        state = self.simulation.reset(self._seed)
        self._food_latch = FoodContactLatch.from_simulation(self.simulation.simulation, "flymaze")
        self._wall_recorder = WallContactRecorder.from_simulation(
            self.simulation.simulation,
            self.map_definition.wall_geom_names_by_side,
        )
        self._previous_state = state
        self._previous_route_distance = self._route_distance(state)
        self._episode_steps = 0
        self._fallen_steps = 0
        observation = self._observation(state, None)
        return observation, {
            "map_id": self.map_definition.map_id,
            "route_distance_mm": self._previous_route_distance,
        }

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._previous_state is None or self._food_latch is None or self._wall_recorder is None:
            raise RuntimeError("reset must be called before step")
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (2,):
            raise ValueError("FlyMaze action must have shape (2,)")
        high_level = HighLevelAction(float(values[0]), float(values[1]))
        signal = self._adapter.to_descending_signal(high_level)
        food_latch = self._food_latch
        wall_recorder = self._wall_recorder
        food_latch.reset()
        wall_contact_before = wall_recorder.contact_substeps

        def callback(simulation: Any) -> None:
            food_latch(simulation)
            wall_recorder(simulation)

        state = self.simulation.step_controller(
            signal,
            substeps=round(APPROVED_DECISION_INTERVAL_SECONDS / APPROVED_PHYSICS_TIMESTEP_SECONDS),
            substep_callback=callback,
        )
        self._episode_steps += 1
        route_distance = self._route_distance(state)
        route_length = self.map_definition.route.total_length_mm
        phi_previous = route_length - self._previous_route_distance
        phi_current = route_length - route_distance
        shaping = SHAPING_GAMMA * phi_current - phi_previous
        wall_contact = wall_recorder.contact_substeps > wall_contact_before
        reward = (
            SHAPING_SCALE * shaping - TIME_PENALTY - (COLLISION_PENALTY if wall_contact else 0.0)
        )
        if food_latch.success:
            reward += SUCCESS_REWARD

        up_z = quaternion_wxyz_to_up_z(state.thorax_quaternion)
        self._fallen_steps = self._fallen_steps + 1 if up_z < 0.0 else 0
        fallen = self._fallen_steps >= FALLEN_REQUIRED_STEPS
        terminated = food_latch.success or fallen
        max_steps = round(self.map_definition.timeout_seconds / APPROVED_DECISION_INTERVAL_SECONDS)
        truncated = self._episode_steps >= max_steps and not terminated
        observation = self._observation(state, self._previous_state)
        self._previous_state = state
        self._previous_route_distance = route_distance
        info = {
            "map_id": self.map_definition.map_id,
            "route_distance_mm": route_distance,
            "wall_contact": wall_contact,
            "success": food_latch.success,
            "success_role": food_latch.role,
            "fallen": fallen,
            "up_z": up_z,
            "episode_steps": self._episode_steps,
        }
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._simulation is not None:
            self._simulation.close()
        self._simulation = None
        self._food_latch = None
        self._wall_recorder = None
        self._previous_state = None
