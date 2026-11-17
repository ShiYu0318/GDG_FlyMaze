"""Interactive MuJoCo viewer for manual PPO review under WSLg."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter, sleep

import mujoco
from stable_baselines3 import PPO

from flymaze.config import APPROVED_DECISION_INTERVAL_SECONDS
from flymaze.env import FlyMazeEnv
from flymaze.ppo import SB3AgentAdapter


def run_visual_review(
    *,
    map_id: str,
    seed: int,
    model_path: Path,
    playback_speed: float,
    hold_seconds: float,
    persist: bool,
    show_contacts: bool,
    show_odor: bool,
    odor_stride: int,
    odor_alpha: float,
) -> dict[str, object]:
    """Run one live episode in MuJoCo's passive interactive viewer."""

    if playback_speed <= 0:
        raise ValueError("playback_speed must be positive")
    if hold_seconds < 0:
        raise ValueError("hold_seconds must be non-negative")
    agent = SB3AgentAdapter(PPO.load(model_path, device="cpu"))
    env = FlyMazeEnv(
        map_id,
        visualize_odor=show_odor,
        odor_visual_stride=odor_stride,
        odor_visual_alpha=odor_alpha,
    )
    observation, _ = env.reset(seed=seed)
    viewer_module = __import__("mujoco.viewer", fromlist=["launch_passive"])
    model = env.simulation.simulation.mj_model
    data = env.simulation.simulation.mj_data
    terminated = False
    truncated = False
    total_reward = 0.0
    info: dict[str, object] = {"episode_steps": 0}
    try:
        with viewer_module.launch_passive(model, data) as viewer:
            min_x, max_x, min_y, max_y = env.map_definition.odor_bounds_mm
            viewer.cam.lookat[:] = [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, 0.5]
            viewer.cam.distance = 1.6 * max(max_x - min_x, max_y - min_y)
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -55.0
            # Maze walls and food intentionally use dedicated groups for sensing/render control.
            viewer.opt.geomgroup[3] = 1
            viewer.opt.geomgroup[4] = 1
            viewer.opt.geomgroup[5] = int(show_odor)
            if show_contacts:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
            viewer.sync()
            while viewer.is_running() and not (terminated or truncated):
                started = perf_counter()
                action = agent.predict(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                viewer.sync()
                target_wall_seconds = APPROVED_DECISION_INTERVAL_SECONDS / playback_speed
                remaining = target_wall_seconds - (perf_counter() - started)
                if remaining > 0:
                    sleep(remaining)
            if persist:
                while viewer.is_running():
                    viewer.sync()
                    sleep(0.05)
            else:
                deadline = perf_counter() + hold_seconds
                while viewer.is_running() and perf_counter() < deadline:
                    viewer.sync()
                    sleep(0.05)
    finally:
        env.close()
    steps_value = info.get("episode_steps", 0)
    if isinstance(steps_value, bool) or not isinstance(steps_value, int):
        raise TypeError("episode_steps must be an integer")
    return {
        "map_id": map_id,
        "agent": "ppo",
        "model_path": str(model_path),
        "seed": seed,
        "success": bool(info.get("success", False)),
        "success_role": info.get("success_role"),
        "fallen": bool(info.get("fallen", False)),
        "terminated": terminated,
        "truncated": truncated,
        "steps": steps_value,
        "total_reward": total_reward,
        "route_distance_mm": info.get("route_distance_mm"),
    }
