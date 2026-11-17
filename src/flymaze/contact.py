"""Approved anatomical success-contact whitelist and compiled-ID resolution."""

from __future__ import annotations

import mujoco

CONTACT_SEGMENT_WHITELIST = {
    "head": "c_head",
    "left_forefoot": "lf_tarsus5",
    "right_forefoot": "rf_tarsus5",
}


def resolve_contact_whitelist(model: mujoco.MjModel, fly_name: str) -> dict[str, int]:
    """Resolve the approved semantic targets to compiled MuJoCo geom IDs."""

    resolved: dict[str, int] = {}
    for role, segment in CONTACT_SEGMENT_WHITELIST.items():
        compiled_name = f"{fly_name}/{segment}"
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, compiled_name)
        if geom_id < 0:
            raise RuntimeError(f"approved contact geom not found: {compiled_name}")
        resolved[role] = geom_id
    if len(set(resolved.values())) != len(resolved):
        raise RuntimeError("approved contact roles did not resolve to unique geom IDs")
    return resolved
