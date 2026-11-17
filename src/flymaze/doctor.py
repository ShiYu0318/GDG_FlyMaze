"""Environment diagnostics for reproducibility reports."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import torch

PACKAGES = (
    "flygym",
    "mujoco",
    "gymnasium",
    "stable-baselines3",
    "torch",
    "numpy",
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def environment_report() -> dict[str, object]:
    """Return machine-readable environment metadata."""

    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: _package_version(name) for name in PACKAGES},
        "torch_cuda_available": torch.cuda.is_available(),
    }
