"""FlyMaze package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flymaze")
except PackageNotFoundError:  # pragma: no cover - only for an uninstalled source tree
    __version__ = "0.0.0+uninstalled"

__all__ = ["__version__"]
