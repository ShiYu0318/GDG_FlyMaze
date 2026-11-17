"""Approved fixed maze dimensions for FlyMaze MVP v0.1."""

STANDARD_CORRIDOR_WIDTH_MM = 10.0


def validate_corridor_width(width_mm: float) -> None:
    """Reject map geometry that deviates from the fixed MVP corridor width."""

    if width_mm != STANDARD_CORRIDOR_WIDTH_MM:
        raise ValueError(
            f"MVP corridor width must be exactly {STANDARD_CORRIDOR_WIDTH_MM} mm, got {width_mm}"
        )
