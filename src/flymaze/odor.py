"""Static wall-aware steady-state odor diffusion on a two-dimensional free-space grid."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from flygym import Simulation
from scipy import sparse
from scipy.sparse.linalg import spsolve

APPROVED_ODOR_CELL_SIZE_MM = 0.25
APPROVED_ODOR_DIFFUSION_LENGTH_MM = 10.0


@dataclass(frozen=True, slots=True)
class OdorDiffusionParameters:
    """Calibration inputs; no MVP defaults are selected yet."""

    diffusion_coefficient: float
    decay_rate: float
    source_strength: float

    def __post_init__(self) -> None:
        if self.diffusion_coefficient <= 0:
            raise ValueError("diffusion_coefficient must be positive")
        if self.decay_rate <= 0:
            raise ValueError("decay_rate must be positive")
        if self.source_strength <= 0:
            raise ValueError("source_strength must be positive")


APPROVED_ODOR_PARAMETERS = OdorDiffusionParameters(
    diffusion_coefficient=100.0,
    decay_rate=1.0,
    source_strength=1.0,
)


@dataclass(frozen=True, slots=True)
class StaticOdorField:
    """Normalized steady-state concentration plus the free-space mask."""

    origin_x_mm: float
    origin_y_mm: float
    cell_size_mm: float
    free_mask: np.ndarray
    normalized_concentration: np.ndarray
    raw_concentration: np.ndarray
    source_cell: tuple[int, int]

    def __post_init__(self) -> None:
        if self.cell_size_mm <= 0:
            raise ValueError("cell_size_mm must be positive")
        if self.free_mask.ndim != 2:
            raise ValueError("free_mask must be two-dimensional")
        if self.normalized_concentration.shape != self.free_mask.shape:
            raise ValueError("normalized concentration shape must match free_mask")
        if self.raw_concentration.shape != self.free_mask.shape:
            raise ValueError("raw concentration shape must match free_mask")

    def sample(self, x_mm: float, y_mm: float) -> float:
        """Bilinearly sample nearby free cells without using wall-cell values."""

        column = (x_mm - self.origin_x_mm) / self.cell_size_mm
        row = (y_mm - self.origin_y_mm) / self.cell_size_mm
        if (
            row < 0
            or column < 0
            or row > self.free_mask.shape[0] - 1
            or column > self.free_mask.shape[1] - 1
        ):
            return 0.0
        row0 = int(np.floor(row))
        column0 = int(np.floor(column))
        row1 = min(row0 + 1, self.free_mask.shape[0] - 1)
        column1 = min(column0 + 1, self.free_mask.shape[1] - 1)
        row_fraction = row - row0
        column_fraction = column - column0
        candidates = (
            (row0, column0, (1.0 - row_fraction) * (1.0 - column_fraction)),
            (row0, column1, (1.0 - row_fraction) * column_fraction),
            (row1, column0, row_fraction * (1.0 - column_fraction)),
            (row1, column1, row_fraction * column_fraction),
        )
        weighted_sum = 0.0
        total_weight = 0.0
        for candidate_row, candidate_column, weight in candidates:
            if not self.free_mask[candidate_row, candidate_column]:
                continue
            weighted_sum += weight * float(
                self.normalized_concentration[candidate_row, candidate_column]
            )
            total_weight += weight
        return 0.0 if total_weight == 0.0 else weighted_sum / total_weight


def solve_static_odor_field(
    *,
    free_mask: np.ndarray,
    source_cell: tuple[int, int],
    origin_x_mm: float,
    origin_y_mm: float,
    cell_size_mm: float,
    parameters: OdorDiffusionParameters,
) -> StaticOdorField:
    """Solve `D laplacian(C) - lambda C + S = 0` over connected free cells."""

    mask = np.asarray(free_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("free_mask must be two-dimensional")
    if cell_size_mm <= 0:
        raise ValueError("cell_size_mm must be positive")
    source_row, source_column = source_cell
    if not (0 <= source_row < mask.shape[0] and 0 <= source_column < mask.shape[1]):
        raise ValueError("source_cell is outside the grid")
    if not mask[source_cell]:
        raise ValueError("source_cell must be free space")

    free_cells = [tuple(index) for index in np.argwhere(mask)]
    index_by_cell = {cell: index for index, cell in enumerate(free_cells)}
    matrix = sparse.lil_matrix((len(free_cells), len(free_cells)), dtype=np.float64)
    source = np.zeros(len(free_cells), dtype=np.float64)
    neighbor_scale = parameters.diffusion_coefficient / (cell_size_mm * cell_size_mm)
    for cell, index in index_by_cell.items():
        row, column = cell
        neighbors = [
            candidate
            for candidate in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            )
            if candidate in index_by_cell
        ]
        matrix[index, index] = parameters.decay_rate + neighbor_scale * len(neighbors)
        for neighbor in neighbors:
            matrix[index, index_by_cell[neighbor]] = -neighbor_scale
    source[index_by_cell[source_cell]] = parameters.source_strength
    solved = np.asarray(spsolve(matrix.tocsr(), source), dtype=np.float64)
    solved = np.maximum(solved, 0.0)
    raw = np.zeros(mask.shape, dtype=np.float64)
    for cell, index in index_by_cell.items():
        raw[cell] = solved[index]
    maximum = float(np.max(raw))
    normalized = np.zeros_like(raw) if maximum == 0.0 else raw / maximum
    return StaticOdorField(
        origin_x_mm=origin_x_mm,
        origin_y_mm=origin_y_mm,
        cell_size_mm=cell_size_mm,
        free_mask=mask,
        normalized_concentration=normalized,
        raw_concentration=raw,
        source_cell=source_cell,
    )


class AntennaOdorSampler:
    """Sample a solved field at the actual left/right funiculus positions."""

    def sample(
        self, simulation: Simulation, fly_name: str, field: StaticOdorField
    ) -> tuple[float, float]:
        values: list[float] = []
        for segment_name in ("l_funiculus", "r_funiculus"):
            geom_id = mujoco.mj_name2id(
                simulation.mj_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{fly_name}/{segment_name}",
            )
            if geom_id < 0:
                raise RuntimeError(f"antenna geom not found: {segment_name}")
            position = simulation.mj_data.geom_xpos[geom_id]
            values.append(field.sample(float(position[0]), float(position[1])))
        return values[0], values[1]
