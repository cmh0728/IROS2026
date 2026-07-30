from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from earth_rover.core.types import CandidateTrajectory


DEFAULT_CURVATURES = (-0.80, -0.50, -0.25, 0.0, 0.25, 0.50, 0.80)
MINIMUM_SAMPLE_INTERVAL_M = 0.001


class ConstantCurvatureTrajectorySampler:
    """Generate deterministic rover-frame arcs and footprint boundaries.

    The rover frame uses ``+x`` forward and ``+y`` left. Positive curvature
    turns left, negative curvature turns right, and zero is straight.
    ``rover_width_m`` and ``safety_margin_m`` are explicit because the project
    does not yet contain validated live-rover geometry.
    """

    def __init__(
        self,
        curvatures: Iterable[float],
        horizon_m: float,
        sample_interval_m: float,
        rover_width_m: float,
        safety_margin_m: float,
    ) -> None:
        self.curvatures = _validate_curvatures(curvatures)
        self.horizon_m = _finite_positive(horizon_m, "horizon_m")
        self.sample_interval_m = _finite_positive(
            sample_interval_m,
            "sample_interval_m",
        )
        if self.sample_interval_m < MINIMUM_SAMPLE_INTERVAL_M:
            raise ValueError(
                f"sample_interval_m must be at least {MINIMUM_SAMPLE_INTERVAL_M}"
            )
        self.rover_width_m = _finite_positive(rover_width_m, "rover_width_m")
        self.safety_margin_m = _finite_nonnegative(
            safety_margin_m,
            "safety_margin_m",
        )
        self.effective_half_width_m = (
            self.rover_width_m / 2.0 + self.safety_margin_m
        )

    def sample(self) -> tuple[CandidateTrajectory, ...]:
        distances = _sample_distances(self.horizon_m, self.sample_interval_m)
        return tuple(
            self._sample_curvature(curvature, distances)
            for curvature in self.curvatures
        )

    def _sample_curvature(
        self,
        curvature: float,
        distances: np.ndarray,
    ) -> CandidateTrajectory:
        headings = curvature * distances
        if curvature == 0.0:
            x = distances.copy()
            y = np.zeros_like(distances)
        else:
            x = np.sin(headings) / curvature
            y = (1.0 - np.cos(headings)) / curvature
        points = np.column_stack((x, y))
        normals = np.column_stack((-np.sin(headings), np.cos(headings)))
        offset = self.effective_half_width_m * normals
        left = points + offset
        right = points - offset
        arrays = (points, headings, distances.copy(), left, right)
        for array in arrays:
            array.setflags(write=False)
        return CandidateTrajectory(
            curvature=curvature,
            points_xy=points,
            headings_rad=headings,
            horizon_m=self.horizon_m,
            sample_distances_m=arrays[2],
            left_boundary_xy=left,
            right_boundary_xy=right,
            effective_half_width_m=self.effective_half_width_m,
        )


def _sample_distances(horizon_m: float, interval_m: float) -> np.ndarray:
    full_steps = int(math.floor(horizon_m / interval_m))
    distances = np.arange(full_steps + 1, dtype=np.float64) * interval_m
    if not math.isclose(float(distances[-1]), horizon_m, rel_tol=0.0, abs_tol=1e-12):
        distances = np.append(distances, horizon_m)
    else:
        distances[-1] = horizon_m
    return distances


def _validate_curvatures(values: Iterable[float]) -> tuple[float, ...]:
    try:
        curvatures = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("curvatures must be a finite iterable") from exc
    if not curvatures:
        raise ValueError("curvatures must not be empty")
    if not all(math.isfinite(value) for value in curvatures):
        raise ValueError("curvatures must contain only finite values")
    normalized = tuple(0.0 if value == 0.0 else value for value in curvatures)
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate curvatures are not allowed")
    return normalized


def _finite_positive(value: object, name: str) -> float:
    parsed = _finite(value, name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _finite_nonnegative(value: object, name: str) -> float:
    parsed = _finite(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _finite(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed
