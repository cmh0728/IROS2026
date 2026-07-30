from __future__ import annotations

import numpy as np
import pytest

from earth_rover.planning.trajectory_sampler import (
    DEFAULT_CURVATURES,
    ConstantCurvatureTrajectorySampler,
)


def sampler(**overrides) -> ConstantCurvatureTrajectorySampler:
    values = {
        "curvatures": DEFAULT_CURVATURES,
        "horizon_m": 2.0,
        "sample_interval_m": 0.1,
        "rover_width_m": 0.4,
        "safety_margin_m": 0.1,
    }
    values.update(overrides)
    return ConstantCurvatureTrajectorySampler(**values)


def test_sampler_generates_seven_deterministic_candidates() -> None:
    first = sampler().sample()
    repeated = sampler().sample()

    assert len(first) == 7
    assert tuple(item.curvature for item in first) == DEFAULT_CURVATURES
    for left, right in zip(first, repeated):
        assert np.array_equal(left.points_xy, right.points_xy)
        assert np.array_equal(left.headings_rad, right.headings_rad)
        assert not left.points_xy.flags.writeable


def test_zero_curvature_is_exact_straight_line_with_corridor() -> None:
    trajectory = sampler(curvatures=(0.0,)).sample()[0]

    assert np.array_equal(trajectory.points_xy[:, 0], trajectory.sample_distances_m)
    assert np.count_nonzero(trajectory.points_xy[:, 1]) == 0
    assert np.count_nonzero(trajectory.headings_rad) == 0
    assert np.allclose(trajectory.left_boundary_xy[:, 1], 0.3)
    assert np.allclose(trajectory.right_boundary_xy[:, 1], -0.3)
    assert trajectory.effective_half_width_m == pytest.approx(0.3)


def test_positive_turns_left_and_negative_turns_right_symmetrically() -> None:
    right, left = sampler(curvatures=(-0.5, 0.5)).sample()

    assert left.points_xy[-1, 1] > 0.0
    assert right.points_xy[-1, 1] < 0.0
    assert left.headings_rad[-1] > 0.0
    assert right.headings_rad[-1] < 0.0
    assert np.allclose(left.points_xy[:, 0], right.points_xy[:, 0])
    assert np.allclose(left.points_xy[:, 1], -right.points_xy[:, 1])
    assert np.allclose(left.headings_rad, -right.headings_rad)
    assert np.allclose(left.left_boundary_xy[:, 0], right.right_boundary_xy[:, 0])
    assert np.allclose(left.left_boundary_xy[:, 1], -right.right_boundary_xy[:, 1])


def test_sampling_includes_zero_and_exact_horizon_with_bounded_spacing() -> None:
    trajectory = sampler(
        curvatures=(0.25,),
        horizon_m=1.0,
        sample_interval_m=0.3,
    ).sample()[0]

    assert trajectory.sample_distances_m[0] == 0.0
    assert trajectory.sample_distances_m[-1] == 1.0
    assert np.all(np.diff(trajectory.sample_distances_m) > 0.0)
    assert np.all(np.diff(trajectory.sample_distances_m) <= 0.3 + 1e-12)
    assert trajectory.headings_rad[-1] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"horizon_m": 0.0}, "horizon_m"),
        ({"horizon_m": float("nan")}, "horizon_m"),
        ({"sample_interval_m": 0.0}, "sample_interval_m"),
        ({"sample_interval_m": 0.0001}, "at least"),
        ({"curvatures": (0.0, float("inf"))}, "finite"),
        ({"curvatures": (0.0, -0.0)}, "duplicate"),
        ({"rover_width_m": 0.0}, "rover_width_m"),
        ({"safety_margin_m": -0.1}, "safety_margin_m"),
    ],
)
def test_sampler_rejects_invalid_configuration(overrides: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        sampler(**overrides)
