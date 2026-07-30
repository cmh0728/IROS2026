from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from earth_rover.core.types import CandidateTrajectory, TraversabilityOutput
from earth_rover.perception.sam_tp_adapter import SamTpOutputAdapter
from training.sam_tp_reproduction import SamTpPrediction


class Predictor(Protocol):
    def predict(self, image_rgb: np.ndarray) -> SamTpPrediction: ...


@dataclass(frozen=True)
class Phase1FrameResult:
    """Validated SAM-TP output paired with fixed rover-frame candidates."""

    traversability: TraversabilityOutput
    trajectories: tuple[CandidateTrajectory, ...]
    prediction: SamTpPrediction
    image_path: ImageSpacePathProposal


@dataclass(frozen=True)
class ImageSpacePathProposal:
    """Display-only path constrained to connected high-score image pixels."""

    valid: bool
    points_uv: np.ndarray
    mean_score: float
    minimum_score: float
    reason: str


class SamTpPhase1FrameProcessor:
    """Run the common RGB frame boundary used by replay and SDK shadow mode."""

    def __init__(
        self,
        predictor: Predictor,
        trajectories: tuple[CandidateTrajectory, ...],
        model_version: str,
        adapter: SamTpOutputAdapter | None = None,
        minimum_path_score: float = 0.55,
        corridor_half_width_ratio: float = 0.018,
    ) -> None:
        if not trajectories:
            raise ValueError("trajectories must not be empty")
        self.predictor = predictor
        self.trajectories = trajectories
        self.model_version = model_version
        self.adapter = adapter or SamTpOutputAdapter()
        if not 0.0 <= minimum_path_score <= 1.0:
            raise ValueError("minimum_path_score must be in [0, 1]")
        if not 0.0 < corridor_half_width_ratio < 0.25:
            raise ValueError("corridor_half_width_ratio must be in (0, 0.25)")
        self.minimum_path_score = minimum_path_score
        self.corridor_half_width_ratio = corridor_half_width_ratio

    def process(
        self,
        image_rgb: np.ndarray,
        frame_timestamp: float,
    ) -> Phase1FrameResult:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape HxWx3")
        if image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb must use uint8 pixels")
        prediction = self.predictor.predict(image_rgb)
        traversability = self.adapter.adapt(
            prediction,
            image_rgb.shape[:2],
            frame_timestamp,
            self.model_version,
        )
        image_path = propose_image_space_path(
            traversability.score_map,
            traversability.valid_mask,
            self.minimum_path_score,
            self.corridor_half_width_ratio,
        )
        return Phase1FrameResult(
            traversability=traversability,
            trajectories=self.trajectories,
            prediction=prediction,
            image_path=image_path,
        )


def propose_image_space_path(
    score_map: np.ndarray,
    valid_mask: np.ndarray,
    minimum_score: float,
    corridor_half_width_ratio: float,
) -> ImageSpacePathProposal:
    """Find a bottom-to-top image path without crossing low-score pixels.

    This proposal is for perception review only. Pixel displacement has no
    calibrated relationship to rover curvature or steering.
    """

    score = np.asarray(score_map, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if score.ndim != 2 or valid.shape != score.shape:
        raise ValueError("score_map and valid_mask must be matching 2D arrays")
    if not np.isfinite(score).all() or score.size == 0:
        raise ValueError("score_map must be finite and non-empty")
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be in [0, 1]")
    if not 0.0 < corridor_half_width_ratio < 0.25:
        raise ValueError("corridor_half_width_ratio must be in (0, 0.25)")

    height, width = score.shape
    half_width = max(1, round(width * corridor_half_width_ratio))
    safe = (valid & (score >= minimum_score)).astype(np.uint8)
    corridor_safe = cv2.erode(
        safe,
        np.ones((1, half_width * 2 + 1), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    start_y = min(height - 1, max(0, round(height * 0.92)))
    end_y = min(start_y, max(0, round(height * 0.35)))
    row_step = max(1, height // 50)
    rows = list(range(start_y, end_y - 1, -row_step))
    if rows[-1] != end_y:
        rows.append(end_y)
    x_step = max(1, width // 96)
    xs = np.arange(0, width, x_step, dtype=np.int32)
    if xs[-1] != width - 1:
        xs = np.append(xs, width - 1)
    maximum_lateral_pixels = max(x_step, round(width * 0.055))
    maximum_index_change = max(1, maximum_lateral_pixels // x_step)

    scores = np.full(xs.shape, -np.inf, dtype=np.float64)
    start_allowed = corridor_safe[rows[0], xs]
    center_penalty = 0.08 * np.abs(xs - (width - 1) / 2.0) / max(width / 2.0, 1.0)
    scores[start_allowed] = score[rows[0], xs[start_allowed]] - center_penalty[start_allowed]
    parents: list[np.ndarray] = []
    for row in rows[1:]:
        next_scores = np.full(xs.shape, -np.inf, dtype=np.float64)
        parent = np.full(xs.shape, -1, dtype=np.int32)
        for index in np.flatnonzero(corridor_safe[row, xs]):
            left = max(0, index - maximum_index_change)
            right = min(len(xs), index + maximum_index_change + 1)
            previous = scores[left:right]
            finite = np.isfinite(previous)
            if not finite.any():
                continue
            candidate_indexes = np.arange(left, right)[finite]
            candidate_scores = previous[finite] - (
                0.12
                * np.abs(xs[candidate_indexes] - xs[index])
                / maximum_lateral_pixels
            )
            best_offset = int(np.argmax(candidate_scores))
            parent[index] = int(candidate_indexes[best_offset])
            next_scores[index] = (
                float(candidate_scores[best_offset]) + float(score[row, xs[index]])
            )
        parents.append(parent)
        scores = next_scores

    if not np.isfinite(scores).any():
        return _invalid_image_path("NO_CONNECTED_TRAVERSABLE_PATH")
    current = int(np.argmax(scores))
    indexes = [current]
    for parent in reversed(parents):
        current = int(parent[current])
        if current < 0:
            return _invalid_image_path("NO_CONNECTED_TRAVERSABLE_PATH")
        indexes.append(current)
    indexes.reverse()
    points = np.column_stack(
        (
            xs[np.asarray(indexes, dtype=np.int32)],
            np.asarray(rows, dtype=np.int32),
        )
    ).astype(np.int32)
    centerline = np.zeros(score.shape, dtype=np.uint8)
    cv2.polylines(centerline, [points], False, 1, 1, cv2.LINE_8)
    if np.any((centerline == 1) & ~corridor_safe):
        return _invalid_image_path("NO_CONNECTED_TRAVERSABLE_PATH")
    sampled_scores = score[points[:, 1], points[:, 0]]
    points.setflags(write=False)
    return ImageSpacePathProposal(
        valid=True,
        points_uv=points,
        mean_score=float(sampled_scores.mean()),
        minimum_score=float(sampled_scores.min()),
        reason="CONNECTED_HIGH_TRAVERSABILITY_IMAGE_PATH",
    )


def _invalid_image_path(reason: str) -> ImageSpacePathProposal:
    points = np.empty((0, 2), dtype=np.int32)
    points.setflags(write=False)
    return ImageSpacePathProposal(
        valid=False,
        points_uv=points,
        mean_score=0.0,
        minimum_score=0.0,
        reason=reason,
    )


def draw_image_path_rgb(
    image_rgb: np.ndarray,
    proposal: ImageSpacePathProposal,
    corridor_half_width_ratio: float,
) -> np.ndarray:
    """Draw a display-only path on an RGB frame without changing its geometry."""

    output = image_rgb.copy()
    if not proposal.valid:
        return output
    half_width = max(1, round(image_rgb.shape[1] * corridor_half_width_ratio))
    overlay = output.copy()
    cv2.polylines(
        overlay,
        [proposal.points_uv],
        False,
        (40, 210, 80),
        half_width * 2,
        cv2.LINE_AA,
    )
    output = cv2.addWeighted(output, 0.62, overlay, 0.38, 0.0)
    cv2.polylines(
        output,
        [proposal.points_uv],
        False,
        (255, 255, 255),
        max(2, half_width // 3),
        cv2.LINE_AA,
    )
    return output


def render_trajectory_geometry_rgb(
    trajectories: tuple[CandidateTrajectory, ...],
    width: int,
    height: int,
) -> np.ndarray:
    """Render rover-frame geometry only; this is not a camera projection."""

    if width <= 0 or height <= 0:
        raise ValueError("render dimensions must be positive")
    if not trajectories:
        raise ValueError("trajectories must not be empty")
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    all_points = np.concatenate(
        [
            trajectory.points_xy
            for trajectory in trajectories
        ]
        + [
            trajectory.left_boundary_xy
            for trajectory in trajectories
        ]
        + [
            trajectory.right_boundary_xy
            for trajectory in trajectories
        ],
        axis=0,
    )
    maximum_x = max(float(all_points[:, 0].max()), 0.1)
    maximum_abs_y = max(float(np.abs(all_points[:, 1]).max()), 0.1)
    margin = 24

    def pixel(points: np.ndarray) -> np.ndarray:
        x_pixels = margin + points[:, 0] / maximum_x * (width - margin * 2)
        y_pixels = height / 2.0 - points[:, 1] / maximum_abs_y * (
            height / 2.0 - margin
        )
        return np.rint(np.column_stack((x_pixels, y_pixels))).astype(np.int32)

    cv2.line(canvas, (margin, height // 2), (width - margin, height // 2), (80, 80, 80), 1)
    for trajectory in trajectories:
        color = (60, 220, 60) if trajectory.curvature == 0.0 else (80, 180, 245)
        cv2.polylines(canvas, [pixel(trajectory.left_boundary_xy)], False, (70, 70, 70), 1)
        cv2.polylines(canvas, [pixel(trajectory.right_boundary_xy)], False, (70, 70, 70), 1)
        centerline = pixel(trajectory.points_xy)
        cv2.polylines(canvas, [centerline], False, color, 2)
        endpoint = tuple(int(value) for value in centerline[-1])
        cv2.circle(canvas, endpoint, 3, color, -1)
    cv2.putText(
        canvas,
        "ROVER FRAME: +x forward, +y left",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "GEOMETRY ONLY - NOT CAMERA PROJECTED",
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (80, 180, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas
