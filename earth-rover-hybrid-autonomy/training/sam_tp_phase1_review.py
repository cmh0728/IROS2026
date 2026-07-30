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


class SamTpPhase1FrameProcessor:
    """Run the common RGB frame boundary used by replay and SDK shadow mode."""

    def __init__(
        self,
        predictor: Predictor,
        trajectories: tuple[CandidateTrajectory, ...],
        model_version: str,
        adapter: SamTpOutputAdapter | None = None,
    ) -> None:
        if not trajectories:
            raise ValueError("trajectories must not be empty")
        self.predictor = predictor
        self.trajectories = trajectories
        self.model_version = model_version
        self.adapter = adapter or SamTpOutputAdapter()

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
        return Phase1FrameResult(
            traversability=traversability,
            trajectories=self.trajectories,
            prediction=prediction,
        )


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
