from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from earth_rover.core.types import TraversabilityOutput


class SamTpPredictionLike(Protocol):
    traversability_score: np.ndarray
    output_shape: tuple[int, int]
    inference_time_ms: float


class SamTpAdapterError(ValueError):
    """Rejected SAM-TP output with a stable downstream safety reason."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


class SamTpOutputAdapter:
    """Validate SAM-TP output without changing its score semantics.

    SAM-TP scores near zero indicate low traversability and scores near one
    indicate high traversability. They are continuous model evidence, not
    calibrated probabilities. The output ``confidence`` is only the valid
    pixel fraction and never uses score distance from 0.5.
    """

    def __init__(self, maximum_inference_time_ms: float | None = None) -> None:
        if maximum_inference_time_ms is not None and (
            not math.isfinite(maximum_inference_time_ms)
            or maximum_inference_time_ms <= 0.0
        ):
            raise ValueError("maximum_inference_time_ms must be finite and positive")
        self.maximum_inference_time_ms = maximum_inference_time_ms

    def adapt(
        self,
        prediction: SamTpPredictionLike,
        frame_shape: tuple[int, int],
        frame_timestamp: float,
        model_version: str,
        valid_mask: np.ndarray | None = None,
    ) -> TraversabilityOutput:
        height, width = _validate_frame_shape(frame_shape)
        timestamp = _finite_nonnegative(
            frame_timestamp,
            "INVALID_FRAME_TIMESTAMP",
            "frame_timestamp",
        )
        inference_time_ms = _finite_nonnegative(
            prediction.inference_time_ms,
            "INVALID_INFERENCE_TIME",
            "inference_time_ms",
        )
        if (
            self.maximum_inference_time_ms is not None
            and inference_time_ms > self.maximum_inference_time_ms
        ):
            raise SamTpAdapterError(
                "INFERENCE_TIMEOUT",
                f"{inference_time_ms}ms exceeds {self.maximum_inference_time_ms}ms",
            )
        if not isinstance(model_version, str) or not model_version.strip():
            raise SamTpAdapterError(
                "INVALID_MODEL_VERSION",
                "model_version must be a non-empty string",
            )

        score = np.asarray(prediction.traversability_score)
        if score.ndim != 2:
            raise SamTpAdapterError("INVALID_SCORE_DIMENSION", "score_map must be 2D")
        if score.shape != (height, width):
            raise SamTpAdapterError(
                "SCORE_FRAME_SHAPE_MISMATCH",
                f"score shape {score.shape} differs from frame {(height, width)}",
            )
        if tuple(prediction.output_shape) != score.shape:
            raise SamTpAdapterError(
                "PREDICTION_SHAPE_MISMATCH",
                f"declared output {prediction.output_shape} differs from {score.shape}",
            )
        if not np.issubdtype(score.dtype, np.floating):
            raise SamTpAdapterError(
                "INVALID_SCORE_DTYPE",
                "score_map must use a floating-point dtype",
            )
        if not np.isfinite(score).all():
            raise SamTpAdapterError("NONFINITE_SCORE", "score_map contains NaN or Inf")
        if score.size and (float(score.min()) < 0.0 or float(score.max()) > 1.0):
            raise SamTpAdapterError("SCORE_OUT_OF_RANGE", "score_map must remain in [0, 1]")

        if valid_mask is None:
            mask = np.ones(score.shape, dtype=bool)
        else:
            mask = np.asarray(valid_mask)
            if mask.ndim != 2 or mask.shape != score.shape:
                raise SamTpAdapterError(
                    "VALID_MASK_SHAPE_MISMATCH",
                    f"valid_mask shape {mask.shape} differs from {score.shape}",
                )
            mask = mask.astype(bool, copy=True)

        normalized_score = score.astype(np.float32, copy=True)
        normalized_score.setflags(write=False)
        mask.setflags(write=False)
        return TraversabilityOutput(
            score_map=normalized_score,
            valid_mask=mask,
            confidence=float(mask.mean()) if mask.size else 0.0,
            frame_timestamp=timestamp,
            inference_time_ms=inference_time_ms,
            model_version=model_version.strip(),
        )


def _validate_frame_shape(frame_shape: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(frame_shape, tuple) or len(frame_shape) != 2:
        raise SamTpAdapterError(
            "INVALID_FRAME_SHAPE",
            "frame_shape must be a (height, width) tuple",
        )
    height, width = frame_shape
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise SamTpAdapterError(
            "INVALID_FRAME_SHAPE",
            "frame height and width must be positive integers",
        )
    return height, width


def _finite_nonnegative(value: object, reason: str, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SamTpAdapterError(reason, f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise SamTpAdapterError(reason, f"{name} must be finite and non-negative")
    return parsed
