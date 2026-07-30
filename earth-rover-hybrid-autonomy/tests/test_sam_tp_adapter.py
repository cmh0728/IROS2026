from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from earth_rover.perception.sam_tp_adapter import (
    SamTpAdapterError,
    SamTpOutputAdapter,
)


@dataclass(frozen=True)
class Prediction:
    traversability_score: np.ndarray
    output_shape: tuple[int, int]
    inference_time_ms: float = 12.5


def prediction(score: np.ndarray | None = None) -> Prediction:
    values = (
        score
        if score is not None
        else np.array([[0.0, 0.25], [0.75, 1.0]], dtype=np.float64)
    )
    return Prediction(values, values.shape)


def test_adapter_normalizes_valid_prediction_deterministically() -> None:
    adapter = SamTpOutputAdapter()
    mask = np.array([[True, False], [True, True]])

    first = adapter.adapt(prediction(), (2, 2), 100.0, "checkpoint:test", mask)
    repeated = adapter.adapt(prediction(), (2, 2), 100.0, "checkpoint:test", mask)

    assert first.score_map.dtype == np.float32
    assert first.valid_mask.dtype == np.bool_
    assert first.confidence == pytest.approx(0.75)
    assert first.frame_timestamp == 100.0
    assert first.inference_time_ms == 12.5
    assert np.array_equal(first.score_map, repeated.score_map)
    assert np.array_equal(first.valid_mask, repeated.valid_mask)
    assert not first.score_map.flags.writeable
    assert not first.valid_mask.flags.writeable


def test_adapter_generates_all_valid_mask_without_model_confidence_guess() -> None:
    output = SamTpOutputAdapter().adapt(
        prediction(np.full((2, 3), 0.5, dtype=np.float32)),
        (2, 3),
        0.0,
        "checkpoint:test",
    )

    assert output.confidence == 1.0
    assert output.valid_mask.all()


@pytest.mark.parametrize(
    "score,reason",
    [
        (np.zeros((2, 2, 1), dtype=np.float32), "INVALID_SCORE_DIMENSION"),
        (np.array([[np.nan]], dtype=np.float32), "NONFINITE_SCORE"),
        (np.array([[np.inf]], dtype=np.float32), "NONFINITE_SCORE"),
        (np.array([[-0.01]], dtype=np.float32), "SCORE_OUT_OF_RANGE"),
        (np.array([[1.01]], dtype=np.float32), "SCORE_OUT_OF_RANGE"),
        (np.ones((1, 1), dtype=np.uint8), "INVALID_SCORE_DTYPE"),
    ],
)
def test_adapter_rejects_invalid_scores(score: np.ndarray, reason: str) -> None:
    item = Prediction(score, score.shape[:2])

    with pytest.raises(SamTpAdapterError) as caught:
        SamTpOutputAdapter().adapt(item, score.shape[:2], 1.0, "checkpoint:test")

    assert caught.value.reason == reason


def test_adapter_rejects_score_frame_and_declared_shape_mismatch() -> None:
    adapter = SamTpOutputAdapter()

    with pytest.raises(SamTpAdapterError, match="SCORE_FRAME_SHAPE_MISMATCH"):
        adapter.adapt(prediction(), (3, 2), 1.0, "checkpoint:test")
    item = Prediction(np.ones((2, 2), dtype=np.float32), (1, 4))
    with pytest.raises(SamTpAdapterError, match="PREDICTION_SHAPE_MISMATCH"):
        adapter.adapt(item, (2, 2), 1.0, "checkpoint:test")


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -0.1])
def test_adapter_rejects_invalid_timestamp(timestamp: float) -> None:
    with pytest.raises(SamTpAdapterError) as caught:
        SamTpOutputAdapter().adapt(
            prediction(),
            (2, 2),
            timestamp,
            "checkpoint:test",
        )

    assert caught.value.reason == "INVALID_FRAME_TIMESTAMP"


@pytest.mark.parametrize("latency", [float("nan"), float("inf"), -0.1])
def test_adapter_rejects_invalid_inference_time(latency: float) -> None:
    item = Prediction(np.ones((2, 2), dtype=np.float32), (2, 2), latency)

    with pytest.raises(SamTpAdapterError) as caught:
        SamTpOutputAdapter().adapt(item, (2, 2), 1.0, "checkpoint:test")

    assert caught.value.reason == "INVALID_INFERENCE_TIME"


def test_adapter_rejects_invalid_valid_mask_and_timeout() -> None:
    with pytest.raises(SamTpAdapterError, match="VALID_MASK_SHAPE_MISMATCH"):
        SamTpOutputAdapter().adapt(
            prediction(),
            (2, 2),
            1.0,
            "checkpoint:test",
            np.ones((1, 2), dtype=bool),
        )
    with pytest.raises(SamTpAdapterError) as caught:
        SamTpOutputAdapter(maximum_inference_time_ms=10.0).adapt(
            prediction(),
            (2, 2),
            1.0,
            "checkpoint:test",
        )

    assert caught.value.reason == "INFERENCE_TIMEOUT"
