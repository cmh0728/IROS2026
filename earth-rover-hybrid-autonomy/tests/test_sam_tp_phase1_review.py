from __future__ import annotations

import numpy as np

from earth_rover.planning.trajectory_sampler import (
    DEFAULT_CURVATURES,
    ConstantCurvatureTrajectorySampler,
)
from training.sam_tp_phase1_review import (
    SamTpPhase1FrameProcessor,
    render_trajectory_geometry_rgb,
)
from training.sam_tp_reproduction import SamTpPrediction


class Predictor:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def predict(self, image_rgb: np.ndarray) -> SamTpPrediction:
        self.inputs.append(image_rgb.copy())
        score = np.full(image_rgb.shape[:2], 0.75, dtype=np.float32)
        return SamTpPrediction(
            raw_logits=np.ones_like(score),
            traversability_score=score,
            heatmap=np.zeros_like(image_rgb),
            input_shape=image_rgb.shape,
            output_shape=score.shape,
            inference_time_ms=4.0,
            device="test",
        )


def trajectories():
    return ConstantCurvatureTrajectorySampler(
        DEFAULT_CURVATURES,
        horizon_m=2.0,
        sample_interval_m=0.1,
        rover_width_m=0.4,
        safety_margin_m=0.1,
    ).sample()


def test_phase1_processor_accepts_replay_or_sdk_rgb_frame() -> None:
    predictor = Predictor()
    processor = SamTpPhase1FrameProcessor(
        predictor,
        trajectories(),
        "checkpoint:test",
    )
    image = np.zeros((36, 64, 3), dtype=np.uint8)

    result = processor.process(image, 100.0)

    assert len(result.trajectories) == 7
    assert result.traversability.score_map.shape == image.shape[:2]
    assert result.traversability.model_version == "checkpoint:test"
    assert np.array_equal(predictor.inputs[0], image)


def test_geometry_panel_is_deterministic_and_not_blank() -> None:
    first = render_trajectory_geometry_rgb(trajectories(), 320, 180)
    second = render_trajectory_geometry_rgb(trajectories(), 320, 180)

    assert first.shape == (180, 320, 3)
    assert np.array_equal(first, second)
    assert np.unique(first.reshape(-1, 3), axis=0).shape[0] > 3
