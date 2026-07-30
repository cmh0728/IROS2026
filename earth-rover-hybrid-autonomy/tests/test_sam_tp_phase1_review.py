from __future__ import annotations

import numpy as np

from earth_rover.planning.trajectory_sampler import (
    DEFAULT_CURVATURES,
    ConstantCurvatureTrajectorySampler,
)
from training.sam_tp_phase1_review import (
    SamTpPhase1FrameProcessor,
    draw_image_path_rgb,
    propose_image_space_path,
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
    assert result.image_path.valid
    assert np.all(result.traversability.score_map[
        result.image_path.points_uv[:, 1],
        result.image_path.points_uv[:, 0],
    ] >= 0.55)


def test_geometry_panel_is_deterministic_and_not_blank() -> None:
    first = render_trajectory_geometry_rgb(trajectories(), 320, 180)
    second = render_trajectory_geometry_rgb(trajectories(), 320, 180)

    assert first.shape == (180, 320, 3)
    assert np.array_equal(first, second)
    assert np.unique(first.reshape(-1, 3), axis=0).shape[0] > 3


def test_image_path_stays_inside_connected_high_score_region() -> None:
    score = np.full((120, 200), 0.1, dtype=np.float32)
    for y in range(35, 115):
        center = 100 + (80 - y) // 3
        score[y, center - 24 : center + 25] = 0.9

    proposal = propose_image_space_path(
        score,
        np.ones_like(score, dtype=bool),
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
    )

    assert proposal.valid
    assert proposal.reason == "CONNECTED_HIGH_TRAVERSABILITY_IMAGE_PATH"
    assert np.all(score[proposal.points_uv[:, 1], proposal.points_uv[:, 0]] >= 0.55)
    rendered = draw_image_path_rgb(
        np.zeros((120, 200, 3), dtype=np.uint8),
        proposal,
        0.02,
    )
    assert rendered.any()


def test_image_path_is_rejected_when_safe_region_is_disconnected() -> None:
    score = np.full((120, 200), 0.9, dtype=np.float32)
    score[65:72] = 0.0

    proposal = propose_image_space_path(
        score,
        np.ones_like(score, dtype=bool),
        minimum_score=0.55,
        corridor_half_width_ratio=0.02,
    )

    assert not proposal.valid
    assert proposal.reason == "NO_CONNECTED_TRAVERSABLE_PATH"
    assert proposal.points_uv.shape == (0, 2)
