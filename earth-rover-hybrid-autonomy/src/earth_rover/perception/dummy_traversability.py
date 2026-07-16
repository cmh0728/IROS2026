from __future__ import annotations

import cv2
import numpy as np

from earth_rover.core.types import FrameData, PerceptionResult
from earth_rover.utils.image import resize_for_model
from earth_rover.utils.math_utils import clamp


class DummyTraversabilityModel:
    def __init__(self, config: dict):
        perception_cfg = config.get("perception", {})
        self.width = int(perception_cfg.get("input_width", 320))
        self.height = int(perception_cfg.get("input_height", 240))

    def infer(self, frame: FrameData) -> PerceptionResult:
        image = resize_for_model(frame.image, self.width, self.height)
        thirds = np.array_split(image, 3, axis=1)
        scores = [self._region_free_score(region) for region in thirds]

        # Bias center slightly for MVP route following unless evidence says otherwise.
        left, center, right = scores
        center = clamp(center + 0.08, 0.0, 1.0)
        obstacle_confidence = clamp(1.0 - min(left, center, right), 0.0, 1.0)
        traversability_confidence = clamp(float(np.mean([left, center, right])), 0.0, 1.0)

        return PerceptionResult(
            left_free_score=left,
            center_free_score=center,
            right_free_score=right,
            obstacle_confidence=obstacle_confidence,
            traversability_confidence=traversability_confidence,
            debug={
                "region_scores": {"left": left, "center": center, "right": right},
                "source": frame.source,
            },
        )

    @staticmethod
    def _region_free_score(region: np.ndarray) -> float:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0
        contrast = float(np.std(gray)) / 128.0
        edges = cv2.Canny(gray, 60, 140)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        darkness_penalty = max(0.0, 0.35 - brightness) / 0.35
        edge_penalty = clamp(edge_density / 0.18, 0.0, 1.0)
        contrast_penalty = clamp(max(0.0, contrast - 0.65), 0.0, 1.0)
        free = 1.0 - (0.45 * darkness_penalty + 0.40 * edge_penalty + 0.15 * contrast_penalty)
        return clamp(free, 0.0, 1.0)

