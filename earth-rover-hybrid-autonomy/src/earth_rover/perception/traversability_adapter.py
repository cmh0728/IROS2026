from __future__ import annotations

import math

import numpy as np

from earth_rover.core.types import LocalTraversability


SOURCE_IGNORE = 0
SOURCE_ON_ROAD = 1
SOURCE_OFF_ROAD = 2
SOURCE_OBSTACLE = 3
VALID_SOURCE_IDS = {SOURCE_IGNORE, SOURCE_ON_ROAD, SOURCE_OFF_ROAD, SOURCE_OBSTACLE}


class TraversabilityAdapter:
    """Reduce a segmentation mask to configurable driving-sector scores.

    For each sector, near and far pixels are weighted separately. ON_ROAD and
    OFF_ROAD contribute configurable rewards, while OBSTACLE and uncertainty
    subtract penalties. IGNORE pixels and pixels below the confidence threshold
    are uncertain. A missing confidence map uses the configured conservative
    fallback for every pixel and is called out in ``reason``.
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("traversability_adapter", {})
        self.roi_top = _unit(cfg.get("roi_top", 0.35), "roi_top")
        self.near_top = _unit(cfg.get("near_top", 0.72), "near_top")
        if self.near_top <= self.roi_top:
            raise ValueError("near_top must be greater than roi_top")
        boundaries = tuple(float(value) for value in cfg.get("sector_boundaries", [0.0, 0.34, 0.66, 1.0]))
        if len(boundaries) != 4 or boundaries[0] != 0.0 or boundaries[-1] != 1.0:
            raise ValueError("sector_boundaries must contain [0, left, right, 1]")
        if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
            raise ValueError("sector_boundaries must be strictly increasing")
        self.boundaries = boundaries
        self.near_weight = _positive(cfg.get("near_weight", 2.0), "near_weight")
        self.on_road_reward = _unit(cfg.get("on_road_reward", 1.0), "on_road_reward")
        self.off_road_reward = _unit(cfg.get("off_road_reward", 0.65), "off_road_reward")
        self.obstacle_penalty = _positive(cfg.get("obstacle_penalty", 1.0), "obstacle_penalty")
        self.uncertainty_penalty = _positive(
            cfg.get("uncertainty_penalty", 0.55), "uncertainty_penalty"
        )
        self.low_confidence_threshold = _unit(
            cfg.get("low_confidence_threshold", 0.55), "low_confidence_threshold"
        )
        self.missing_confidence = _unit(
            cfg.get("missing_confidence", 0.5), "missing_confidence"
        )
        self.near_obstacle_stop_ratio = _unit(
            cfg.get("near_obstacle_stop_ratio", 0.45), "near_obstacle_stop_ratio"
        )
        self.confidence_stop_threshold = _unit(
            cfg.get("confidence_stop_threshold", 0.25), "confidence_stop_threshold"
        )
        offsets = cfg.get("candidate_heading_offsets_deg", {"LEFT": 35.0, "CENTER": 0.0, "RIGHT": -35.0})
        self.heading_offsets = {
            name: math.radians(float(offsets[name])) for name in ("LEFT", "CENTER", "RIGHT")
        }

    def adapt(
        self,
        source_mask: np.ndarray,
        confidence: np.ndarray | None = None,
    ) -> LocalTraversability:
        """Return normalized LEFT/CENTER/RIGHT safety evidence.

        The mask must use source IDs ``0 IGNORE, 1 ON_ROAD, 2 OFF_ROAD,
        3 OBSTACLE``. Confidence, when provided, must match the mask and be
        finite in ``[0, 1]``.
        """

        mask = np.asarray(source_mask)
        if mask.ndim != 2 or mask.size == 0:
            raise ValueError("source_mask must be a non-empty 2D array")
        values = {int(value) for value in np.unique(mask)}
        if not values.issubset(VALID_SOURCE_IDS):
            raise ValueError(f"source_mask contains unsupported IDs: {sorted(values)}")
        if confidence is None:
            confidence_array = np.full(mask.shape, self.missing_confidence, dtype=np.float32)
            confidence_missing = True
        else:
            confidence_array = np.asarray(confidence, dtype=np.float32)
            if confidence_array.shape != mask.shape:
                raise ValueError("confidence shape must match source_mask")
            if not np.isfinite(confidence_array).all():
                raise ValueError("confidence must contain only finite values")
            if confidence_array.min() < 0.0 or confidence_array.max() > 1.0:
                raise ValueError("confidence values must be within [0, 1]")
            confidence_missing = False

        height, width = mask.shape
        roi_start = min(height - 1, int(round(height * self.roi_top)))
        near_start = min(height - 1, max(roi_start + 1, int(round(height * self.near_top))))
        roi_mask = mask[roi_start:]
        roi_confidence = confidence_array[roi_start:]
        row_weights = np.ones((height - roi_start, 1), dtype=np.float32)
        row_weights[near_start - roi_start :] = self.near_weight
        sectors: dict[str, dict[str, float]] = {}
        names = ("LEFT", "CENTER", "RIGHT")
        for index, name in enumerate(names):
            left = int(round(width * self.boundaries[index]))
            right = int(round(width * self.boundaries[index + 1]))
            if right <= left:
                raise ValueError("sector boundaries produce an empty sector")
            sectors[name] = self._score_sector(
                roi_mask[:, left:right],
                roi_confidence[:, left:right],
                np.broadcast_to(row_weights, (height - roi_start, right - left)),
            )

        center_left = int(round(width * self.boundaries[1]))
        center_right = int(round(width * self.boundaries[2]))
        near_center = mask[near_start:, center_left:center_right]
        near_obstacle_ratio = float(np.mean(near_center == SOURCE_OBSTACLE))
        mean_confidence = float(np.mean(roi_confidence))
        scores = {name: sectors[name]["score"] for name in names}
        recommended = max(names, key=lambda name: (scores[name], -abs(self.heading_offsets[name])))
        stop_recommended = (
            near_obstacle_ratio >= self.near_obstacle_stop_ratio
            or mean_confidence < self.confidence_stop_threshold
        )
        reasons = []
        if near_obstacle_ratio >= self.near_obstacle_stop_ratio:
            reasons.append("NEAR_CENTER_OBSTACLE")
        if mean_confidence < self.confidence_stop_threshold:
            reasons.append("LOW_CONFIDENCE_STOP")
        elif mean_confidence < self.low_confidence_threshold:
            reasons.append("LOW_CONFIDENCE")
        if confidence_missing:
            reasons.append("CONFIDENCE_UNAVAILABLE")
        if not reasons:
            reasons.append(f"BEST_{recommended}")
        return LocalTraversability(
            left_score=scores["LEFT"],
            center_score=scores["CENTER"],
            right_score=scores["RIGHT"],
            left_obstacle_ratio=sectors["LEFT"]["obstacle_ratio"],
            center_obstacle_ratio=sectors["CENTER"]["obstacle_ratio"],
            right_obstacle_ratio=sectors["RIGHT"]["obstacle_ratio"],
            near_obstacle_ratio=near_obstacle_ratio,
            mean_confidence=mean_confidence,
            free_space_center=scores["CENTER"],
            recommended_direction="STOP" if stop_recommended else recommended,
            recommended_heading=0.0 if stop_recommended else self.heading_offsets[recommended],
            stop_recommended=stop_recommended,
            reason=";".join(reasons),
            sector_debug={
                "roi_top": self.roi_top,
                "near_top": self.near_top,
                "sector_boundaries": self.boundaries,
                "sectors": sectors,
            },
        )

    def _score_sector(
        self,
        mask: np.ndarray,
        confidence: np.ndarray,
        weights: np.ndarray,
    ) -> dict[str, float]:
        weight_total = float(weights.sum())
        reliable = confidence >= self.low_confidence_threshold
        on_reward = weights * (mask == SOURCE_ON_ROAD) * confidence
        off_reward = weights * (mask == SOURCE_OFF_ROAD) * confidence
        obstacle = mask == SOURCE_OBSTACLE
        uncertain = (mask == SOURCE_IGNORE) | ~reliable
        traversability = float(
            (self.on_road_reward * on_reward.sum() + self.off_road_reward * off_reward.sum())
            / weight_total
        )
        obstacle_ratio = float((weights * obstacle).sum() / weight_total)
        uncertainty_ratio = float((weights * uncertain).sum() / weight_total)
        score = np.clip(
            traversability
            - self.obstacle_penalty * obstacle_ratio
            - self.uncertainty_penalty * uncertainty_ratio,
            0.0,
            1.0,
        )
        return {
            "score": float(score),
            "traversability_reward": traversability,
            "obstacle_ratio": obstacle_ratio,
            "uncertainty_ratio": uncertainty_ratio,
            "mean_confidence": float(np.mean(confidence)),
        }


def _unit(value: object, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return number


def _positive(value: object, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number
