from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RoverData:
    timestamp: float
    latitude: Optional[float]
    longitude: Optional[float]
    orientation: Optional[float]
    speed: Optional[float]
    rpms: Optional[list[float]]
    battery: Optional[float]
    signal_level: Optional[float]
    gps_signal: Optional[float]
    raw: dict
    sdk_timestamp: Optional[float] = None


@dataclass
class FrameData:
    timestamp: float
    image: np.ndarray
    source: str
    sdk_timestamp: Optional[float] = None


@dataclass
class ControlCommand:
    linear: float
    angular: float
    lamp: int = 0
    mode: str = "NORMAL_DRIVE"


@dataclass
class PerceptionResult:
    left_free_score: float
    center_free_score: float
    right_free_score: float
    obstacle_confidence: float
    traversability_confidence: float
    debug: dict = field(default_factory=dict)


@dataclass
class CandidateDirection:
    name: str
    local_goal_error_rad: float
    traversability_score: float
    obstacle_risk: float
    turning_cost: float
    score: float
    debug: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LocalTraversability:
    """Planner-facing summary of a source-ID segmentation mask.

    Scores are normalized to ``[0, 1]``. Obstacle ratios describe each
    direction's configured driving ROI. ``near_obstacle_ratio`` describes the
    lower center ROI, where an obstacle most directly blocks forward motion.
    ``free_space_center`` is the center score before goal preference is added.
    The recommendation is perception-only and never represents mission intent.
    """

    left_score: float
    center_score: float
    right_score: float
    left_obstacle_ratio: float
    center_obstacle_ratio: float
    right_obstacle_ratio: float
    near_obstacle_ratio: float
    mean_confidence: float
    free_space_center: float
    recommended_direction: str
    recommended_heading: float
    stop_recommended: bool
    reason: str
    sector_debug: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LocalPlan:
    """Semantic local target selected before controller command conversion.

    ``steering_target`` is a local heading offset in radians and
    ``speed_target`` is a normalized ``[0, 1]`` speed scale. The existing
    controller converts these values into bounded rover commands.
    """

    selected_direction: str
    steering_target: float
    speed_target: float
    candidate_scores: dict[str, float]
    mode: str
    stop_requested: bool
    reason: str
