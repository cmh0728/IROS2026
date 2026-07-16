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
