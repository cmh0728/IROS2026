from __future__ import annotations

from earth_rover.core.types import PerceptionResult


class RuleBasedObstacleDetector:
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def is_blocked(self, perception: PerceptionResult) -> bool:
        return perception.obstacle_confidence >= self.threshold

