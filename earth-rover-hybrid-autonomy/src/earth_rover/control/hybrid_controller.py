from __future__ import annotations

import math
from typing import Optional

from earth_rover.core.types import CandidateDirection, ControlCommand, PerceptionResult


class HybridReactiveController:
    def __init__(self, config: dict):
        control_cfg = config.get("control", {})
        perception_cfg = config.get("perception", {})
        self.base_linear = float(control_cfg.get("base_linear", 0.22))
        self.heading_kp = float(control_cfg.get("heading_kp", 0.70))
        self.heading_kd = float(control_cfg.get("heading_kd", 0.05))
        self.local_goal_kp = float(control_cfg.get("local_goal_kp", 0.35))
        self.slow_heading_error_deg = float(control_cfg.get("slow_heading_error_deg", 35.0))
        self.rotate_in_place_error_deg = float(control_cfg.get("rotate_in_place_error_deg", 90.0))
        self.rotate_angular = float(control_cfg.get("rotate_angular", 0.55))
        self.obstacle_stop_threshold = float(perception_cfg.get("obstacle_stop_threshold", 0.75))
        self._last_heading_error_rad: float | None = None

    def compute(
        self,
        heading_error_rad: float,
        candidate: CandidateDirection,
        perception: PerceptionResult,
        emergency_stop: bool,
        recovery_command: Optional[ControlCommand],
        dt: float,
    ) -> ControlCommand:
        if emergency_stop:
            return ControlCommand(0.0, 0.0, mode="EMERGENCY_STOP")

        if recovery_command is not None:
            return recovery_command

        if perception.obstacle_confidence > self.obstacle_stop_threshold or candidate.name == "STOP":
            return ControlCommand(0.0, 0.0, mode="OBSTACLE_STOP")

        heading_error_deg = math.degrees(heading_error_rad)
        if abs(heading_error_deg) > self.rotate_in_place_error_deg or candidate.name.startswith("ROTATE"):
            direction = 1.0 if heading_error_rad >= 0 else -1.0
            if candidate.name == "ROTATE_LEFT":
                direction = 1.0
            elif candidate.name == "ROTATE_RIGHT":
                direction = -1.0
            return ControlCommand(0.0, direction * self.rotate_angular, mode="ROTATE_IN_PLACE")

        derivative = 0.0
        if self._last_heading_error_rad is not None and dt > 1e-6:
            derivative = (heading_error_rad - self._last_heading_error_rad) / dt
        self._last_heading_error_rad = heading_error_rad

        heading_confidence = max(0.0, 1.0 - abs(heading_error_deg) / 90.0)
        traversability_score = max(0.0, candidate.traversability_score)
        linear = self.base_linear * traversability_score * heading_confidence
        mode = "NORMAL_DRIVE"
        if abs(heading_error_deg) > self.slow_heading_error_deg:
            linear *= 0.5
            mode = "SLOW_APPROACH"

        angular = (
            self.heading_kp * heading_error_rad
            + self.heading_kd * derivative
            + self.local_goal_kp * candidate.local_goal_error_rad
        )
        return ControlCommand(linear, angular, mode=mode)

