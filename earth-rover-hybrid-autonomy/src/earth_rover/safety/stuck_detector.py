from __future__ import annotations

import statistics
from typing import Optional

from earth_rover.core.types import ControlCommand, RoverData
from earth_rover.navigation.gps_utils import haversine_distance_m


class StuckDetector:
    def __init__(self, config: dict):
        stuck_cfg = config.get("stuck", {})
        self.enabled = bool(stuck_cfg.get("enabled", True))
        self.stuck_time_sec = float(stuck_cfg.get("stuck_time_sec", 3.0))
        self.min_speed_for_not_stuck = float(stuck_cfg.get("min_speed_for_not_stuck", 0.03))
        self.min_position_delta_m = float(stuck_cfg.get("min_position_delta_m", 0.15))
        self.low_rpm_threshold = float(stuck_cfg.get("low_rpm_threshold", 1.0))
        self._candidate_since: Optional[float] = None
        self._start_position: Optional[tuple[float, float]] = None

    def update(self, rover_data: RoverData, last_command: ControlCommand) -> tuple[bool, dict]:
        if not self.enabled:
            return False, {"enabled": False}

        now = rover_data.timestamp
        forward_commanded = last_command.linear > 0.1
        speed_low = rover_data.speed is not None and abs(rover_data.speed) < self.min_speed_for_not_stuck
        rpm_low = self._rpm_is_low(rover_data.rpms)
        candidate = forward_commanded and speed_low and rpm_low

        if not candidate:
            self._candidate_since = None
            self._start_position = self._position(rover_data)
            return False, {
                "forward_commanded": forward_commanded,
                "speed_low": speed_low,
                "rpm_low": rpm_low,
                "duration_sec": 0.0,
            }

        if self._candidate_since is None:
            self._candidate_since = now
            self._start_position = self._position(rover_data)

        duration = now - self._candidate_since
        position_delta = self._position_delta(rover_data)
        stuck = duration >= self.stuck_time_sec
        if position_delta is not None and position_delta >= self.min_position_delta_m:
            stuck = False
            self._candidate_since = now
            self._start_position = self._position(rover_data)

        return stuck, {
            "forward_commanded": forward_commanded,
            "speed_low": speed_low,
            "rpm_low": rpm_low,
            "duration_sec": duration,
            "position_delta_m": position_delta,
        }

    def _rpm_is_low(self, rpms: list[float] | None) -> bool:
        if not rpms:
            return True
        return statistics.fmean(abs(value) for value in rpms) < self.low_rpm_threshold

    @staticmethod
    def _position(rover_data: RoverData) -> Optional[tuple[float, float]]:
        if rover_data.latitude is None or rover_data.longitude is None:
            return None
        return rover_data.latitude, rover_data.longitude

    def _position_delta(self, rover_data: RoverData) -> Optional[float]:
        if self._start_position is None:
            return None
        current = self._position(rover_data)
        if current is None:
            return None
        return haversine_distance_m(self._start_position[0], self._start_position[1], current[0], current[1])

