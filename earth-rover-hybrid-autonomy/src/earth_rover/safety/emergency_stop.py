from __future__ import annotations

from typing import Optional


class EmergencyStopMonitor:
    def __init__(self, config: dict):
        safety_cfg = config.get("safety", {})
        self.enabled = bool(safety_cfg.get("enable_emergency_stop", True))
        self.frame_timeout_sec = float(safety_cfg.get("frame_timeout_sec", 1.0))
        self.data_timeout_sec = float(safety_cfg.get("data_timeout_sec", 1.0))
        self.command_loop_delay_limit_sec = float(safety_cfg.get("command_loop_delay_limit_sec", 0.5))
        self.sdk_failure_limit = int(safety_cfg.get("sdk_failure_limit", 3))

    def update(
        self,
        now: float,
        last_frame_time: Optional[float],
        last_data_time: Optional[float],
        sdk_failure_count: int,
        loop_delay_sec: float,
        gps_valid: bool,
        perception_valid: bool,
    ) -> tuple[bool, str]:
        if not self.enabled:
            return False, ""
        if sdk_failure_count >= self.sdk_failure_limit:
            return True, "SDK_FAILURE_LIMIT"
        if last_frame_time is None or now - last_frame_time > self.frame_timeout_sec:
            return True, "FRAME_TIMEOUT"
        if last_data_time is None or now - last_data_time > self.data_timeout_sec:
            return True, "DATA_TIMEOUT"
        if loop_delay_sec > self.command_loop_delay_limit_sec:
            return True, "LOOP_DELAY"
        if not gps_valid:
            return True, "INVALID_GPS"
        if not perception_valid:
            return True, "INVALID_PERCEPTION"
        return False, ""

