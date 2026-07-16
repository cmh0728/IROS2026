from __future__ import annotations


class LatencyCompensator:
    def __init__(self, config: dict):
        latency_cfg = config.get("latency", {})
        self.enabled = bool(latency_cfg.get("enabled", True))
        self.assumed_latency_sec = float(latency_cfg.get("assumed_latency_sec", 0.5))
        self.angular_scale = float(latency_cfg.get("angular_scale_deg_per_sec_at_full_cmd", 60.0))
        self.disable_when_stuck = bool(latency_cfg.get("disable_when_stuck", True))
        self.disable_when_signal_low = bool(latency_cfg.get("disable_when_signal_low", True))

    def predict_heading_deg(
        self,
        current_heading_deg: float,
        last_angular_cmd: float,
        is_stuck: bool,
        signal_low: bool,
    ) -> float:
        if not self.enabled:
            return current_heading_deg
        if is_stuck and self.disable_when_stuck:
            return current_heading_deg
        if signal_low and self.disable_when_signal_low:
            return current_heading_deg
        predicted = current_heading_deg + last_angular_cmd * self.angular_scale * self.assumed_latency_sec
        return predicted % 360.0

