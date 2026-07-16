from __future__ import annotations

from earth_rover.core.types import ControlCommand


class RecoveryController:
    def __init__(self, config: dict):
        recovery_cfg = config.get("recovery", {})
        self.enabled = bool(recovery_cfg.get("enabled", True))
        self.stop_duration_sec = float(recovery_cfg.get("stop_duration_sec", 0.5))
        self.reverse_duration_sec = float(recovery_cfg.get("reverse_duration_sec", 1.0))
        self.rotate_duration_sec = float(recovery_cfg.get("rotate_duration_sec", 1.2))
        self.reverse_linear = float(recovery_cfg.get("reverse_linear", -0.18))
        self.rotate_angular = float(recovery_cfg.get("rotate_angular", 0.55))
        self.max_attempts = int(recovery_cfg.get("max_attempts", 3))
        self.state = "IDLE"
        self.attempts = 0
        self._state_started_at: float | None = None
        self._rotate_sign = 1.0

    def trigger(self):
        if not self.enabled:
            return
        if self.state not in {"IDLE", "RETRY"}:
            return
        self.attempts += 1
        if self.attempts > self.max_attempts:
            self.state = "FAILED"
            return
        self._rotate_sign *= -1.0
        self.state = "STOP"
        self._state_started_at = None

    def update(self, now: float) -> ControlCommand | None:
        if not self.enabled or self.state == "IDLE":
            return None
        if self.state == "FAILED":
            return ControlCommand(0.0, 0.0, mode="RECOVERY_FAILED")

        if self._state_started_at is None:
            self._state_started_at = now
        elapsed = now - self._state_started_at

        if self.state == "STOP":
            if elapsed >= self.stop_duration_sec:
                self._transition("REVERSE", now)
                return self.update(now)
            return ControlCommand(0.0, 0.0, mode="RECOVERY_STOP")

        if self.state == "REVERSE":
            if elapsed >= self.reverse_duration_sec:
                self._transition("ROTATE", now)
                return self.update(now)
            return ControlCommand(self.reverse_linear, 0.0, mode="RECOVERY_REVERSE")

        if self.state == "ROTATE":
            if elapsed >= self.rotate_duration_sec:
                self._transition("RETRY", now)
                return None
            return ControlCommand(0.0, self._rotate_sign * self.rotate_angular, mode="RECOVERY_ROTATE")

        if self.state == "RETRY":
            self.state = "IDLE"
            self._state_started_at = None
            return None
        return None

    def reset(self):
        self.state = "IDLE"
        self.attempts = 0
        self._state_started_at = None

    def _transition(self, state: str, now: float) -> None:
        self.state = state
        self._state_started_at = now

