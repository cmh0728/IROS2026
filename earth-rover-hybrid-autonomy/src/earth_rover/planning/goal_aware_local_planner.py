from __future__ import annotations

import math

from earth_rover.core.types import ControlCommand, LocalPlan, LocalTraversability
from earth_rover.navigation.gps_utils import normalize_angle_rad


class GoalAwareLocalPlanner:
    """Select a safe local direction using perception and goal alignment.

    Camera safety gates candidates before GPS preference is considered.
    Hysteresis retains the previous safe direction when a new winner does not
    exceed it by the configured margin. This class returns semantic targets and
    never sends or constructs SDK requests.
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("goal_aware_planner", {})
        offsets = cfg.get("candidate_heading_offsets_deg", {"LEFT": 35.0, "CENTER": 0.0, "RIGHT": -35.0})
        self.offsets = {
            name: math.radians(float(offsets[name])) for name in ("LEFT", "CENTER", "RIGHT")
        }
        self.traversability_weight = float(cfg.get("traversability_weight", 1.0))
        self.gps_alignment_weight = float(cfg.get("gps_alignment_weight", 0.45))
        self.obstacle_weight = float(cfg.get("obstacle_weight", 1.2))
        self.uncertainty_weight = float(cfg.get("uncertainty_weight", 0.35))
        self.direction_change_penalty = float(cfg.get("direction_change_penalty", 0.12))
        self.hysteresis_margin = float(cfg.get("hysteresis_margin", 0.08))
        self.hard_obstacle_ratio = float(cfg.get("hard_obstacle_ratio", 0.55))
        self.low_confidence_slow_threshold = float(cfg.get("low_confidence_slow_threshold", 0.55))
        self.low_confidence_speed_scale = float(cfg.get("low_confidence_speed_scale", 0.4))
        self.minimum_safe_score = float(cfg.get("minimum_safe_score", 0.08))
        self._previous_direction: str | None = None

    def select(
        self,
        traversability: LocalTraversability,
        heading_error_rad: float,
        previous_direction: str | None = None,
        previous_command: ControlCommand | None = None,
        stale_input: bool = False,
        goal_valid: bool = True,
    ) -> LocalPlan:
        if not math.isfinite(heading_error_rad):
            goal_valid = False
        if stale_input:
            return self._stop("STALE_INPUT")
        if not goal_valid:
            return self._stop("INVALID_GOAL_INPUT")
        if traversability.stop_recommended:
            return self._stop(traversability.reason)

        scores: dict[str, float] = {}
        blocked: list[str] = []
        effective_previous = previous_direction or self._previous_direction
        for name in ("LEFT", "CENTER", "RIGHT"):
            obstacle = _field(traversability, name, "obstacle_ratio")
            if obstacle >= self.hard_obstacle_ratio:
                scores[name] = -1.0
                blocked.append(name)
                continue
            local_score = _field(traversability, name, "score")
            alignment_error = normalize_angle_rad(heading_error_rad - self.offsets[name])
            alignment = (math.cos(alignment_error) + 1.0) / 2.0
            uncertainty = 1.0 - traversability.mean_confidence
            change_penalty = (
                self.direction_change_penalty
                if effective_previous in {"LEFT", "CENTER", "RIGHT"} and name != effective_previous
                else 0.0
            )
            scores[name] = (
                self.traversability_weight * local_score
                + self.gps_alignment_weight * alignment
                - self.obstacle_weight * obstacle
                - self.uncertainty_weight * uncertainty
                - change_penalty
            )

        selectable = [name for name in scores if name not in blocked]
        if not selectable:
            return self._stop("ALL_DIRECTIONS_BLOCKED", scores)
        selected = max(selectable, key=lambda name: (scores[name], -abs(self.offsets[name])))
        if (
            effective_previous in selectable
            and selected != effective_previous
            and scores[selected] - scores[effective_previous] < self.hysteresis_margin
        ):
            selected = effective_previous
            hysteresis = True
        else:
            hysteresis = False
        if scores[selected] < self.minimum_safe_score:
            return self._stop("NO_DIRECTION_ABOVE_MINIMUM_SCORE", scores)

        speed_target = max(0.0, min(1.0, _field(traversability, selected, "score")))
        speed_target *= max(0.0, 1.0 - _field(traversability, selected, "obstacle_ratio"))
        mode = "NORMAL"
        reason_parts = [f"SELECTED_{selected}"]
        if traversability.mean_confidence < self.low_confidence_slow_threshold:
            speed_target *= self.low_confidence_speed_scale
            mode = "LOW_CONFIDENCE_SLOW"
            reason_parts.append("LOW_CONFIDENCE")
        if previous_command is not None and abs(previous_command.angular) > 0.5:
            speed_target *= 0.8
            mode = "TURNING_SLOW"
            reason_parts.append("PREVIOUS_TURN")
        if hysteresis:
            reason_parts.append("HYSTERESIS_HOLD")
        if blocked:
            reason_parts.append("BLOCKED_" + "_".join(blocked))
        self._previous_direction = selected
        return LocalPlan(
            selected_direction=selected,
            steering_target=self.offsets[selected],
            speed_target=max(0.0, min(1.0, speed_target)),
            candidate_scores=scores,
            mode=mode,
            stop_requested=False,
            reason=";".join(reason_parts),
        )

    def _stop(self, reason: str, scores: dict[str, float] | None = None) -> LocalPlan:
        return LocalPlan(
            selected_direction="STOP",
            steering_target=0.0,
            speed_target=0.0,
            candidate_scores=scores or {},
            mode="STOP",
            stop_requested=True,
            reason=reason,
        )


def _field(value: LocalTraversability, direction: str, suffix: str) -> float:
    return float(getattr(value, f"{direction.lower()}_{suffix}"))
