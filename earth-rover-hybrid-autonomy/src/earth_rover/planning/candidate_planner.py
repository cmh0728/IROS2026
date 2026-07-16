from __future__ import annotations

import math

from earth_rover.core.types import CandidateDirection, PerceptionResult


class CandidateDirectionPlanner:
    def __init__(self, config: dict):
        urban_cfg = config.get("urban", {})
        perception_cfg = config.get("perception", {})
        self.w_goal = float(urban_cfg.get("goal_alignment_weight", 0.45))
        self.w_trav = float(urban_cfg.get("traversability_weight", 0.40))
        self.w_obs = float(urban_cfg.get("obstacle_weight", 0.60))
        self.w_turn = float(urban_cfg.get("turning_cost_weight", 0.10))
        self.obstacle_stop_threshold = float(perception_cfg.get("obstacle_stop_threshold", 0.75))
        self.traversability_threshold = float(perception_cfg.get("traversability_threshold", 0.45))

    def select(
        self,
        heading_error_rad: float,
        perception: PerceptionResult,
        force_stop: bool = False,
    ) -> CandidateDirection:
        candidates = self._score_candidates(heading_error_rad, perception)
        debug = {"candidates": [candidate.__dict__ for candidate in candidates]}

        if force_stop or perception.obstacle_confidence >= self.obstacle_stop_threshold:
            return CandidateDirection("STOP", 0.0, 0.0, perception.obstacle_confidence, 0.0, -1.0, debug)

        if abs(math.degrees(heading_error_rad)) >= 90.0:
            name = "ROTATE_LEFT" if heading_error_rad > 0 else "ROTATE_RIGHT"
            selected = next(candidate for candidate in candidates if candidate.name == name)
            selected.debug = debug
            return selected

        forward = [
            candidate
            for candidate in candidates
            if candidate.name in {"LEFT", "CENTER", "RIGHT"}
            and candidate.traversability_score >= self.traversability_threshold
        ]
        selectable = forward or [candidate for candidate in candidates if candidate.name != "STOP"]
        selected = max(selectable, key=lambda item: item.score)
        selected.debug = debug
        return selected

    def _score_candidates(self, heading_error_rad: float, perception: PerceptionResult) -> list[CandidateDirection]:
        raw = [
            ("LEFT", math.radians(35.0), perception.left_free_score, 1.0 - perception.left_free_score),
            ("CENTER", 0.0, perception.center_free_score, 1.0 - perception.center_free_score),
            ("RIGHT", math.radians(-35.0), perception.right_free_score, 1.0 - perception.right_free_score),
            ("ROTATE_LEFT", math.radians(90.0), 0.5, perception.obstacle_confidence * 0.4),
            ("ROTATE_RIGHT", math.radians(-90.0), 0.5, perception.obstacle_confidence * 0.4),
            ("STOP", 0.0, 0.0, 0.0),
        ]
        candidates = []
        for name, local_error, traversability, obstacle_risk in raw:
            turning_cost = abs(local_error) / math.pi
            if name == "STOP":
                score = -0.2
            else:
                goal_alignment = math.cos(local_error - heading_error_rad)
                score = (
                    self.w_goal * goal_alignment
                    + self.w_trav * traversability
                    - self.w_obs * obstacle_risk
                    - self.w_turn * turning_cost
                )
            candidates.append(
                CandidateDirection(
                    name=name,
                    local_goal_error_rad=local_error,
                    traversability_score=traversability,
                    obstacle_risk=obstacle_risk,
                    turning_cost=turning_cost,
                    score=score,
                )
            )
        return candidates

