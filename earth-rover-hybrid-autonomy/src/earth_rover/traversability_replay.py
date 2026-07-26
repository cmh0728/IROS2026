from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from earth_rover.control.command_filter import CommandFilter
from earth_rover.control.hybrid_controller import HybridReactiveController
from earth_rover.core.types import (
    CandidateDirection,
    ControlCommand,
    LocalPlan,
    LocalTraversability,
    PerceptionResult,
)
from earth_rover.perception.traversability_adapter import TraversabilityAdapter
from earth_rover.planning.goal_aware_local_planner import GoalAwareLocalPlanner
from earth_rover.safety.emergency_stop import EmergencyStopMonitor


@dataclass(frozen=True)
class ReplayObservation:
    source_timestamp: float
    frame_timestamp: float
    sensor_timestamp: float
    source_mask: np.ndarray
    confidence: np.ndarray | None
    inference_latency_ms: float
    heading_error_deg: float
    goal_input_mode: str
    goal_input_valid: bool
    gps_valid: bool = False
    current_waypoint: str | None = None
    distance_to_waypoint_m: float | None = None
    goal_bearing_deg: float | None = None
    ride_id: str = ""
    frame_id: int = -1


@dataclass(frozen=True)
class ReplayStepResult:
    traversability: LocalTraversability
    plan: LocalPlan
    expected_command: ControlCommand
    record: dict[str, object]


class SensorSource(Protocol):
    def __iter__(self) -> Iterable[ReplayObservation]: ...


class ControlSink(Protocol):
    def write(self, result: ReplayStepResult) -> None: ...

    def close(self) -> None: ...


class LogOnlyControlSink:
    """Persist expected commands without exposing any SDK transmission path."""

    FIELDNAMES = (
        "source_timestamp",
        "frame_timestamp",
        "sensor_timestamp",
        "simulated_latency_sec",
        "inference_latency_ms",
        "goal_input_mode",
        "goal_input_valid",
        "gps_valid",
        "current_waypoint",
        "distance_to_waypoint_m",
        "goal_bearing_deg",
        "heading_error_deg",
        "left_traversability_score",
        "center_traversability_score",
        "right_traversability_score",
        "left_obstacle_ratio",
        "center_obstacle_ratio",
        "right_obstacle_ratio",
        "near_obstacle_ratio",
        "mean_confidence",
        "candidate_scores",
        "selected_direction",
        "planner_reason",
        "safety_state",
        "safety_reason",
        "steering_target",
        "speed_target",
        "expected_linear",
        "expected_angular",
        "command_transmitted",
        "ride_id",
        "frame_id",
    )

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.csv_path = self.output_dir / "replay_steps.csv"
        self.jsonl_path = self.output_dir / "replay_steps.jsonl"
        self._csv_handle = self.csv_path.open("w", newline="", encoding="utf-8")
        self._jsonl_handle = self.jsonl_path.open("w", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_handle, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self.count = 0

    def write(self, result: ReplayStepResult) -> None:
        record = dict(result.record)
        if record.get("command_transmitted") is not False:
            raise ValueError("LogOnlyControlSink requires command_transmitted=false")
        csv_record = dict(record)
        csv_record["candidate_scores"] = json.dumps(
            csv_record["candidate_scores"], sort_keys=True, separators=(",", ":")
        )
        self._writer.writerow(csv_record)
        self._jsonl_handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.count += 1

    def close(self) -> None:
        self._csv_handle.close()
        self._jsonl_handle.close()


class OfflineTraversabilityPipeline:
    """Run adapter, planner, existing safety/controller, and command filter offline."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.adapter = TraversabilityAdapter(config)
        self.planner = GoalAwareLocalPlanner(config)
        self.controller = HybridReactiveController(config)
        self.command_filter = CommandFilter(config)
        self.emergency_monitor = EmergencyStopMonitor(config)
        self.last_command = ControlCommand(0.0, 0.0)
        self.last_direction: str | None = None
        self.last_timestamp: float | None = None

    def process(self, observation: ReplayObservation) -> ReplayStepResult:
        now = float(observation.source_timestamp)
        dt = (
            max(1e-3, now - self.last_timestamp)
            if self.last_timestamp is not None
            else 1.0 / float(self.config.get("project", {}).get("loop_hz", 5.0))
        )
        self.last_timestamp = now
        frame_age = max(0.0, now - observation.frame_timestamp)
        sensor_age = max(0.0, now - observation.sensor_timestamp)
        safety_cfg = self.config.get("safety", {})
        frame_stale = frame_age > float(safety_cfg.get("frame_timeout_sec", 1.0))
        sensor_stale = sensor_age > float(safety_cfg.get("data_timeout_sec", 1.0))

        traversability = self.adapter.adapt(observation.source_mask, observation.confidence)
        finite = all(
            math.isfinite(value)
            for value in (
                traversability.left_score,
                traversability.center_score,
                traversability.right_score,
                traversability.near_obstacle_ratio,
                traversability.mean_confidence,
            )
        )
        emergency, emergency_reason = self.emergency_monitor.update(
            now=now,
            last_frame_time=observation.frame_timestamp,
            last_data_time=observation.sensor_timestamp,
            sdk_failure_count=0,
            loop_delay_sec=dt,
            gps_valid=observation.goal_input_valid,
            perception_valid=finite,
        )
        if emergency_reason == "INVALID_GPS" and observation.goal_input_mode != "recorded_gps":
            emergency_reason = "INVALID_GOAL_INPUT"
        plan = self.planner.select(
            traversability,
            math.radians(observation.heading_error_deg),
            previous_direction=self.last_direction,
            previous_command=self.last_command,
            stale_input=frame_stale or sensor_stale,
            goal_valid=observation.goal_input_valid,
        )
        selected_obstacle = (
            float(getattr(traversability, f"{plan.selected_direction.lower()}_obstacle_ratio"))
            if plan.selected_direction in {"LEFT", "CENTER", "RIGHT"}
            else traversability.near_obstacle_ratio
        )
        perception = PerceptionResult(
            traversability.left_score,
            traversability.center_score,
            traversability.right_score,
            selected_obstacle,
            traversability.mean_confidence,
            {"adapter": asdict(traversability)},
        )
        candidate = CandidateDirection(
            name=plan.selected_direction,
            local_goal_error_rad=plan.steering_target,
            traversability_score=plan.speed_target,
            obstacle_risk=selected_obstacle,
            turning_cost=abs(plan.steering_target) / math.pi,
            score=plan.candidate_scores.get(plan.selected_direction, -1.0),
            debug={"planner_reason": plan.reason},
        )
        raw = self.controller.compute(
            heading_error_rad=math.radians(observation.heading_error_deg),
            candidate=candidate,
            perception=perception,
            emergency_stop=emergency,
            recovery_command=None,
            dt=dt,
        )
        force_stop = emergency or plan.stop_requested
        if force_stop:
            self.command_filter.apply(raw, dt, frame_is_stale=True, data_is_stale=True)
            command = ControlCommand(0.0, 0.0, mode=raw.mode)
        else:
            command = self.command_filter.apply(raw, dt, frame_stale, sensor_stale)
        safety_reason = emergency_reason
        if not safety_reason and plan.stop_requested:
            safety_reason = plan.reason
        if not safety_reason and command.mode in {"STALE_DATA_STOP", "INVALID_COMMAND_STOP"}:
            safety_reason = command.mode
        safety_state = "STOP" if safety_reason or command.mode.endswith("STOP") else "CLEAR"
        if plan.selected_direction in {"LEFT", "CENTER", "RIGHT"}:
            self.last_direction = plan.selected_direction
        self.last_command = command
        record = {
            "source_timestamp": now,
            "frame_timestamp": observation.frame_timestamp,
            "sensor_timestamp": observation.sensor_timestamp,
            "simulated_latency_sec": frame_age,
            "inference_latency_ms": observation.inference_latency_ms,
            "goal_input_mode": observation.goal_input_mode,
            "goal_input_valid": observation.goal_input_valid,
            "gps_valid": observation.gps_valid,
            "current_waypoint": observation.current_waypoint,
            "distance_to_waypoint_m": observation.distance_to_waypoint_m,
            "goal_bearing_deg": observation.goal_bearing_deg,
            "heading_error_deg": observation.heading_error_deg,
            "left_traversability_score": traversability.left_score,
            "center_traversability_score": traversability.center_score,
            "right_traversability_score": traversability.right_score,
            "left_obstacle_ratio": traversability.left_obstacle_ratio,
            "center_obstacle_ratio": traversability.center_obstacle_ratio,
            "right_obstacle_ratio": traversability.right_obstacle_ratio,
            "near_obstacle_ratio": traversability.near_obstacle_ratio,
            "mean_confidence": traversability.mean_confidence,
            "candidate_scores": plan.candidate_scores,
            "selected_direction": plan.selected_direction,
            "planner_reason": plan.reason,
            "safety_state": safety_state,
            "safety_reason": safety_reason,
            "steering_target": plan.steering_target,
            "speed_target": plan.speed_target,
            "expected_linear": command.linear,
            "expected_angular": command.angular,
            "command_transmitted": False,
            "ride_id": observation.ride_id,
            "frame_id": observation.frame_id,
        }
        return ReplayStepResult(traversability, plan, command, record)


def run_offline_replay(
    source: SensorSource,
    pipeline: OfflineTraversabilityPipeline,
    sink: ControlSink,
) -> int:
    count = 0
    try:
        for observation in source:
            sink.write(pipeline.process(observation))
            count += 1
    finally:
        sink.close()
    return count
