from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import yaml

from earth_rover.core.types import CandidateDirection, ControlCommand, FrameData, PerceptionResult, RoverData


class RunLogger:
    def __init__(self, config: dict):
        logging_cfg = config.get("logging", {})
        self.enabled = bool(logging_cfg.get("enabled", True))
        self.save_frames = bool(logging_cfg.get("save_frames", True))
        self.save_debug_images = bool(logging_cfg.get("save_debug_images", True))
        self.step = 0
        self.root = Path(logging_cfg.get("log_dir", "logs")) / time.strftime("run_%Y%m%d_%H%M%S")
        self.frames_dir = self.root / "frames"
        self.rear_frames_dir = self.root / "frames_rear"
        self.debug_dir = self.root / "debug"
        self.recovery_dir = self.root / "recovery"
        self._data_file = None
        self._commands_file = None
        self._timeline_file = None
        self._events_file = None
        self._data_writer = None
        self._commands_writer = None
        self._timeline_writer = None
        self._recovery_count = 0

        if self.enabled:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            self.rear_frames_dir.mkdir(parents=True, exist_ok=True)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.recovery_dir.mkdir(parents=True, exist_ok=True)
            with (self.root / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            self._open_csvs()

    def _open_csvs(self) -> None:
        self._data_file = (self.root / "data.csv").open("w", newline="", encoding="utf-8")
        self._commands_file = (self.root / "commands.csv").open("w", newline="", encoding="utf-8")
        self._timeline_file = (self.root / "timeline.csv").open("w", newline="", encoding="utf-8")
        self._events_file = (self.root / "events.jsonl").open("a", encoding="utf-8")
        self._data_writer = csv.DictWriter(
            self._data_file,
            fieldnames=[
                "timestamp",
                "sdk_timestamp",
                "latitude",
                "longitude",
                "orientation",
                "speed",
                "battery",
                "signal_level",
                "gps_signal",
                "mean_rpm",
            ],
        )
        self._commands_writer = csv.DictWriter(
            self._commands_file,
            fieldnames=[
                "timestamp",
                "command_timestamp",
                "linear",
                "angular",
                "lamp",
                "mode",
                "raw_linear",
                "raw_angular",
                "candidate_name",
                "heading_error_deg",
                "obstacle_confidence",
                "left_free_score",
                "center_free_score",
                "right_free_score",
                "emergency_reason",
                "stuck",
                "recovery_state",
            ],
        )
        self._timeline_writer = csv.DictWriter(
            self._timeline_file,
            fieldnames=[
                "timestamp",
                "local_timestamp",
                "frame_timestamp",
                "frame_sdk_timestamp",
                "data_timestamp",
                "data_sdk_timestamp",
                "command_timestamp",
                "front_frame_path",
                "rear_frame_path",
                "latitude",
                "longitude",
                "orientation",
                "speed",
                "rpms",
                "battery",
                "signal_level",
                "gps_signal",
                "target_checkpoint",
                "distance_to_checkpoint",
                "heading_error",
                "linear_command",
                "angular_command",
                "mode",
                "failure_flag",
            ],
        )
        self._data_writer.writeheader()
        self._commands_writer.writeheader()
        self._timeline_writer.writeheader()

    def log_event(self, event: str, detail: dict[str, Any] | None = None) -> None:
        if not self.enabled or self._events_file is None:
            return
        self._events_file.write(json.dumps({"timestamp": time.time(), "event": event, "detail": detail or {}}) + "\n")
        self._events_file.flush()

    def log_recovery_event(
        self,
        metadata: dict[str, Any],
        front_frame: FrameData | None = None,
        rear_frame: FrameData | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return metadata
        self._recovery_count += 1
        recovery_id = int(metadata.get("recovery_id") or self._recovery_count)
        prefix = f"recovery_{recovery_id:04d}"
        enriched = dict(metadata)
        enriched["recovery_id"] = recovery_id

        front_path = None
        if front_frame is not None:
            front_path = self.recovery_dir / f"{prefix}_front_before.jpg"
            if cv2.imwrite(str(front_path), front_frame.image):
                enriched["front_frame_path"] = self._relative_log_path(front_path)

        rear_path = None
        if rear_frame is not None:
            rear_path = self.recovery_dir / f"{prefix}_rear_before.jpg"
            if cv2.imwrite(str(rear_path), rear_frame.image):
                enriched["rear_frame_path"] = self._relative_log_path(rear_path)
        elif enriched.get("recovery_action") == "reverse_then_rotate":
            enriched["warning"] = "rear_frame_unavailable_before_reverse"

        meta_path = self.recovery_dir / f"{prefix}_meta.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(enriched, handle, indent=2, ensure_ascii=False)
        self.log_event("RECOVERY_EVENT", enriched)
        return enriched

    def log_step(
        self,
        frame: FrameData,
        data: RoverData,
        perception: PerceptionResult,
        candidate: CandidateDirection,
        raw_command: ControlCommand,
        command: ControlCommand,
        rear_frame: FrameData | None = None,
        extra_debug: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.step += 1
        extra_debug = extra_debug or {}
        front_frame_path = ""
        rear_frame_path = ""
        if self.save_frames:
            front_path = self.frames_dir / f"front_{self.step:06d}.jpg"
            if cv2.imwrite(str(front_path), frame.image):
                front_frame_path = self._relative_log_path(front_path)
            if rear_frame is not None:
                rear_path = self.rear_frames_dir / f"rear_{self.step:06d}.jpg"
                if cv2.imwrite(str(rear_path), rear_frame.image):
                    rear_frame_path = self._relative_log_path(rear_path)
        if self.save_debug_images:
            cv2.imwrite(str(self.debug_dir / f"debug_{self.step:06d}.jpg"), self._overlay(frame, perception, candidate, command))

        command_timestamp = time.time()
        mean_rpm = None
        if data.rpms:
            mean_rpm = sum(abs(value) for value in data.rpms) / len(data.rpms)
        self._data_writer.writerow(
            {
                "timestamp": data.timestamp,
                "sdk_timestamp": data.sdk_timestamp,
                "latitude": data.latitude,
                "longitude": data.longitude,
                "orientation": data.orientation,
                "speed": data.speed,
                "battery": data.battery,
                "signal_level": data.signal_level,
                "gps_signal": data.gps_signal,
                "mean_rpm": mean_rpm,
            }
        )
        self._commands_writer.writerow(
            {
                "timestamp": command_timestamp,
                "command_timestamp": command_timestamp,
                "linear": command.linear,
                "angular": command.angular,
                "lamp": command.lamp,
                "mode": command.mode,
                "raw_linear": raw_command.linear,
                "raw_angular": raw_command.angular,
                "candidate_name": candidate.name,
                "heading_error_deg": extra_debug.get("heading_error_deg"),
                "obstacle_confidence": perception.obstacle_confidence,
                "left_free_score": perception.left_free_score,
                "center_free_score": perception.center_free_score,
                "right_free_score": perception.right_free_score,
                "emergency_reason": extra_debug.get("emergency_reason", ""),
                "stuck": extra_debug.get("stuck", False),
                "recovery_state": extra_debug.get("recovery_state", ""),
            }
        )
        self._timeline_writer.writerow(
            {
                "timestamp": command_timestamp,
                "local_timestamp": command_timestamp,
                "frame_timestamp": frame.timestamp,
                "frame_sdk_timestamp": frame.sdk_timestamp,
                "data_timestamp": data.timestamp,
                "data_sdk_timestamp": data.sdk_timestamp,
                "command_timestamp": command_timestamp,
                "front_frame_path": front_frame_path,
                "rear_frame_path": rear_frame_path,
                "latitude": data.latitude,
                "longitude": data.longitude,
                "orientation": data.orientation,
                "speed": data.speed,
                "rpms": json.dumps(data.rpms or []),
                "battery": data.battery,
                "signal_level": data.signal_level,
                "gps_signal": data.gps_signal,
                "target_checkpoint": json.dumps(extra_debug.get("target_checkpoint")),
                "distance_to_checkpoint": extra_debug.get("distance_to_checkpoint"),
                "heading_error": extra_debug.get("heading_error_deg"),
                "linear_command": command.linear,
                "angular_command": command.angular,
                "mode": command.mode,
                "failure_flag": self._failure_flag(command, extra_debug),
            }
        )
        self._data_file.flush()
        self._commands_file.flush()
        self._timeline_file.flush()

    @staticmethod
    def _overlay(
        frame: FrameData,
        perception: PerceptionResult,
        candidate: CandidateDirection,
        command: ControlCommand,
    ):
        image = frame.image.copy()
        lines = [
            f"L/C/R {perception.left_free_score:.2f} {perception.center_free_score:.2f} {perception.right_free_score:.2f}",
            f"cand {candidate.name} mode {command.mode}",
            f"cmd lin {command.linear:.2f} ang {command.angular:.2f}",
            f"obs {perception.obstacle_confidence:.2f}",
        ]
        y = 24
        for line in lines:
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
        return image

    def close(self) -> None:
        for handle in [self._data_file, self._commands_file, self._timeline_file, self._events_file]:
            if handle is not None:
                handle.close()

    def _relative_log_path(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    @staticmethod
    def _failure_flag(command: ControlCommand, extra_debug: dict[str, Any]) -> bool:
        if extra_debug.get("stuck", False):
            return True
        if extra_debug.get("emergency_reason"):
            return True
        return command.mode not in {"NORMAL_DRIVE", "ROTATE_IN_PLACE", "MISSION_COMPLETE"}
