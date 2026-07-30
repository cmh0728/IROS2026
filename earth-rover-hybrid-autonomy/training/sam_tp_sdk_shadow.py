from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from earth_rover.core.types import FrameData, RoverData
from earth_rover.planning.trajectory_sampler import (
    DEFAULT_CURVATURES,
    ConstantCurvatureTrajectorySampler,
)
from training.sam_tp_phase1_review import (
    SamTpPhase1FrameProcessor,
    draw_image_path_rgb,
)
from training.sam_tp_reproduction import SamTpPrediction, score_to_heatmap


class ReadOnlySdkSource(Protocol):
    def get_front_frame(self) -> FrameData: ...

    def get_data(self) -> RoverData: ...


class Predictor(Protocol):
    def predict(self, image_rgb: np.ndarray) -> SamTpPrediction: ...


_PROVISIONAL_PHASE1_TRAJECTORIES = ConstantCurvatureTrajectorySampler(
    DEFAULT_CURVATURES,
    horizon_m=2.0,
    sample_interval_m=0.1,
    rover_width_m=0.4,
    safety_margin_m=0.1,
).sample()


@dataclass(frozen=True)
class ShadowStep:
    dashboard_bgr: np.ndarray
    record: dict[str, object]


def run_shadow_step(
    sdk: ReadOnlySdkSource,
    predictor: Predictor,
    frame_index: int,
    telemetry: RoverData | None,
    fetch_telemetry: bool,
    started_monotonic: float,
    checkpoint_sha256: str,
    maximum_frame_age_sec: float,
    maximum_telemetry_age_sec: float,
    clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
    panel_width: int = 480,
    phase1_processor: SamTpPhase1FrameProcessor | None = None,
) -> tuple[ShadowStep, RoverData | None]:
    """Fetch one live frame, infer once, and return a read-only dashboard step.

    The source interface intentionally exposes only GET-equivalent methods.
    This function has no control, mission, checkpoint, or SDK write path.
    """

    request_started = clock()
    request_started_monotonic = monotonic()
    frame = sdk.get_front_frame()
    frame_received = clock()
    frame_received_monotonic = monotonic()
    if fetch_telemetry:
        telemetry = sdk.get_data()
    if frame.image.ndim != 3 or frame.image.shape[2] != 3:
        raise ValueError("SDK front frame must be an HxWx3 BGR image")
    if frame.image.dtype != np.uint8:
        raise ValueError("SDK front frame must use uint8 pixels")

    image_bgr = np.asarray(frame.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    processor = phase1_processor or SamTpPhase1FrameProcessor(
        predictor,
        _PROVISIONAL_PHASE1_TRAJECTORIES,
        checkpoint_sha256[:12],
    )
    phase1 = processor.process(image_rgb, float(frame.timestamp))
    prediction = phase1.prediction
    traversability = phase1.traversability
    finished = clock()
    finished_monotonic = monotonic()
    acquisition_latency_sec = frame_received_monotonic - request_started_monotonic
    end_to_end_latency_sec = finished_monotonic - request_started_monotonic
    if acquisition_latency_sec < 0.0 or not math.isfinite(acquisition_latency_sec):
        raise ValueError("acquisition latency is invalid")
    if end_to_end_latency_sec < 0.0 or not math.isfinite(end_to_end_latency_sec):
        raise ValueError("end-to-end latency is invalid")
    local_frame_age_sec = finished - float(frame.timestamp)
    if not math.isfinite(local_frame_age_sec):
        raise ValueError("local frame age is not finite")
    sdk_frame_age_sec = (
        finished - float(frame.sdk_timestamp)
        if frame.sdk_timestamp is not None
        else None
    )
    if sdk_frame_age_sec is not None and not math.isfinite(sdk_frame_age_sec):
        raise ValueError("SDK frame age is not finite")
    frame_stale = (
        local_frame_age_sec < 0.0
        or local_frame_age_sec > maximum_frame_age_sec
        or (
            sdk_frame_age_sec is not None
            and (
                sdk_frame_age_sec < -maximum_frame_age_sec
                or sdk_frame_age_sec > maximum_frame_age_sec
            )
        )
    )
    telemetry_age_sec = (
        finished - float(telemetry.timestamp) if telemetry is not None else None
    )
    if telemetry_age_sec is not None and not math.isfinite(telemetry_age_sec):
        raise ValueError("telemetry age is not finite")
    telemetry_stale = (
        telemetry_age_sec is None
        or telemetry_age_sec < 0.0
        or telemetry_age_sec > maximum_telemetry_age_sec
    )
    if frame_stale:
        shadow_state = "STALE_FRAME"
    elif telemetry_stale:
        shadow_state = "STALE_TELEMETRY"
    else:
        shadow_state = "CLEAR"
    elapsed = finished_monotonic - started_monotonic
    effective_fps = (frame_index + 1) / elapsed if elapsed > 0.0 else 0.0
    record = {
        "frame_index": frame_index,
        "request_started_timestamp": request_started,
        "frame_received_timestamp": frame_received,
        "local_frame_timestamp": frame.timestamp,
        "sdk_frame_timestamp": frame.sdk_timestamp,
        "local_frame_age_sec": local_frame_age_sec,
        "sdk_frame_age_sec": sdk_frame_age_sec,
        "telemetry_age_sec": telemetry_age_sec,
        "acquisition_latency_ms": acquisition_latency_sec * 1000.0,
        "inference_latency_ms": prediction.inference_time_ms,
        "end_to_end_latency_ms": end_to_end_latency_sec * 1000.0,
        "effective_fps": effective_fps,
        "score_min": float(prediction.traversability_score.min()),
        "score_max": float(prediction.traversability_score.max()),
        "score_mean": float(prediction.traversability_score.mean()),
        "score_std": float(prediction.traversability_score.std()),
        "adapter_confidence": traversability.confidence,
        "candidate_trajectory_count": len(phase1.trajectories),
        "trajectory_geometry_only": True,
        "camera_projection_applied": False,
        "image_path_valid": phase1.image_path.valid,
        "image_path_reason": phase1.image_path.reason,
        "image_path_mean_score": phase1.image_path.mean_score,
        "image_path_visualization_only": True,
        "prediction_valid": not frame_stale,
        "telemetry_valid": not telemetry_stale,
        "shadow_state": shadow_state,
        "checkpoint_sha256": checkpoint_sha256,
        "telemetry": telemetry_record(telemetry),
        "sdk_allowed_read_endpoints": ["/v2/front", "/front", "/data"],
        "command_transmitted": False,
    }
    dashboard = compose_shadow_dashboard(
        image_bgr,
        prediction.traversability_score,
        record,
        telemetry,
        panel_width,
        phase1.image_path,
    )
    return ShadowStep(dashboard, record), telemetry


def telemetry_record(data: RoverData | None) -> dict[str, object] | None:
    if data is None:
        return None
    return {
        "local_timestamp": data.timestamp,
        "sdk_timestamp": data.sdk_timestamp,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "orientation": data.orientation,
        "speed": data.speed,
        "rpms": data.rpms,
        "battery": data.battery,
        "signal_level": data.signal_level,
        "gps_signal": data.gps_signal,
    }


def compose_shadow_dashboard(
    image_bgr: np.ndarray,
    score: np.ndarray,
    record: dict[str, object],
    telemetry: RoverData | None,
    panel_width: int,
    image_path=None,
) -> np.ndarray:
    if panel_width <= 0:
        raise ValueError("panel_width must be positive")
    if score.shape != image_bgr.shape[:2]:
        raise ValueError("score shape must match the SDK frame")
    height, width = image_bgr.shape[:2]
    panel_height = max(1, int(round(panel_width * height / width)))
    original = cv2.resize(
        image_bgr,
        (panel_width, panel_height),
        interpolation=cv2.INTER_AREA,
    )
    if image_path is not None:
        path_rgb = draw_image_path_rgb(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            image_path,
            0.018,
        )
        original = cv2.resize(
            cv2.cvtColor(path_rgb, cv2.COLOR_RGB2BGR),
            (panel_width, panel_height),
            interpolation=cv2.INTER_AREA,
        )
    heatmap_rgb = score_to_heatmap(score)
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    heatmap_bgr = cv2.resize(
        heatmap_bgr,
        (panel_width, panel_height),
        interpolation=cv2.INTER_LINEAR,
    )
    overlay = cv2.addWeighted(original, 0.55, heatmap_bgr, 0.45, 0.0)
    header_height = 64
    footer_height = 118
    canvas = np.full(
        (header_height + panel_height + footer_height, panel_width * 3, 3),
        24,
        dtype=np.uint8,
    )
    canvas[header_height : header_height + panel_height, :panel_width] = original
    canvas[
        header_height : header_height + panel_height,
        panel_width : panel_width * 2,
    ] = overlay
    canvas[
        header_height : header_height + panel_height,
        panel_width * 2 :,
    ] = heatmap_bgr
    titles = (
        "SDK FRONT + IMAGE PATH",
        "SAM-TP OVERLAY + PATH",
        "SCORE: BLUE LOW / RED HIGH",
    )
    for index, title in enumerate(titles):
        _text(canvas, title, index * panel_width + 12, 27, 0.55)
    state = str(record["shadow_state"])
    state_color = (70, 220, 70) if state == "CLEAR" else (40, 80, 240)
    cv2.putText(
        canvas,
        f"READ-ONLY SHADOW | {state} | command_transmitted=false",
        (12, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        state_color,
        1,
        cv2.LINE_AA,
    )
    footer_y = header_height + panel_height + 28
    _text(
        canvas,
        (
            f"frame={record['frame_index']}  acquisition="
            f"{float(record['acquisition_latency_ms']):.1f}ms  inference="
            f"{float(record['inference_latency_ms']):.1f}ms  end-to-end="
            f"{float(record['end_to_end_latency_ms']):.1f}ms  effective="
            f"{float(record['effective_fps']):.2f} FPS"
        ),
        12,
        footer_y,
        0.52,
    )
    _text(
        canvas,
        (
            f"local_age={float(record['local_frame_age_sec']):.3f}s  "
            f"telemetry_age={record['telemetry_age_sec']}  "
            f"sdk_ts={record['sdk_frame_timestamp']}  "
            f"score min/mean/max={float(record['score_min']):.3f}/"
            f"{float(record['score_mean']):.3f}/{float(record['score_max']):.3f}  "
            f"path={record['image_path_reason']}"
        ),
        12,
        footer_y + 28,
        0.50,
    )
    if telemetry is None:
        telemetry_text = "telemetry: waiting"
    else:
        telemetry_text = (
            f"GPS=({telemetry.latitude}, {telemetry.longitude})  "
            f"heading={telemetry.orientation}  speed={telemetry.speed}  "
            f"battery={telemetry.battery}  signal={telemetry.signal_level}"
        )
    _text(canvas, telemetry_text, 12, footer_y + 56, 0.50)
    _text(
        canvas,
        f"checkpoint={str(record['checkpoint_sha256'])[:16]}  q/esc: quit",
        12,
        footer_y + 84,
        0.48,
    )
    return canvas


def write_shadow_summary(
    output_dir: str | Path,
    records: list[dict[str, object]],
    failures: list[dict[str, object]],
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
) -> Path:
    output = Path(output_dir)
    values = [float(item["end_to_end_latency_ms"]) for item in records]
    inference = [float(item["inference_latency_ms"]) for item in records]
    summary = {
        "success": bool(records),
        "processed_frame_count": len(records),
        "failed_frame_count": len(failures),
        "failures": failures,
        "stale_frame_count": sum(not bool(item["prediction_valid"]) for item in records),
        "stale_telemetry_count": sum(
            not bool(item["telemetry_valid"]) for item in records
        ),
        "end_to_end_latency_ms": _latency(values),
        "inference_latency_ms": _latency(inference),
        "effective_fps": records[-1]["effective_fps"] if records else 0.0,
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "sdk_allowed_read_endpoints": ["/v2/front", "/front", "/data"],
        "sdk_write_endpoints": [],
        "command_transmitted": False,
        "live_motion_command_sent_by_process": False,
    }
    path = output / "shadow_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _latency(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _text(
    image: np.ndarray,
    value: str,
    x: int,
    y: int,
    scale: float,
) -> None:
    cv2.putText(
        image,
        value,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
