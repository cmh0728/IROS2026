from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from earth_rover.core.types import FrameData, RoverData
from training.sam_tp_reproduction import SamTpPrediction
from training.sam_tp_sdk_shadow import run_shadow_step, write_shadow_summary


class ReadOnlyFakeSdk:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.image = np.zeros((12, 20, 3), dtype=np.uint8)
        self.image[0, 0] = [10, 20, 30]

    def get_front_frame(self) -> FrameData:
        self.calls.append("get_front_frame")
        return FrameData(99.9, self.image.copy(), "front", sdk_timestamp=99.8)

    def get_data(self) -> RoverData:
        self.calls.append("get_data")
        return RoverData(
            timestamp=99.95,
            latitude=1.0,
            longitude=2.0,
            orientation=3.0,
            speed=0.0,
            rpms=[0.0, 0.0, 0.0, 0.0],
            battery=90.0,
            signal_level=5.0,
            gps_signal=20.0,
            raw={},
            sdk_timestamp=99.85,
        )

    def send_control(self, *_args, **_kwargs) -> None:
        raise AssertionError("shadow mode must not send control")


class RecordingPredictor:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []

    def predict(self, image_rgb: np.ndarray) -> SamTpPrediction:
        self.images.append(image_rgb.copy())
        score = np.full(image_rgb.shape[:2], 0.75, dtype=np.float32)
        logits = np.full(image_rgb.shape[:2], 1.0, dtype=np.float32)
        return SamTpPrediction(
            raw_logits=logits,
            traversability_score=score,
            heatmap=np.zeros_like(image_rgb),
            input_shape=image_rgb.shape,
            output_shape=score.shape,
            inference_time_ms=10.0,
            device="test",
        )


def test_shadow_step_uses_read_only_sdk_and_explicit_bgr_to_rgb() -> None:
    sdk = ReadOnlyFakeSdk()
    predictor = RecordingPredictor()
    # Wall-clock correction must not affect measured durations.
    clock_values = iter((100.0, 90.0, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, telemetry = run_shadow_step(
        sdk,
        predictor,
        frame_index=0,
        telemetry=None,
        fetch_telemetry=True,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
    )

    assert sdk.calls == ["get_front_frame", "get_data"]
    assert predictor.images[0][0, 0].tolist() == [30, 20, 10]
    assert telemetry is not None
    assert step.record["command_transmitted"] is False
    assert step.record["candidate_trajectory_count"] == 7
    assert step.record["adapter_confidence"] == 1.0
    assert step.record["trajectory_geometry_only"] is True
    assert step.record["camera_projection_applied"] is False
    assert step.record["sdk_allowed_read_endpoints"] == [
        "/v2/front",
        "/front",
        "/data",
    ]
    assert abs(float(step.record["acquisition_latency_ms"]) - 20.0) < 1e-9
    assert abs(float(step.record["end_to_end_latency_ms"]) - 190.0) < 1e-9
    assert step.record["shadow_state"] == "CLEAR"
    assert step.record["telemetry_valid"] is True
    assert step.dashboard_bgr.shape == (242, 300, 3)


def test_shadow_step_marks_old_frame_stale_without_command() -> None:
    sdk = ReadOnlyFakeSdk()
    predictor = RecordingPredictor()
    clock_values = iter((100.0, 100.01, 101.2))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, _ = run_shadow_step(
        sdk,
        predictor,
        frame_index=0,
        telemetry=None,
        fetch_telemetry=False,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
    )

    assert sdk.calls == ["get_front_frame"]
    assert step.record["prediction_valid"] is False
    assert step.record["shadow_state"] == "STALE_FRAME"
    assert step.record["command_transmitted"] is False


def test_shadow_step_reports_stale_telemetry_separately() -> None:
    sdk = ReadOnlyFakeSdk()
    predictor = RecordingPredictor()
    telemetry = sdk.get_data()
    telemetry = RoverData(
        **{
            **telemetry.__dict__,
            "timestamp": 98.0,
        }
    )
    sdk.calls.clear()
    clock_values = iter((100.0, 100.01, 100.1))
    monotonic_values = iter((10.01, 10.03, 10.2))

    step, _ = run_shadow_step(
        sdk,
        predictor,
        frame_index=0,
        telemetry=telemetry,
        fetch_telemetry=False,
        started_monotonic=10.0,
        checkpoint_sha256="abc",
        maximum_frame_age_sec=1.0,
        maximum_telemetry_age_sec=1.0,
        clock=lambda: next(clock_values),
        monotonic=lambda: next(monotonic_values),
        panel_width=100,
    )

    assert step.record["prediction_valid"] is True
    assert step.record["telemetry_valid"] is False
    assert step.record["shadow_state"] == "STALE_TELEMETRY"
    assert step.record["command_transmitted"] is False


def test_shadow_summary_records_no_sdk_write_or_motion(tmp_path: Path) -> None:
    records = [
        {
            "shadow_state": "CLEAR",
            "end_to_end_latency_ms": 100.0,
            "inference_latency_ms": 80.0,
            "effective_fps": 8.0,
            "prediction_valid": True,
            "telemetry_valid": True,
        }
    ]

    path = write_shadow_summary(tmp_path, records, [], "checkpoint.pt", "abc")
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["success"]
    assert report["sdk_write_endpoints"] == []
    assert report["command_transmitted"] is False
    assert report["live_motion_command_sent_by_process"] is False
    assert report["processed_frame_count"] == 1


def test_shadow_step_rejects_non_uint8_sdk_frame() -> None:
    sdk = ReadOnlyFakeSdk()
    sdk.image = np.zeros((12, 20, 3), dtype=np.float32)

    try:
        run_shadow_step(
            sdk,
            RecordingPredictor(),
            frame_index=0,
            telemetry=None,
            fetch_telemetry=False,
            started_monotonic=10.0,
            checkpoint_sha256="abc",
            maximum_frame_age_sec=1.0,
            maximum_telemetry_age_sec=1.0,
            panel_width=100,
        )
    except ValueError as exc:
        assert str(exc) == "SDK front frame must use uint8 pixels"
    else:
        raise AssertionError("non-uint8 SDK frame must be rejected")


def test_shadow_launcher_contains_no_sdk_write_call() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "training/run_sam_tp_sdk_shadow.py").read_text(
        encoding="utf-8"
    )
    core = (root / "training/sam_tp_sdk_shadow.py").read_text(encoding="utf-8")

    for source in (launcher, core):
        assert ".send_control(" not in source
        assert ".start_mission(" not in source
        assert ".end_mission(" not in source
        assert ".report_checkpoint(" not in source
