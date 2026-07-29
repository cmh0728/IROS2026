#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from earth_rover.sdk_client import EarthRoverSDKClient  # noqa: E402
from earth_rover.utils.config import load_config  # noqa: E402
from training.sam_tp_reproduction import (  # noqa: E402
    OFFICIAL_COMMIT,
    SamTpPredictor,
    git_provenance,
    sha256_file,
)
from training.sam_tp_sdk_shadow import (  # noqa: E402
    run_shadow_step,
    write_shadow_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only SDK front-camera SAM-TP shadow inference. "
            "No mission or control endpoint is called."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--telemetry-hz", type=float, default=2.0)
    parser.add_argument("--maximum-frame-age-sec", type=float, default=1.0)
    parser.add_argument("--request-timeout-sec", type=float, default=2.0)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--maximum-consecutive-failures", type=int, default=5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--snapshot-interval", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.target_fps <= 0.0 or args.telemetry_hz <= 0.0:
        raise SystemExit("target-fps and telemetry-hz must be positive")
    if args.maximum_frame_age_sec <= 0.0 or args.request_timeout_sec <= 0.0:
        raise SystemExit("timeouts must be positive")
    if args.panel_width <= 0 or args.snapshot_interval <= 0:
        raise SystemExit("panel-width and snapshot-interval must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("max-frames must be positive")
    if args.maximum_consecutive_failures <= 0:
        raise SystemExit("maximum-consecutive-failures must be positive")
    if not args.headless and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise SystemExit("DISPLAY is unavailable; rerun with --headless")

    upstream = Path(args.upstream_root).expanduser().resolve()
    model_config = Path(args.model_config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    config_path = _rooted(args.config)
    for path in (upstream, model_config, checkpoint, config_path):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    provenance = git_provenance(upstream)
    if provenance["commit"] != OFFICIAL_COMMIT or provenance["dirty"]:
        raise SystemExit(f"upstream checkout is not the frozen clean commit: {provenance}")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise SystemExit(
            "checkpoint SHA-256 differs from the explicitly approved value: "
            f"expected={args.expected_checkpoint_sha256} actual={checkpoint_sha}"
        )

    config = load_config(config_path)
    sdk_cfg = config["sdk"]
    sdk = EarthRoverSDKClient(
        sdk_cfg["base_url"],
        args.request_timeout_sec,
    )
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("SAM-TP shadow mode requires torch in its independent environment") from exc
    if not torch.cuda.is_available():
        raise SystemExit("SAM-TP shadow mode requires CUDA")
    predictor = SamTpPredictor(
        upstream,
        model_config,
        checkpoint,
        synchronize=torch.cuda.synchronize,
    )

    output.mkdir(parents=True)
    jsonl_path = output / "shadow_frames.jsonl"
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    telemetry = None
    telemetry_interval = 1.0 / args.telemetry_hz
    next_telemetry = 0.0
    delay = 1.0 / args.target_fps
    started = time.monotonic()
    consecutive_failures = 0
    window_name = "Earth Rover SAM-TP Read-Only Shadow"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("SAM-TP SDK shadow mode: GET-only, command_transmitted=false", flush=True)
    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            frame_index = 0
            while args.max_frames is None or frame_index < args.max_frames:
                loop_started = time.monotonic()
                try:
                    fetch_telemetry = loop_started >= next_telemetry
                    step, telemetry = run_shadow_step(
                        sdk,
                        predictor,
                        frame_index,
                        telemetry,
                        fetch_telemetry,
                        started,
                        checkpoint_sha,
                        args.maximum_frame_age_sec,
                        panel_width=args.panel_width,
                    )
                    if fetch_telemetry:
                        next_telemetry = loop_started + telemetry_interval
                    records.append(step.record)
                    consecutive_failures = 0
                    jsonl.write(json.dumps(step.record, sort_keys=True) + "\n")
                    jsonl.flush()
                    if (
                        frame_index % args.snapshot_interval == 0
                        or args.max_frames == frame_index + 1
                    ):
                        cv2.imwrite(str(output / "latest_dashboard.jpg"), step.dashboard_bgr)
                    print(
                        f"frame={frame_index} state={step.record['shadow_state']} "
                        f"e2e={float(step.record['end_to_end_latency_ms']):.1f}ms "
                        f"fps={float(step.record['effective_fps']):.2f}",
                        flush=True,
                    )
                    if not args.headless:
                        cv2.imshow(window_name, step.dashboard_bgr)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break
                    frame_index += 1
                except Exception as exc:
                    failure = {
                        "timestamp": time.time(),
                        "frame_index": frame_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    consecutive_failures += 1
                    print(f"shadow frame failed: {failure}", file=sys.stderr, flush=True)
                    if consecutive_failures >= args.maximum_consecutive_failures:
                        print(
                            "maximum consecutive failures reached; stopping shadow mode",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    time.sleep(max(delay, 1.0))
                remaining = delay - (time.monotonic() - loop_started)
                if remaining > 0.0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        write_shadow_summary(
            output,
            records,
            failures,
            checkpoint,
            checkpoint_sha,
        )
    print(f"SAM-TP shadow output: {output}")
    print("No SDK write endpoint or live rover command was used.")
    return 0 if records else 2


def _rooted(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
