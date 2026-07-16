#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earth_rover.sdk_client import EarthRoverSDKClient
from earth_rover.utils.config import load_config
from earth_rover.utils.image import decode_base64_image
from earth_rover.utils.math_utils import safe_float


def draw_status(frame, source: str, fps: float, sdk_timestamp: float | None) -> None:
    timestamp_text = f"sdk_ts={sdk_timestamp:.3f}" if sdk_timestamp is not None else "sdk_ts=N/A"
    text = f"{source} | {fps:.1f} FPS | {timestamp_text} | q/esc to quit"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def fetch_v2_frame(sdk: EarthRoverSDKClient, source: str) -> tuple:
    payload = sdk._request_json("GET", f"/v2/{source}")
    key = f"{source}_frame"
    encoded = payload.get(key)
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError(f"response did not contain {key}")
    return decode_base64_image(encoded), safe_float(payload.get("timestamp"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Display Earth Rovers SDK camera frames in a local OpenCV window.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML relative to project root.")
    parser.add_argument("--source", choices=["front", "rear"], default="front", help="Camera source to display.")
    parser.add_argument("--fps", type=float, default=15.0, help="Polling rate for camera frames.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout for camera frame requests.")
    parser.add_argument("--start-mission", action="store_true", help="Call POST /start-mission before viewing.")
    parser.add_argument("--window-name", default="Earth Rover SDK Camera", help="OpenCV window title.")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than 0")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")

    config = load_config(ROOT / args.config)
    sdk_cfg = config["sdk"]
    sdk = EarthRoverSDKClient(sdk_cfg["base_url"], args.timeout)

    if args.start_mission:
        print("starting mission...")
        sdk.start_mission()
        print("mission started")

    delay_sec = 1.0 / args.fps
    last_frame_time = time.monotonic()
    display_fps = 0.0

    print(f"viewing {args.source} camera from {sdk.base_url}")
    print("press q or esc in the image window to quit")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            loop_start = time.monotonic()
            try:
                frame, sdk_timestamp = fetch_v2_frame(sdk, args.source)
            except Exception as exc:
                print(f"frame fetch failed: {exc}", file=sys.stderr)
                time.sleep(max(delay_sec, 1.0))
                continue

            now = time.monotonic()
            elapsed = now - last_frame_time
            if elapsed > 0:
                display_fps = 1.0 / elapsed
            last_frame_time = now

            frame = frame.copy()
            draw_status(frame, args.source, display_fps, sdk_timestamp)
            cv2.imshow(args.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            remaining = delay_sec - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
