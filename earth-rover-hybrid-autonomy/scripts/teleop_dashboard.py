#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earth_rover.core.types import ControlCommand
from earth_rover.sdk_client import EarthRoverSDKClient
from earth_rover.utils.config import load_config
from earth_rover.utils.image import decode_base64_image
from earth_rover.utils.math_utils import safe_float


class SharedState:
    def __init__(self, trail_limit: int):
        self.lock = threading.Lock()
        self.api_lock = threading.Lock()
        self.running = True
        self.frame = None
        self.frame_timestamp = None
        self.frame_error = ""
        self.data = None
        self.data_error = ""
        self.trail = deque(maxlen=trail_limit)
        self.linear = 0.0
        self.angular = 0.0
        self.lamp = 0
        self.drive_speed = 0.2
        self.turn_speed = 0.35
        self.angular_hold_timeout = 0.35
        self.last_angular_key_time = 0.0
        self.control_error = ""


def fetch_v2_frame(sdk: EarthRoverSDKClient, source: str) -> tuple[np.ndarray, float | None]:
    payload = sdk._request_json("GET", f"/v2/{source}")
    key = f"{source}_frame"
    encoded = payload.get(key)
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError(f"response did not contain {key}")
    return decode_base64_image(encoded), safe_float(payload.get("timestamp"))


def camera_worker(state: SharedState, base_url: str, timeout: float, source: str, fps: float) -> None:
    sdk = EarthRoverSDKClient(base_url, timeout)
    delay = 1.0 / fps
    while True:
        with state.lock:
            if not state.running:
                return
        started = time.monotonic()
        try:
            with state.api_lock:
                frame, timestamp = fetch_v2_frame(sdk, source)
            with state.lock:
                state.frame = frame
                state.frame_timestamp = timestamp
                state.frame_error = ""
        except Exception as exc:
            with state.lock:
                state.frame_error = str(exc)
            time.sleep(max(delay, 1.0))
            continue
        remaining = delay - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)


def telemetry_worker(state: SharedState, base_url: str, timeout: float, hz: float) -> None:
    sdk = EarthRoverSDKClient(base_url, timeout)
    delay = 1.0 / hz
    while True:
        with state.lock:
            if not state.running:
                return
        try:
            with state.api_lock:
                data = sdk.get_data()
            with state.lock:
                state.data = data
                state.data_error = ""
                if data.latitude is not None and data.longitude is not None:
                    state.trail.append((data.latitude, data.longitude))
        except Exception as exc:
            with state.lock:
                state.data_error = str(exc)
            time.sleep(max(delay, 1.0))
            continue
        time.sleep(delay)


def control_worker(state: SharedState, base_url: str, timeout: float, hz: float) -> None:
    sdk = EarthRoverSDKClient(base_url, timeout)
    delay = 1.0 / hz
    while True:
        with state.lock:
            if not state.running:
                return
            if time.monotonic() - state.last_angular_key_time > state.angular_hold_timeout:
                state.angular = 0.0
            linear = state.linear
            angular = state.angular
            lamp = state.lamp
        try:
            with state.api_lock:
                sdk.send_control(ControlCommand(linear, angular, lamp=lamp, mode="TELEOP"))
            with state.lock:
                state.control_error = ""
        except Exception as exc:
            with state.lock:
                state.control_error = str(exc)
        time.sleep(delay)


def latlon_to_xy_m(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(origin_lat))
    x = (lon - origin_lon) * meters_per_deg_lon
    y = (lat - origin_lat) * meters_per_deg_lat
    return x, y


def draw_text(image: np.ndarray, text: str, x: int, y: int, scale: float = 0.55) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA)


def clamp_command(value: float) -> float:
    return max(-1.0, min(1.0, value))


def draw_minimap(panel: np.ndarray, trail: list[tuple[float, float]], orientation: float | None) -> None:
    x0, y0, w, h = 14, 260, panel.shape[1] - 28, 280
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (45, 45, 45), -1)
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (120, 120, 120), 1)
    draw_text(panel, "Local GPS trail", x0 + 8, y0 + 24, 0.55)

    for i in range(1, 4):
        gx = x0 + int(w * i / 4)
        gy = y0 + int(h * i / 4)
        cv2.line(panel, (gx, y0), (gx, y0 + h), (65, 65, 65), 1)
        cv2.line(panel, (x0, gy), (x0 + w, gy), (65, 65, 65), 1)

    if not trail:
        draw_text(panel, "waiting for GPS", x0 + 80, y0 + h // 2, 0.55)
        return

    origin_lat, origin_lon = trail[0]
    points_m = [latlon_to_xy_m(lat, lon, origin_lat, origin_lon) for lat, lon in trail]
    max_abs = max([5.0] + [abs(v) for point in points_m for v in point])
    scale = 0.45 * min(w, h) / max_abs
    cx, cy = x0 + w // 2, y0 + h // 2

    points_px = []
    for mx, my in points_m:
        px = int(cx + mx * scale)
        py = int(cy - my * scale)
        points_px.append((px, py))

    for idx in range(1, len(points_px)):
        cv2.line(panel, points_px[idx - 1], points_px[idx], (0, 180, 255), 2)

    marker = points_px[-1]
    cv2.circle(panel, marker, 6, (0, 255, 0), -1)
    if orientation is not None:
        heading_rad = math.radians(orientation)
        end = (
            int(marker[0] + math.sin(heading_rad) * 24),
            int(marker[1] - math.cos(heading_rad) * 24),
        )
        cv2.arrowedLine(panel, marker, end, (0, 255, 0), 2, tipLength=0.35)

    draw_text(panel, f"scale +/- {max_abs:.1f} m", x0 + 8, y0 + h - 10, 0.45)


def build_dashboard(state: SharedState, window_size: tuple[int, int], source: str) -> np.ndarray:
    width, height = window_size
    panel_w = 390
    cam_w = width - panel_w
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    with state.lock:
        frame = None if state.frame is None else state.frame.copy()
        frame_timestamp = state.frame_timestamp
        frame_error = state.frame_error
        data = state.data
        data_error = state.data_error
        trail = list(state.trail)
        linear = state.linear
        angular = state.angular
        lamp = state.lamp
        drive_speed = state.drive_speed
        turn_speed = state.turn_speed
        control_error = state.control_error

    if frame is None:
        cv2.rectangle(canvas, (0, 0), (cam_w, height), (20, 20, 20), -1)
        draw_text(canvas, f"waiting for {source} camera", 30, 50, 0.8)
        if frame_error:
            draw_text(canvas, frame_error[:90], 30, 85, 0.45)
    else:
        fh, fw = frame.shape[:2]
        scale = min(cam_w / fw, height / fh)
        resized = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
        y = (height - resized.shape[0]) // 2
        x = (cam_w - resized.shape[1]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.rectangle(canvas, (0, 0), (cam_w, 34), (0, 0, 0), -1)
        draw_text(canvas, f"{source} camera | sdk_ts={frame_timestamp}", 12, 23, 0.6)

    panel = canvas[:, cam_w:]
    cv2.rectangle(panel, (0, 0), (panel_w, height), (28, 30, 34), -1)
    draw_text(panel, "Teleop Dashboard", 14, 30, 0.75)
    draw_text(panel, "W/S linear +/-  hold A/D turn", 14, 62, 0.5)
    draw_text(panel, "Space stop  L lamp  +/- step", 14, 84, 0.5)
    draw_text(panel, "Q or Esc quit", 14, 106, 0.5)

    y = 142
    if data is None:
        draw_text(panel, "telemetry: waiting", 14, y)
        if data_error:
            draw_text(panel, data_error[:48], 14, y + 24, 0.42)
    else:
        draw_text(panel, f"battery: {data.battery}", 14, y)
        draw_text(panel, f"signal: {data.signal_level}  gps: {data.gps_signal}", 14, y + 24)
        draw_text(panel, f"lat: {data.latitude}", 14, y + 48)
        draw_text(panel, f"lon: {data.longitude}", 14, y + 72)
        draw_text(panel, f"heading: {data.orientation}  speed: {data.speed}", 14, y + 96)
        rpms = data.rpms if data.rpms is not None else []
        draw_text(panel, f"rpms: {rpms[:4]}", 14, y + 120, 0.48)

    draw_minimap(panel, trail, data.orientation if data else None)

    y2 = 580
    draw_text(panel, f"cmd linear: {linear:+.2f}", 14, y2)
    draw_text(panel, f"cmd angular: {angular:+.2f}", 14, y2 + 24)
    draw_text(panel, f"step linear: {drive_speed:.2f}  angular: {turn_speed:.2f}", 14, y2 + 48)
    draw_text(panel, f"lamp: {'on' if lamp else 'off'}", 14, y2 + 72)
    if control_error:
        draw_text(panel, f"control error: {control_error[:42]}", 14, y2 + 102, 0.42)

    return canvas


def handle_key(state: SharedState, key: int) -> bool:
    if key in (27, ord("q")):
        return False
    with state.lock:
        if key in (ord("w"), ord("W")):
            state.linear = clamp_command(state.linear + state.drive_speed)
        elif key in (ord("s"), ord("S")):
            state.linear = clamp_command(state.linear - state.drive_speed)
        elif key in (ord("a"), ord("A")):
            state.angular = state.turn_speed
            state.last_angular_key_time = time.monotonic()
        elif key in (ord("d"), ord("D")):
            state.angular = -state.turn_speed
            state.last_angular_key_time = time.monotonic()
        elif key == ord(" "):
            state.linear = 0.0
            state.angular = 0.0
            state.last_angular_key_time = 0.0
        elif key in (ord("l"), ord("L")):
            state.lamp = 1 - state.lamp
        elif key in (ord("+"), ord("=")):
            state.drive_speed = min(1.0, state.drive_speed + 0.05)
            state.turn_speed = min(1.0, state.turn_speed + 0.05)
        elif key in (ord("-"), ord("_")):
            state.drive_speed = max(0.05, state.drive_speed - 0.05)
            state.turn_speed = max(0.05, state.turn_speed - 0.05)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Camera, keyboard teleop, and GPS dashboard for the Earth Rovers SDK.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", choices=["front", "rear"], default="front")
    parser.add_argument("--camera-fps", type=float, default=15.0)
    parser.add_argument("--telemetry-hz", type=float, default=2.0)
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--trail-limit", type=int, default=500)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--start-mission", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    sdk_cfg = config["sdk"]
    base_url = sdk_cfg["base_url"]
    state = SharedState(args.trail_limit)

    if args.start_mission:
        EarthRoverSDKClient(base_url, args.timeout).start_mission()
        print("mission started")

    workers = [
        threading.Thread(target=camera_worker, args=(state, base_url, args.timeout, args.source, args.camera_fps), daemon=True),
        threading.Thread(target=telemetry_worker, args=(state, base_url, args.timeout, args.telemetry_hz), daemon=True),
        threading.Thread(target=control_worker, args=(state, base_url, args.timeout, args.control_hz), daemon=True),
    ]
    for worker in workers:
        worker.start()

    window_name = "Earth Rover Teleop Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    try:
        while True:
            dashboard = build_dashboard(state, (args.window_width, args.window_height), args.source)
            cv2.imshow(window_name, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key != 255 and not handle_key(state, key):
                break
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        pass
    finally:
        with state.lock:
            state.running = False
        try:
            with state.api_lock:
                EarthRoverSDKClient(base_url, args.timeout).send_control(
                    ControlCommand(0.0, 0.0, lamp=state.lamp, mode="TELEOP_STOP")
                )
        except Exception:
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
