from __future__ import annotations

import time
from typing import Any

import requests

from earth_rover.core.types import ControlCommand, FrameData, RoverData
from earth_rover.utils.image import decode_base64_image
from earth_rover.utils.math_utils import safe_float


class SDKClientError(RuntimeError):
    """Raised when the SDK cannot return a usable response."""


class EarthRoverSDKClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request_json(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()
        except requests.RequestException as exc:
            raise SDKClientError(f"{method} {path} failed: {exc}") from exc
        except ValueError as exc:
            raise SDKClientError(f"{method} {path} returned invalid JSON") from exc

    def _get_frame(self, source: str) -> FrameData:
        paths = [f"/v2/{source}", f"/{source}"]
        last_error: Exception | None = None
        for path in paths:
            try:
                local_timestamp = time.time()
                payload = self._request_json("GET", path)
                encoded = self._extract_image_payload(payload)
                return FrameData(
                    timestamp=local_timestamp,
                    image=decode_base64_image(encoded),
                    source=source,
                    sdk_timestamp=safe_float(payload.get("timestamp")),
                )
            except Exception as exc:
                last_error = exc
        raise SDKClientError(f"Could not fetch {source} frame: {last_error}") from last_error

    @staticmethod
    def _extract_image_payload(payload: dict[str, Any]) -> str:
        candidates = [
            payload.get("image"),
            payload.get("frame"),
            payload.get("data"),
            payload.get("front"),
            payload.get("rear"),
            payload.get("front_frame"),
            payload.get("rear_frame"),
            payload.get("map_frame"),
            payload.get("rear_video_frame"),
        ]
        nested = payload.get("camera") or payload.get("result") or {}
        if isinstance(nested, dict):
            candidates.extend(
                [
                    nested.get("image"),
                    nested.get("frame"),
                    nested.get("data"),
                    nested.get("front_frame"),
                    nested.get("rear_frame"),
                ]
            )
        for value in candidates:
            if isinstance(value, str) and value:
                return value
        raise SDKClientError("Frame response did not contain a base64 image")

    def get_front_frame(self) -> FrameData:
        return self._get_frame("front")

    def get_rear_frame(self) -> FrameData:
        return self._get_frame("rear")

    def get_data(self) -> RoverData:
        local_timestamp = time.time()
        payload = self._request_json("GET", "/data")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        gps = data.get("gps") if isinstance(data.get("gps"), dict) else {}
        status = data.get("status") if isinstance(data.get("status"), dict) else {}
        imu = data.get("imu") if isinstance(data.get("imu"), dict) else {}

        latitude = safe_float(data.get("latitude", gps.get("latitude", gps.get("lat"))))
        longitude = safe_float(data.get("longitude", gps.get("longitude", gps.get("lon"))))
        orientation = safe_float(
            data.get("orientation", data.get("heading", imu.get("orientation", imu.get("yaw"))))
        )
        speed = safe_float(data.get("speed", status.get("speed")))
        rpms_raw = data.get("rpms", data.get("rpm", status.get("rpms", status.get("rpm"))))
        rpms = self._parse_rpms(rpms_raw)

        return RoverData(
            timestamp=local_timestamp,
            latitude=latitude,
            longitude=longitude,
            orientation=orientation,
            speed=speed,
            rpms=rpms,
            battery=safe_float(data.get("battery", status.get("battery"))),
            signal_level=safe_float(data.get("signal_level", status.get("signal_level", data.get("signal")))),
            gps_signal=safe_float(data.get("gps_signal", gps.get("signal", gps.get("quality")))),
            raw=payload,
            sdk_timestamp=safe_float(data.get("timestamp", payload.get("timestamp"))),
        )

    @staticmethod
    def _parse_rpms(value: Any) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            value = list(value.values())
        if not isinstance(value, (list, tuple)):
            value = [value]
        if value and all(isinstance(item, (list, tuple)) for item in value):
            latest = list(value[-1])
            if latest and safe_float(latest[-1]) is not None and abs(float(latest[-1])) > 1_000_000:
                latest = latest[:-1]
            value = latest
        parsed = [safe_float(item) for item in value]
        return [item for item in parsed if item is not None]

    def send_control(self, command: ControlCommand) -> bool:
        payload = {
            "command": {
                "linear": float(command.linear),
                "angular": float(command.angular),
                "lamp": int(command.lamp),
            }
        }
        self._request_json("POST", "/control", json=payload)
        return True

    def get_checkpoints(self) -> list[dict]:
        return self.get_checkpoint_state()["checkpoints"]

    def get_checkpoint_state(self) -> dict[str, Any]:
        payload = self._request_json("GET", "/checkpoints-list")
        checkpoints = payload.get("checkpoints_list", payload.get("checkpoints", payload.get("data", payload)))
        if isinstance(checkpoints, dict):
            checkpoints = checkpoints.get("checkpoints_list", checkpoints.get("checkpoints", []))
        return {
            "checkpoints": checkpoints if isinstance(checkpoints, list) else [],
            "latest_scanned_checkpoint": safe_float(payload.get("latest_scanned_checkpoint"), 0),
            "raw": payload,
        }

    def report_checkpoint(self) -> bool:
        self._request_json("POST", "/checkpoint-reached", json={})
        return True

    def start_mission(self) -> bool:
        self._request_json("POST", "/start-mission")
        return True

    def end_mission(self) -> bool:
        self._request_json("POST", "/end-mission")
        return True
