import base64

import cv2
import numpy as np

from earth_rover.core.types import ControlCommand
from earth_rover.sdk_client import EarthRoverSDKClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.content = b"{}"

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def request(self, method, url, timeout=None, **kwargs):
        path = "/" + url.split("/", 3)[3]
        self.calls.append((method, path, kwargs))
        return FakeResponse(self.routes[(method, path)])


def encoded_image():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(buffer).decode("ascii")


def client_with(routes):
    client = EarthRoverSDKClient("http://localhost:8000", 0.25)
    client.session = FakeSession(routes)
    return client


def test_front_frame_uses_v2_front_and_front_frame_key():
    client = client_with({("GET", "/v2/front"): {"front_frame": encoded_image(), "timestamp": 1.0}})

    frame = client.get_front_frame()

    assert frame.image.shape == (3, 4, 3)
    assert frame.sdk_timestamp == 1.0
    assert client.session.calls[0][0:2] == ("GET", "/v2/front")


def test_mission_and_checkpoint_endpoints_match_official_sdk():
    routes = {
        ("POST", "/start-mission"): {"message": "Mission started successfully"},
        ("GET", "/checkpoints-list"): {
            "checkpoints_list": [{"sequence": 1, "latitude": "30.48243713", "longitude": "114.3026428"}],
            "latest_scanned_checkpoint": 0,
        },
        ("POST", "/checkpoint-reached"): {"message": "Checkpoint reached successfully"},
        ("POST", "/end-mission"): {"message": "Mission ended successfully"},
    }
    client = client_with(routes)

    assert client.start_mission() is True
    assert client.get_checkpoints()[0]["sequence"] == 1
    assert client.get_checkpoint_state()["latest_scanned_checkpoint"] == 0.0
    assert client.report_checkpoint() is True
    assert client.end_mission() is True

    assert [call[0:2] for call in client.session.calls] == [
        ("POST", "/start-mission"),
        ("GET", "/checkpoints-list"),
        ("GET", "/checkpoints-list"),
        ("POST", "/checkpoint-reached"),
        ("POST", "/end-mission"),
    ]


def test_control_payload_matches_official_sdk():
    client = client_with({("POST", "/control"): {"message": "Command sent successfully"}})

    assert client.send_control(ControlCommand(0.1, -0.2, lamp=1)) is True

    method, path, kwargs = client.session.calls[0]
    assert (method, path) == ("POST", "/control")
    assert kwargs["json"] == {"command": {"linear": 0.1, "angular": -0.2, "lamp": 1}}


def test_data_parses_official_nested_rpm_shape():
    client = client_with(
        {
            ("GET", "/data"): {
                "battery": 100,
                "signal_level": 5,
                "orientation": 128,
                "speed": 0,
                "gps_signal": 31.25,
                "latitude": 22.753774642944336,
                "longitude": 114.09095001220703,
                "timestamp": 1724189733.208559,
                "rpms": [
                    [1, 2, 3, 4, 1725434567.194],
                    [5, 6, 7, 8, 1725434597.726],
                ],
            }
        }
    )

    data = client.get_data()

    assert data.latitude == 22.753774642944336
    assert data.longitude == 114.09095001220703
    assert data.orientation == 128
    assert data.rpms == [5.0, 6.0, 7.0, 8.0]
    assert data.sdk_timestamp == 1724189733.208559
