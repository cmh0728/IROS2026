from pathlib import Path

from fastapi.testclient import TestClient

import main


ROOT = Path(__file__).resolve().parents[1]


def test_mission_status_does_not_expose_identifiers(monkeypatch) -> None:
    monkeypatch.setenv("MISSION_SLUG", "private-mission")
    monkeypatch.setenv("BOT_SLUG", "private-rover")
    monkeypatch.setattr(main, "auth_response_data", {"CHANNEL_NAME": "private-channel"})
    monkeypatch.setattr(
        main,
        "checkpoints_list_data",
        {
            "checkpoints_list": [{"sequence": 1}],
            "latest_scanned_checkpoint": 0,
        },
    )

    payload = main.mission_status_payload()

    assert payload["mission_configured"] is True
    assert payload["mission_active"] is True
    assert payload["checkpoint_count"] == 1
    assert payload["latest_scanned_checkpoint"] == 0
    serialized = str(payload)
    assert "private-mission" not in serialized
    assert "private-rover" not in serialized
    assert "private-channel" not in serialized


def test_dashboard_is_available_without_active_mission(monkeypatch) -> None:
    monkeypatch.setattr(main, "auth_response_data", {})
    monkeypatch.setattr(main, "checkpoints_list_data", {})
    client = TestClient(main.app)

    status = client.get("/mission-status")
    dashboard = client.get("/dashboard")

    assert status.status_code == 200
    assert status.json()["mission_active"] is False
    assert dashboard.status_code == 200
    assert "Start Mission" in dashboard.text
    assert "Mission API Results" in dashboard.text


def test_dashboard_javascript_has_no_control_endpoint() -> None:
    source = (ROOT / "static/mission_dashboard.js").read_text(encoding="utf-8")

    assert '"/start-mission"' in source
    assert '"/mission-status"' in source
    assert '"/checkpoints-list"' in source
    assert '"/end-mission"' in source
    assert '"/control"' not in source
    assert "send_control" not in source
