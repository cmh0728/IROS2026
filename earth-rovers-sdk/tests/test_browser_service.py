from pathlib import Path

import pytest

import browser_service


def test_resolve_chrome_executable_uses_valid_configured_path(
    monkeypatch, tmp_path: Path
) -> None:
    executable = tmp_path / "chrome"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("CHROME_EXECUTABLE_PATH", str(executable))

    assert browser_service.resolve_chrome_executable() == str(executable)


def test_resolve_chrome_executable_rejects_invalid_configured_path(
    monkeypatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing-chrome"
    monkeypatch.setenv("CHROME_EXECUTABLE_PATH", str(missing))

    with pytest.raises(
        browser_service.BrowserConfigurationError,
        match="CHROME_EXECUTABLE_PATH is not an executable file",
    ):
        browser_service.resolve_chrome_executable()


def test_resolve_chrome_executable_detects_linux_browser(monkeypatch) -> None:
    monkeypatch.delenv("CHROME_EXECUTABLE_PATH", raising=False)
    monkeypatch.setattr(
        browser_service.shutil,
        "which",
        lambda name: "/usr/bin/chromium" if name == "chromium" else None,
    )

    assert browser_service.resolve_chrome_executable() == "/usr/bin/chromium"
