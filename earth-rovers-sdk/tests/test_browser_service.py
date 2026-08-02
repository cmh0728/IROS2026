import asyncio
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


def test_resolve_chrome_executable_rejects_missing_linux_browser(monkeypatch) -> None:
    monkeypatch.delenv("CHROME_EXECUTABLE_PATH", raising=False)
    monkeypatch.setattr(browser_service.shutil, "which", lambda _name: None)

    with pytest.raises(
        browser_service.BrowserConfigurationError,
        match="Chrome/Chromium was not found",
    ):
        browser_service.resolve_chrome_executable()


def test_concurrent_initialization_launches_browser_once(monkeypatch) -> None:
    launch_calls = []

    class FakePage:
        async def setViewport(self, _viewport):
            pass

        async def setExtraHTTPHeaders(self, _headers):
            pass

        async def goto(self, _url, _options):
            pass

        async def click(self, _selector):
            pass

        async def waitForSelector(self, _selector):
            pass

        async def waitFor(self, _milliseconds):
            pass

        async def evaluate(self, _script):
            pass

    class FakeBrowser:
        async def newPage(self):
            return FakePage()

        async def close(self):
            pass

    async def fake_launch(**options):
        launch_calls.append(options)
        await asyncio.sleep(0)
        return FakeBrowser()

    async def run_test():
        service = browser_service.BrowserService()
        await asyncio.gather(
            service.initialize_browser(),
            service.initialize_browser(),
        )

    monkeypatch.setattr(browser_service, "resolve_chrome_executable", lambda: "/chrome")
    monkeypatch.setattr(browser_service, "launch", fake_launch)

    asyncio.run(run_test())

    assert len(launch_calls) == 1
    assert launch_calls[0]["handleSIGINT"] is False
    assert launch_calls[0]["handleSIGTERM"] is False
    assert launch_calls[0]["handleSIGHUP"] is False
    assert launch_calls[0]["autoClose"] is False
