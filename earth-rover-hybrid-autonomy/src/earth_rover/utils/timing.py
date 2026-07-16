from __future__ import annotations

import time


def sleep_to_maintain_loop_hz(loop_hz: float, loop_start: float | None = None) -> None:
    if loop_hz <= 0:
        return
    elapsed = 0.0 if loop_start is None else time.monotonic() - loop_start
    delay = max(0.0, (1.0 / loop_hz) - elapsed)
    time.sleep(delay)

