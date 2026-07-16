from __future__ import annotations

from typing import Protocol

from earth_rover.core.types import FrameData, PerceptionResult


class PerceptionModel(Protocol):
    def infer(self, frame: FrameData) -> PerceptionResult:
        ...

