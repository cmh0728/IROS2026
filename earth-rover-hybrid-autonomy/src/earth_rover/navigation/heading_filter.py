from __future__ import annotations

import math


class HeadingFilter:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._sin: float | None = None
        self._cos: float | None = None

    def update_deg(self, heading_deg: float) -> float:
        radians = math.radians(float(heading_deg))
        new_sin = math.sin(radians)
        new_cos = math.cos(radians)
        if self._sin is None or self._cos is None:
            self._sin = new_sin
            self._cos = new_cos
        else:
            self._sin = self.alpha * self._sin + (1.0 - self.alpha) * new_sin
            self._cos = self.alpha * self._cos + (1.0 - self.alpha) * new_cos
        return (math.degrees(math.atan2(self._sin, self._cos)) + 360.0) % 360.0

