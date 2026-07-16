from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2

from earth_rover.core.types import ControlCommand, FrameData, RoverData
from earth_rover.utils.math_utils import safe_float


class ReplayDataset:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.frames = sorted((self.run_dir / "frames").glob("front_*.jpg"))
        self.rows = self._load_rows(self.run_dir / "data.csv")

    def __iter__(self):
        count = min(len(self.frames), len(self.rows))
        for index in range(count):
            image = cv2.imread(str(self.frames[index]))
            if image is None:
                continue
            row = self.rows[index]
            timestamp = safe_float(row.get("timestamp"), time.time())
            yield FrameData(timestamp=timestamp, image=image, source="replay"), self._row_to_rover_data(row)

    @staticmethod
    def _load_rows(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _row_to_rover_data(row: dict) -> RoverData:
        return RoverData(
            timestamp=safe_float(row.get("timestamp"), time.time()),
            latitude=safe_float(row.get("latitude")),
            longitude=safe_float(row.get("longitude")),
            orientation=safe_float(row.get("orientation")),
            speed=safe_float(row.get("speed")),
            rpms=[safe_float(row["mean_rpm"])] if row.get("mean_rpm") not in {None, ""} else None,
            battery=safe_float(row.get("battery")),
            signal_level=safe_float(row.get("signal_level")),
            gps_signal=safe_float(row.get("gps_signal")),
            raw=row,
        )

