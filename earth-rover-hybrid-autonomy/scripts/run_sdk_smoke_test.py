#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from earth_rover.core.types import ControlCommand
from earth_rover.sdk_client import EarthRoverSDKClient
from earth_rover.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--start-mission", action="store_true", help="Call POST /start-mission before reading data.")
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--no-motion", action="store_true", default=True)
    motion.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    sdk_cfg = config["sdk"]
    sdk = EarthRoverSDKClient(sdk_cfg["base_url"], float(sdk_cfg["request_timeout_sec"]))

    if args.start_mission:
        sdk.start_mission()
        print("mission started")

    data = sdk.get_data()
    print(f"data latitude={data.latitude} longitude={data.longitude} orientation={data.orientation} speed={data.speed}")
    frame = sdk.get_front_frame()
    print(f"front frame shape={frame.image.shape}")

    sdk.send_control(ControlCommand(0.0, 0.0, mode="SMOKE_STOP"))
    if args.allow_motion:
        sdk.send_control(ControlCommand(0.05, 0.0, mode="SMOKE_MOTION"))
        time.sleep(1.0)
    sdk.send_control(ControlCommand(0.0, 0.0, mode="SMOKE_STOP"))
    print("smoke test complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
