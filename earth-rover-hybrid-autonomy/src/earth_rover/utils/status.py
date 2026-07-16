from __future__ import annotations

import math
from typing import Any


def format_urban_status(status: dict[str, Any]) -> str:
    return (
        "[URBAN] "
        f"mode={_text(status.get('mode'))} "
        f"target={_text(status.get('target_checkpoint_sequence'))} "
        f"dist={_fmt(status.get('distance_to_checkpoint_m'), '.1f')}m "
        f"bearing={_fmt(status.get('target_bearing_deg'), '.1f')} "
        f"heading={_fmt(status.get('current_heading_deg'), '.1f')} "
        f"err={_fmt(status.get('heading_error_deg'), '.1f')} | "
        f"raw=({_fmt(status.get('raw_linear'), '.2f')},{_fmt(status.get('raw_angular'), '.2f')}) "
        f"safe=({_fmt(status.get('safe_linear'), '.2f')},{_fmt(status.get('safe_angular'), '.2f')}) | "
        f"gps={_fmt(status.get('gps_signal'), '.1f')} "
        f"signal={_fmt(status.get('signal_level'), '.1f')} "
        f"data_age={_fmt(status.get('data_age_sec'), '.2f')}s "
        f"frame_age={_fmt(status.get('frame_age_sec'), '.2f')}s "
        f"stuck={_text(status.get('stuck_state'))} "
        f"recovery={_text(status.get('recovery_state'))} "
        f"log={_text(status.get('log_dir'))}"
    )


def _fmt(value: Any, spec: str) -> str:
    if value is None:
        return "None"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return format(number, spec)


def _text(value: Any) -> str:
    if value is None or value == "":
        return "None"
    return str(value)
