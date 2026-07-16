from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_000.0


def haversine_distance_m(lat1, lon1, lat2, lon2) -> float:
    lat1_rad = math.radians(float(lat1))
    lat2_rad = math.radians(float(lat2))
    delta_lat = math.radians(float(lat2) - float(lat1))
    delta_lon = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    lat1_rad = math.radians(float(lat1))
    lat2_rad = math.radians(float(lat2))
    delta_lon = math.radians(float(lon2) - float(lon1))
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = (
        math.cos(lat1_rad) * math.sin(lat2_rad)
        - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    )
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def normalize_angle_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def normalize_angle_rad(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

