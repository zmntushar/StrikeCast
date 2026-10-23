"""Small geodesy + unit helpers (no heavy GIS dependency needed for these)."""
from __future__ import annotations

import math

import numpy as np

EARTH_R_KM = 6371.0088

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def km_to_mi(km: float) -> float:
    return km * 0.621371


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def destination(lat: float, lon: float, bearing: float, dist_km: float) -> tuple[float, float]:
    d = dist_km / EARTH_R_KM
    b = math.radians(bearing)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a-b in degrees, range (-180, 180]."""
    d = (a - b + 180) % 360 - 180
    return 180.0 if d == -180 else d


def compass(deg: float | None) -> str:
    if deg is None or (isinstance(deg, float) and math.isnan(deg)):
        return "—"
    return COMPASS[int(((deg % 360) + 11.25) // 22.5) % 16]


def compass_to_deg(text: str | None) -> float | None:
    if not text:
        return None
    t = text.strip().upper()
    if t in COMPASS:
        return COMPASS.index(t) * 22.5
    return None


def wind_components(speed, from_deg, facing_deg: float):
    """Split wind into onshore (+) / offshore (-) and alongshore parts.

    ``from_deg`` is the meteorological direction the wind blows FROM.
    ``facing_deg`` points from shore toward the water, so wind coming FROM the
    facing direction blows onshore. Works on scalars or numpy arrays.
    """
    theta = np.radians(np.asarray(from_deg, dtype=float) - facing_deg)
    spd = np.asarray(speed, dtype=float)
    return spd * np.cos(theta), spd * np.sin(theta)


# ----------------------------------------------------------------- units
def c_to_f(c):
    return np.asarray(c, dtype=float) * 9 / 5 + 32 if c is not None else None


def m_to_ft(m):
    return np.asarray(m, dtype=float) * 3.28084


def ms_to_mph(v):
    return np.asarray(v, dtype=float) * 2.236936


def kmh_to_kt(v):
    return np.asarray(v, dtype=float) * 0.539957


def hpa_to_inhg(p):
    return np.asarray(p, dtype=float) * 0.0295300
