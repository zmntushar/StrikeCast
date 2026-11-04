"""Full solar cycle for a place and local date, computed locally with PyEphem.

Provides every twilight boundary (astronomical, nautical, civil), sunrise,
sunset, solar noon and a continuous solar-elevation series at any resolution.
Feeding models use the *elevation* rather than clock times so that seasonal
changes in day length and light intensity are represented automatically.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import ephem
import numpy as np
import pandas as pd

# Standard solar-elevation thresholds (degrees). Sunrise/sunset use the upper limb
# with refraction (-0.833 deg for the centre), twilights use the centre of the disk.
SUNRISE_DEG = -0.833
CIVIL_DEG = -6.0
NAUTICAL_DEG = -12.0
ASTRONOMICAL_DEG = -18.0

# (lower bound deg, band name) in ascending order; used for labels and chart shading.
BANDS = [(-90.0, "night"), (ASTRONOMICAL_DEG, "astronomical twilight"), (NAUTICAL_DEG, "nautical twilight"),
         (CIVIL_DEG, "civil twilight"), (SUNRISE_DEG, "daylight")]


def _obs(lat: float, lon: float, when_utc: datetime, horizon_deg: float = 0.0, refraction: bool = False) -> ephem.Observer:
    o = ephem.Observer()
    o.lat, o.lon = str(lat), str(lon)
    o.elevation = 2
    o.pressure = 1010 if refraction else 0
    o.horizon = str(horizon_deg)
    o.date = ephem.Date(when_utc.astimezone(timezone.utc).replace(tzinfo=None))
    return o


def _utc(d: ephem.Date) -> datetime:
    return d.datetime().replace(tzinfo=timezone.utc)


def solar_elevation(lat: float, lon: float, index: pd.DatetimeIndex) -> np.ndarray:
    """Geometric solar elevation (degrees) for each timestamp (tz-aware)."""
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return np.array([], dtype=float)
    obs = _obs(lat, lon, idx[0].to_pydatetime())
    sun = ephem.Sun()
    out = np.empty(len(idx))
    for i, ts in enumerate(idx.tz_convert("UTC")):
        obs.date = ephem.Date(ts.to_pydatetime().replace(tzinfo=None))
        sun.compute(obs)
        out[i] = math.degrees(sun.alt)
    return out


def solar_events(lat: float, lon: float, day: date, tz: str) -> dict[str, datetime | None]:
    """All solar events for a LOCAL calendar day as UTC datetimes (None when the sun never crosses)."""
    start = pd.Timestamp(day).tz_localize(tz).tz_convert("UTC").to_pydatetime()
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC").to_pydatetime()
    sun = ephem.Sun()

    def ev(fn: str, deg: float, center: bool) -> datetime | None:
        o = _obs(lat, lon, start, deg if center else -0.5667, refraction=not center)
        try:
            t = _utc(getattr(o, fn)(sun, use_center=center))
        except (ephem.AlwaysUpError, ephem.NeverUpError):
            return None
        return t if t < end else None

    noon_o = _obs(lat, lon, start)
    noon = _utc(noon_o.next_transit(sun))
    return {
        "astronomical_dawn": ev("next_rising", ASTRONOMICAL_DEG, True),
        "nautical_dawn": ev("next_rising", NAUTICAL_DEG, True),
        "civil_dawn": ev("next_rising", CIVIL_DEG, True),
        "sunrise": ev("next_rising", 0.0, False),
        "solar_noon": noon if noon < end else None,
        "sunset": ev("next_setting", 0.0, False),
        "civil_dusk": ev("next_setting", CIVIL_DEG, True),
        "nautical_dusk": ev("next_setting", NAUTICAL_DEG, True),
        "astronomical_dusk": ev("next_setting", ASTRONOMICAL_DEG, True),
    }


def day_length_hours(events: dict[str, datetime | None]) -> float | None:
    a, b = events.get("sunrise"), events.get("sunset")
    return (b - a).total_seconds() / 3600 if a and b else None


def light_band(elev: np.ndarray | float) -> np.ndarray:
    """Band name for each elevation: night / astronomical / nautical / civil twilight / daylight."""
    e = np.atleast_1d(np.asarray(elev, dtype=float))
    out = np.full(len(e), "night", dtype=object)
    for lo, name in BANDS:
        out[e >= lo] = name
    return out


def band_segments(index: pd.DatetimeIndex, elev: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Contiguous (start, end, band) runs, coarse classes night / twilight / daylight, for chart shading."""
    coarse = np.where(np.asarray(elev) < ASTRONOMICAL_DEG, "night",
                      np.where(np.asarray(elev) < SUNRISE_DEG, "twilight", "daylight"))
    segs: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if len(index) == 0:
        return segs
    s = 0
    for i in range(1, len(index) + 1):
        if i == len(index) or coarse[i] != coarse[s]:
            end = index[i] if i < len(index) else index[-1] + (index[-1] - index[-2] if len(index) > 1 else timedelta(minutes=5))
            segs.append((index[s], end, str(coarse[s])))
            s = i
    return segs
