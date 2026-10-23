"""Sun & moon astronomy computed locally with PyEphem (no network needed).

Provides hourly solar/lunar elevation and illumination plus daily events:
sunrise/sunset, civil twilight, moonrise/moonset, lunar upper transit, moon
phase and spring/neap proximity. Moon phase is kept separate from measured
tidal range on purpose: phase alone does not set local tidal strength.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import ephem
import numpy as np
import pandas as pd

SYNODIC_DAYS = 29.530588


def _observer(lat: float, lon: float, when: datetime, horizon: str = "0") -> ephem.Observer:
    obs = ephem.Observer()
    obs.lat, obs.lon = str(lat), str(lon)
    obs.elevation = 2
    obs.pressure = 0  # disable refraction model for altitude series; horizon handles it
    obs.horizon = horizon
    obs.date = ephem.Date(when.astimezone(timezone.utc).replace(tzinfo=None))
    return obs


def _to_dt(d: ephem.Date) -> datetime:
    return d.datetime().replace(tzinfo=timezone.utc)


def hourly_astro(lat: float, lon: float, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sun altitude, moon altitude and illumination for each UTC timestamp."""
    obs = _observer(lat, lon, index[0].to_pydatetime())
    sun, moon = ephem.Sun(), ephem.Moon()
    sun_alt, moon_alt, illum = [], [], []
    for ts in index:
        obs.date = ephem.Date(ts.tz_convert("UTC").to_pydatetime().replace(tzinfo=None))
        sun.compute(obs)
        moon.compute(obs)
        sun_alt.append(math.degrees(sun.alt))
        moon_alt.append(math.degrees(moon.alt))
        illum.append(moon.phase)
    return pd.DataFrame({"sun_alt": sun_alt, "moon_alt": moon_alt, "moon_illum": illum}, index=index)


def moon_phase_info(when: datetime) -> dict:
    d = ephem.Date(when.astimezone(timezone.utc).replace(tzinfo=None))
    prev_new = ephem.previous_new_moon(d)
    next_new = ephem.next_new_moon(d)
    prev_full = ephem.previous_full_moon(d)
    next_full = ephem.next_full_moon(d)
    age = float(d - prev_new)
    frac = age / SYNODIC_DAYS
    names = [(0.033, "New Moon"), (0.216, "Waxing Crescent"), (0.283, "First Quarter"),
             (0.466, "Waxing Gibbous"), (0.533, "Full Moon"), (0.716, "Waning Gibbous"),
             (0.783, "Last Quarter"), (0.966, "Waning Crescent"), (1.01, "New Moon")]
    name = next(n for lim, n in names if frac <= lim)
    days_to_syzygy = min(abs(float(d - prev_new)), abs(float(next_new - d)),
                         abs(float(d - prev_full)), abs(float(next_full - d)))
    # 1.0 at new/full (spring tides), 0.0 at quarter moons (neap tides).
    spring_index = 0.5 * (1 + math.cos(math.pi * min(days_to_syzygy, SYNODIC_DAYS / 4) / (SYNODIC_DAYS / 4)))
    moon = ephem.Moon(d)
    return {
        "phase_name": name, "age_days": round(age, 1), "illumination_pct": round(moon.phase, 1),
        "days_to_spring": round(days_to_syzygy, 1), "spring_index": round(spring_index, 2),
        "next_new": _to_dt(next_new), "next_full": _to_dt(next_full),
        "tide_cycle": "spring" if spring_index > 0.66 else ("neap" if spring_index < 0.33 else "mid"),
    }


def daily_events(lat: float, lon: float, day: date, tz: str) -> dict:
    """Rise/set/twilight/transit for a LOCAL calendar day (returned as UTC datetimes)."""
    local_midnight = pd.Timestamp(day).tz_localize(tz)
    start = local_midnight.tz_convert("UTC").to_pydatetime()

    def _event(fn, body, horizon="0", use_center=False):
        obs = _observer(lat, lon, start, horizon)
        obs.pressure = 1010
        try:
            t = getattr(obs, fn)(body, use_center=use_center) if fn != "next_transit" else obs.next_transit(body)
            dt = _to_dt(t)
            return dt if dt < start + timedelta(days=1) else None
        except (ephem.AlwaysUpError, ephem.NeverUpError):
            return None

    sun, moon = ephem.Sun(), ephem.Moon()
    return {
        "date": day,
        "civil_dawn": _event("next_rising", sun, "-6", True),
        "sunrise": _event("next_rising", sun, "-0:34"),
        "sunset": _event("next_setting", sun, "-0:34"),
        "civil_dusk": _event("next_setting", sun, "-6", True),
        "moonrise": _event("next_rising", moon),
        "moonset": _event("next_setting", moon),
        "moon_transit": _event("next_transit", moon),
        **{k: v for k, v in moon_phase_info(start + timedelta(hours=12)).items()
           if k in ("phase_name", "illumination_pct", "spring_index", "tide_cycle", "age_days")},
    }


def light_phase(sun_alt: np.ndarray, sun_alt_prev: np.ndarray) -> np.ndarray:
    """night / dawn / day / dusk from solar elevation (+ whether the sun is rising).

    dawn/dusk = sun between -12 and +8 degrees (nautical twilight to ~1 h after
    sunrise / before sunset): the classic low-light feeding window.
    """
    sun_alt = np.asarray(sun_alt, dtype=float)
    rising = np.asarray(sun_alt) >= np.asarray(sun_alt_prev)
    out = np.where(sun_alt < -12, "night", "day").astype(object)
    low = (sun_alt >= -12) & (sun_alt < 8)
    out[low & rising] = "dawn"
    out[low & ~rising] = "dusk"
    return out
