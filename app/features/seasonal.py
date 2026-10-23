"""Location-specific seasonal migration curves.

Each species/bait has a day-of-year curve calibrated for south-shore Long Island
(reference latitude 40.7 N). The curve is shifted for the spot's latitude and
water body: spring arrivals come later and fall departures earlier further north;
bays warm earlier in spring; Long Island Sound holds fish later into the fall.
Water temperature is scored separately and is the stronger real-time signal.
"""
from __future__ import annotations

import numpy as np

from app.core.config import species_config


def curve_value(points: list[list[float]], doy: np.ndarray | float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    # wrap the year so Dec->Jan interpolates smoothly
    xx = np.concatenate([x - 365, x, x + 365])
    yy = np.concatenate([y, y, y])
    return np.interp(np.asarray(doy, dtype=float), xx, yy)


def seasonal_factor(points: list[list[float]], doy: np.ndarray, lat: float, water_body: str) -> np.ndarray:
    """Season score 0-1 for a spot.

    * latitude: north of the reference, spring runs arrive later and fall runs
      leave earlier (evaluate the reference curve at an earlier / later date).
    * water body: a positive shift (Sound) delays the calendar; a negative one
      (bays) advances it — bays warm sooner in spring and cool sooner in fall.
    """
    doy = np.asarray(doy, dtype=float)
    cfg = species_config()
    lat_shift = (lat - cfg.get("reference_latitude", 40.7)) * cfg.get("latitude_shift_days_per_degree", 6)
    wb = cfg.get("water_body_shift_days", {}).get(water_body, 0)
    spring = doy < 200
    eff = np.where(spring, doy - lat_shift, doy + lat_shift) - wb
    return np.clip(curve_value(points, eff), 0, 1)


def trapezoid(x, a: float, b: float, c: float, d: float, floor: float = 0.0) -> np.ndarray:
    """1 between b and c, linear ramps to 0 at a and d. NaN stays NaN."""
    x = np.asarray(x, dtype=float)
    up = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
    down = np.clip((d - x) / max(d - c, 1e-6), 0, 1)
    out = np.minimum(up, down)
    out = np.where(np.isnan(x), np.nan, out)
    return np.maximum(out, floor) if floor else out
