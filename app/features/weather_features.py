"""Derived weather/ocean features: pressure tendency, onshore wind, nearshore surf."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.geo import wind_components


def pressure_tendency(pressure: pd.Series) -> pd.DataFrame:
    p = pressure.astype(float)
    return pd.DataFrame({f"pressure_tend_{h}h": p - p.shift(h) for h in (3, 6, 12, 24)}, index=p.index)


def wind_features(wind_mph: pd.Series, wind_dir: pd.Series, facing_deg: float) -> pd.DataFrame:
    on, along = wind_components(wind_mph.to_numpy(), wind_dir.to_numpy(), facing_deg)
    on_s = pd.Series(on, index=wind_mph.index)
    onshore_hours = (on_s > 6).astype(float).rolling(12, min_periods=1).sum()
    offshore_hours = (on_s < -6).astype(float).rolling(12, min_periods=1).sum()
    return pd.DataFrame({"onshore_mph": on, "alongshore_mph": along,
                         "onshore_hours_12h": onshore_hours, "offshore_hours_12h": offshore_hours},
                        index=wind_mph.index)


def nearshore_waves(offshore_ft: pd.Series, wave_dir: pd.Series, onshore_mph: pd.Series,
                    facing_deg: float, exposure: float, water_body: str) -> pd.Series:
    """Heuristic nearshore wave height (ft).

    Model waves ~1 km off the spot are scaled by the spot's exposure and by how
    squarely they arrive relative to the shore normal (cos²(θ/2): a simple stand-in
    for refraction and directional spreading). Sheltered bay
    and Sound spots also get a fetch-limited local wind chop estimate.
    """
    off = offshore_ft.astype(float)
    deg_off = np.abs(((wave_dir.astype(float) - facing_deg + 180) % 360) - 180)
    # Smooth exposure curve: 1.0 head-on, 0.75 at 60°, 0.5 at 90° (swell running along the
    # beach still wraps in via refraction + directional spread), ~0.07 at 150°, 0 heading away.
    alignment = np.cos(np.radians(deg_off) / 2) ** 2
    swell_part = off * exposure * np.where(np.isfinite(deg_off), alignment, 0.6)
    on = onshore_mph.astype(float).clip(lower=0)
    chop_coef = {"sound": 0.012, "bay": 0.006}.get(water_body, 0.0)
    chop = chop_coef * on ** 1.6
    return pd.Series(np.fmax(swell_part.to_numpy(), chop.to_numpy()), index=off.index)


def weather_flags(df: pd.DataFrame) -> pd.Series:
    """Thunder / heavy rain flags from NWS text and WMO weather codes."""
    txt = df.get("short_forecast", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    code = df.get("weather_code", pd.Series(np.nan, index=df.index)).astype(float)
    thunder = txt.str.contains("thunder") | code.isin([95, 96, 99])
    return thunder
