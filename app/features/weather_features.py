"""Derived weather features: pressure tendency, onshore wind, thunder flags.

Nearshore surf lives in app/features/surf.py and app/services/surf.py.
"""
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


def clarity_index(df: pd.DataFrame, sheltered: bool) -> pd.Series:
    """Water-clarity PROXY, 0 (clear) .. 1 (very dirty) — a heuristic, not a measurement.

    Suspended sand and runoff come mainly from (a) the last day of surf stirring
    the bottom, (b) the last two days of rain, (c) storms. Bays and the Sound use
    a lower surf scale because their wind chop is smaller.
    """
    surf = df.get("surf_ft", pd.Series(np.nan, index=df.index)).astype(float)
    surf24 = surf.rolling(24, min_periods=6).mean()
    lo, hi = (0.5, 2.5) if sheltered else (1.0, 6.0)
    s = ((surf24 - lo) / (hi - lo)).clip(0, 1)
    rain = df.get("precip_in", pd.Series(0.0, index=df.index)).astype(float).fillna(0).rolling(48, min_periods=1).sum()
    r = (rain / 1.0).clip(0, 1)
    phase = df.get("storm_phase", pd.Series("none", index=df.index)).astype(str)
    since = df.get("hours_since_storm", pd.Series(np.nan, index=df.index)).astype(float)
    st = np.where(phase == "during", 0.3, np.where((phase == "post") & (since.fillna(99) < 24), 0.2, 0.0))
    out = (0.6 * s.fillna(0) + 0.3 * r + st).clip(0, 1)
    return out.where(surf24.notna() | (rain > 0))


def weather_flags(df: pd.DataFrame) -> pd.Series:
    """Thunder / heavy rain flags from NWS text and WMO weather codes."""
    txt = df.get("short_forecast", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    code = df.get("weather_code", pd.Series(np.nan, index=df.index)).astype(float)
    thunder = txt.str.contains("thunder") | code.isin([95, 96, 99])
    return thunder
