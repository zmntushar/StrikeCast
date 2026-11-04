"""Tidal-current conditions at any time resolution.

Tidal height and tidal-current velocity are different physical variables:
high/low water rarely coincide with slack current, especially near inlets.
So current speed, direction and slack times come ONLY from NOAA current-station
predictions (max flood / max ebb / slack events). When no station exists, an
approximate *tidal-phase* flow index (0-1, no knots) is derived from the tide
curve and clearly labeled, and forecast confidence is reduced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.tide_features import current_features
from app.schemas.env import CurrentEvent

# source labels
STATION = "station"                  # NOAA current prediction station at/near the spot
PROXY_STATION = "proxy_station"      # nearby station used as a timing/strength proxy
TIDE_PHASE = "tide_phase_approx"     # no current data: approximate index from tide phase
MISSING = "missing"

SOURCE_TEXT = {
    STATION: "NOAA current predictions",
    PROXY_STATION: "NOAA current predictions from a nearby station (proxy for this shoreline)",
    TIDE_PHASE: "approximate tidal-phase flow index (no current station — not a velocity)",
    MISSING: "no tide or current data",
}


def current_series(grid: pd.DatetimeIndex, events: list[CurrentEvent], tide: pd.DataFrame | None,
                   is_proxy: bool) -> tuple[pd.DataFrame, str]:
    """current_kt (signed, + flood), current_norm (share of local peak 0-1), current_phase,
    current_dir (deg toward) on ``grid``; plus the source label."""
    out = pd.DataFrame(index=grid, columns=["current_kt", "current_norm", "current_phase", "current_dir"], dtype=object)
    if len(events) >= 2:
        cf = current_features(events, grid)
        kt = cf["current_kt"].astype(float)
        if kt.notna().sum() >= max(3, len(grid) // 4):
            flood = next((e.direction_deg for e in events if e.velocity_kt > 0 and e.direction_deg is not None), None)
            ebb = next((e.direction_deg for e in events if e.velocity_kt < 0 and e.direction_deg is not None), None)
            out["current_kt"] = kt
            out["current_norm"] = cf["current_norm"].astype(float)
            out["current_phase"] = cf["current_phase"]
            out["current_dir"] = np.where(kt > 0, flood if flood is not None else np.nan,
                                          np.where(kt < 0, ebb if ebb is not None else np.nan, np.nan))
            return _floats(out), PROXY_STATION if is_proxy else STATION
    if tide is not None and "tide_frac" in tide and tide["tide_frac"].notna().any():
        f = tide["tide_frac"].astype(float)
        # The cosine tide model's rate of change is a half-sine between turns. This is a
        # flow *timing* proxy only: real slack often lags high/low water by an hour or more.
        out["current_norm"] = np.abs(np.sin(np.pi * f))
        out["current_kt"] = np.nan
        out["current_dir"] = np.nan
        out["current_phase"] = np.where(tide["tide_dir"] == "incoming", "flood (est.)",
                                        np.where(tide["tide_dir"] == "outgoing", "ebb (est.)", None))
        return _floats(out), TIDE_PHASE
    return _floats(out), MISSING


def slack_times(events: list[CurrentEvent], start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Official predicted slack-water times in [start, end] (empty without a current station)."""
    return [pd.Timestamp(e.time) for e in events
            if e.type == "slack" and start <= pd.Timestamp(e.time) <= end]


def _floats(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("current_kt", "current_norm", "current_dir"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df
