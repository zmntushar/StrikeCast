"""Species-specific environmental suitability at 5-minute resolution.

Component scores (all 0-1, 1 = most favorable):

* solar       — continuous function of solar elevation (species curve, overcast adjustment)
* current     — tidal-current strength vs the species' preferred share of peak flow
                (NOAA current predictions; tidal-phase approximation when unavailable)
* tide        — tide phase & direction (incoming / outgoing / near the turn) per spot type
* habitat     — spot structure affinity (rocks, jetty, inlet, ...)
* seasonal    — migration calendar
* temperature — water temperature vs the species' range
* conditions  — wind, surf, pressure, water clarity, bait (the hourly feeding factors)

The total is a weighted mean of the components available (missing ones are
dropped and the weights renormalized, never filled with invented values), then
multiplied by optional "limiting" terms and by shore fishability (safety).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features import current_model
from app.features.solar import solar_elevation
from app.features.tide_features import tide_features
from app.models.feeding import WAVE_WEIGHT_BY_EXPOSURE, current_factor, tide_stage_factor
from app.models.feeding_profiles import COMPONENTS, FeedingProfile, solar_suitability
from app.schemas.env import CurrentEvent, TideEvent


def hourly_to_grid(s: pd.Series | None, grid: pd.DatetimeIndex, carry_samples: int = 24) -> np.ndarray:
    """Time-interpolate an hourly series onto the 5-min grid; short carry past either end."""
    if s is None:
        return np.full(len(grid), np.nan)
    s = pd.to_numeric(s, errors="coerce").astype(float).dropna()
    if s.empty:
        return np.full(len(grid), np.nan)
    s = s[~s.index.duplicated()]
    u = s.reindex(s.index.union(grid)).interpolate("time", limit_area="inside")
    u = u.ffill(limit=carry_samples).bfill(limit=carry_samples)
    return u.reindex(grid).to_numpy(dtype=float)


def conditions_hourly(hourly: pd.DataFrame, species: str, profile: FeedingProfile, exposure_class: str) -> pd.Series | None:
    """Weighted mean of the hourly environmental feeding factors (wind, waves, pressure, bait, clarity...)."""
    num = pd.Series(0.0, index=hourly.index)
    den = pd.Series(0.0, index=hourly.index)
    used = False
    for name, w in profile.condition_weights.items():
        col = f"ff_{species}_{name}"
        if col not in hourly:
            continue
        if name == "waves":
            w *= WAVE_WEIGHT_BY_EXPOSURE.get(exposure_class, 1.0)
        v = pd.to_numeric(hourly[col], errors="coerce")
        ok = v.notna()
        num += np.where(ok, w * v.fillna(0), 0)
        den += np.where(ok, w, 0)
        used = True
    if not used:
        return None
    return (num / den.where(den > 0)).astype(float)


def component_frame(grid: pd.DatetimeIndex, *, lat: float, lon: float, environment: str, exposure_class: str,
                    species: str, profile: FeedingProfile, tide_events: list[TideEvent],
                    current_events: list[CurrentEvent], current_is_proxy: bool,
                    hourly: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    """All raw drivers + component scores + total on ``grid`` (UTC). Returns (frame, current source)."""
    hourly = hourly if hourly is not None else pd.DataFrame()
    hcol = lambda c: hourly[c] if c in hourly else None  # noqa: E731
    df = pd.DataFrame(index=grid)

    # ---- solar
    df["sun_alt"] = solar_elevation(lat, lon, grid)
    df["cloud_pct"] = hourly_to_grid(hcol("cloud_pct"), grid)
    df["solar"] = solar_suitability(profile, df["sun_alt"].to_numpy(), df["cloud_pct"].to_numpy())

    # ---- tide (height & phase) — from official high/low predictions
    tide = tide_features(tide_events, grid)
    for c in ("tide_ft", "tide_rate_ft_h", "tide_frac", "hours_since_turn", "hours_to_turn"):
        df[c] = pd.to_numeric(tide[c], errors="coerce").astype(float)
    for c in ("tide_dir", "tide_stage", "tide_stage_tag", "next_turn_type"):
        df[c] = tide[c]
    df["tide"] = tide_stage_factor(tide["tide_stage_tag"].to_numpy(), df["hours_since_turn"].to_numpy(),
                                   df["hours_to_turn"].to_numpy(), profile.tide_prefs(environment))

    # ---- current — from current predictions only (separate physical variable)
    cur, source = current_model.current_series(grid, current_events, tide, current_is_proxy)
    df = df.join(cur)
    lo, hi = profile.current_opt
    cscore = current_factor(df["current_norm"].to_numpy(), lo, hi, profile.slack_score)
    cap = profile.manageable_current_kt
    if cap and source == current_model.STATION:
        over = np.clip((np.abs(df["current_kt"].to_numpy()) - cap) / cap, 0, 1)
        cscore = cscore * (1 - 0.6 * np.nan_to_num(over))
    df["current"] = cscore

    # ---- slowly varying context from the hourly forecast
    df["habitat"] = hourly_to_grid(hcol(f"pf_{species}_structure"), grid)
    df["seasonal"] = hourly_to_grid(hcol(f"pf_{species}_season"), grid)
    df["temperature"] = hourly_to_grid(hcol(f"pf_{species}_temp"), grid)
    df["water_temp_f"] = hourly_to_grid(hcol("water_temp_f"), grid)
    cond = conditions_hourly(hourly, species, profile, exposure_class) if not hourly.empty else None
    df["conditions"] = hourly_to_grid(cond, grid)
    fish = hourly_to_grid(hcol(f"fishability_{species}"), grid)
    df["fishability"] = np.where(np.isnan(fish), 1.0, fish)

    df["total"] = total_score(df, profile)
    return df, source


def total_score(df: pd.DataFrame, profile: FeedingProfile) -> np.ndarray:
    """0-100 suitability for every grid row (weighted mean of available components x limits x fishability)."""
    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for c in COMPONENTS:
        w = profile.weights.get(c, 0.0)
        if w <= 0 or c not in df:
            continue
        v = df[c].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        num += np.where(ok, w * v, 0)
        den += np.where(ok, w, 0)
    base = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    for c, ex in profile.limiting.items():
        if c in df:
            v = df[c].to_numpy(dtype=float)
            base = np.where(np.isnan(v), base, base * np.clip(v, 0.01, 1) ** ex)
    return np.clip(100 * base * df["fishability"].to_numpy(dtype=float), 0, 100)
