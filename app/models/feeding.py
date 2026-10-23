"""Hourly Feeding Activity Index, 0-100, per species.

Captures *when* fish that are present are likely to feed: moving water, low
light, wind and surf, pressure trends, bait concentration and temperature trend.
It is distinct from catch probability, which also depends on lure choice,
casting reach, angler skill and effort.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.common import col, ramp_band, species_meta
from app.models.storms import storm_multiplier
from app.schemas.env import Spot

FACTORS = ["current", "tide_stage", "light", "wind", "waves", "pressure", "bait", "temp_trend"]


def current_factor(cn: np.ndarray, lo: float, hi: float, slack: float) -> np.ndarray:
    cn = np.asarray(cn, dtype=float)
    below = slack + (1 - slack) * np.clip(cn / max(lo, 1e-6), 0, 1)
    above = 1 - 0.55 * np.clip((cn - hi) / max(1.0 - hi, 0.05), 0, 1)
    out = np.where(cn < lo, below, np.where(cn <= hi, 1.0, above))
    return np.where(np.isnan(cn), np.nan, out)


def light_factor(phase: np.ndarray, cloud: np.ndarray, light: dict, overcast_bonus: float) -> np.ndarray:
    base = np.array([light.get(p, 0.6) if isinstance(p, str) else 0.6 for p in phase], dtype=float)
    day = np.array([p == "day" for p in phase])
    bonus = overcast_bonus * np.nan_to_num(cloud, nan=30) / 100
    return np.clip(np.where(day, base + bonus, base), 0, 1)


def tide_stage_factor(tags: np.ndarray, since: np.ndarray, until: np.ndarray, preferred: list[str]) -> np.ndarray:
    out = np.full(len(tags), 0.55)
    near_turn = (np.nan_to_num(since, nan=9) < 1.5) | (np.nan_to_num(until, nan=9) < 1.5)
    for n, t in enumerate(tags):
        if t is None or (isinstance(t, float) and np.isnan(t)):
            out[n] = np.nan
            continue
        hit = False
        for p in preferred:
            if p == "slack_window" and near_turn[n]:
                hit = True
            elif p in ("outgoing", "incoming") and str(t).endswith(p):
                hit = True
            elif p == t:
                hit = True
        out[n] = 1.0 if hit else 0.5
    return out


def wind_factor(onshore: np.ndarray, opt: list[float], water_body: str) -> np.ndarray:
    lo, hi = opt
    on = np.asarray(onshore, dtype=float)
    calm_val = 0.75
    out = np.where(on < lo, calm_val - 0.2 * np.clip((-on - 8) / 12, 0, 1),
                   np.where(on <= hi, 1.0, 1 - 0.7 * np.clip((on - hi) / 15, 0, 1)))
    if lo <= 0:  # species that like it calm: light winds from any direction are best
        out = np.where(on <= hi, 1.0 - 0.15 * np.clip((-on - 10) / 15, 0, 1), out)
    return np.where(np.isnan(on), np.nan, out)


def pressure_factor(t3: np.ndarray, falling_bonus: float, rising_penalty: float) -> np.ndarray:
    t3 = np.asarray(t3, dtype=float)
    out = np.where(t3 <= -0.6, 0.55 + np.clip(-t3 / 2.5, 0, 1) * falling_bonus * 1.8,
                   np.where(t3 >= 1.0, 0.55 - np.clip(t3 / 3, 0, 1) * rising_penalty * 2, 0.55))
    return np.where(np.isnan(t3), np.nan, np.clip(out, 0, 1))


def temp_trend_factor(trend_3d: np.ndarray, doy: np.ndarray) -> np.ndarray:
    sign = np.where(doy > 220, -1.0, 1.0)  # fall: cooling triggers feeding/migration
    tr = np.asarray(trend_3d, dtype=float)
    return np.where(np.isnan(tr), np.nan, np.clip(0.55 + 0.45 * np.clip(sign * tr / 4, -1, 1), 0, 1))


def feeding_scores(df: pd.DataFrame, spot: Spot, species: str, bait_av: np.ndarray | None) -> pd.DataFrame:
    cfg = species_meta(species)
    doy = df.index.tz_convert(spot.timezone).day_of_year.to_numpy()
    phase = df.get("light_phase", pd.Series("day", index=df.index)).to_numpy()
    tags = df.get("tide_stage_tag", pd.Series(None, index=df.index)).to_numpy()
    prefs = cfg.get("preferred_tide", {}).get(spot.environment, ["incoming", "outgoing"])
    wl = cfg["waves_ft"]

    f = {
        "current": current_factor(col(df, "current_norm"), *cfg["current_opt"], cfg["slack_score"]),
        "tide_stage": tide_stage_factor(tags, col(df, "hours_since_turn"), col(df, "hours_to_turn"), prefs),
        "light": light_factor(phase, col(df, "cloud_pct"), cfg["light"], cfg.get("overcast_day_bonus", 0)),
        "wind": wind_factor(col(df, "onshore_mph"), cfg["onshore_wind_opt_mph"], spot.water_body),
        "waves": ramp_band(col(df, "nearshore_wave_ft"), 0.65 if wl[1] > 0 else 1.0, max(wl[1], 1e-6),
                           wl[2], wl[3], 0.12),
        "pressure": pressure_factor(col(df, "pressure_tend_3h"), cfg["pressure"]["falling_bonus"],
                                    cfg["pressure"]["rising_penalty"]),
        "bait": bait_av if bait_av is not None else np.full(len(df), np.nan),
        "temp_trend": temp_trend_factor(col(df, "water_temp_trend_3d"), doy),
    }
    w = cfg["feeding_weights"]
    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for name in FACTORS:
        wt = w.get(name, 0)
        if wt <= 0:
            continue
        val = f[name]
        ok = ~np.isnan(val)
        num += np.where(ok, wt * val, 0)
        den += np.where(ok, wt, 0)
    base = np.where(den > 0, num / np.maximum(den, 1e-9), 0.5)
    storm = storm_multiplier(df.get("storm_phase", pd.Series("none", index=df.index)).to_numpy(),
                             col(df, "hours_since_storm"), cfg.get("storm", {}))
    # Convex stretch: only hours where most factors line up score near the top.
    feeding = np.clip(100 * np.clip(base, 0, 1) ** 1.4 * storm, 0, 100)
    out = {f"feeding_{species}": feeding, f"ff_{species}_storm": storm}
    for name in FACTORS:
        out[f"ff_{species}_{name}"] = f[name]
    return pd.DataFrame(out, index=df.index)
