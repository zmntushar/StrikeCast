"""Storm / nor'easter detection and storm-phase features.

A storm hour is flagged from combinations of sustained wind, offshore wave
height, rapid pressure fall and coastal water-level anomaly (surge). Effects on
fishing are NOT hard-coded as negative: each species defines configurable
pre / during / post multipliers in configs/species.yaml.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import species_config


def detect_storms(df: pd.DataFrame) -> pd.DataFrame:
    cfg = species_config().get("storm", {})
    wind_thr = cfg.get("wind_mph", 25)
    min_h = int(cfg.get("min_hours", 3))
    wave_thr = cfg.get("offshore_wave_ft", 8)
    p_drop = cfg.get("pressure_drop_12h_hpa", -8)
    post_h = cfg.get("post_window_hours", 72)
    pre_h = cfg.get("pre_window_hours", 12)

    wind = df.get("wind_mph", pd.Series(np.nan, index=df.index)).astype(float)
    wave = df.get("wave_ft", pd.Series(np.nan, index=df.index)).astype(float)
    ptend = df.get("pressure_tend_12h", pd.Series(np.nan, index=df.index)).astype(float)
    anom = df.get("water_level_anom_ft", pd.Series(np.nan, index=df.index)).astype(float)
    wdir = df.get("wind_dir", pd.Series(np.nan, index=df.index)).astype(float)

    sustained = (wind >= wind_thr).astype(int).rolling(min_h, min_periods=min_h).sum().ge(min_h)
    sustained = sustained | sustained.shift(-(min_h - 1), fill_value=False)
    buoy = df.get("buoy_wave_ft", pd.Series(np.nan, index=df.index)).astype(float)
    big_seas = (wave >= wave_thr) | (buoy >= wave_thr + 1)
    bomb = (ptend <= p_drop) & (wind >= wind_thr * 0.8)
    surge = (anom >= 1.5) & (wind >= 18)
    storm = (sustained | big_seas | bomb | surge).fillna(False)

    # intensity 0-1: combination of normalized drivers
    drivers = np.vstack([((wind - 15) / 25).to_numpy(), ((wave - 4) / 10).to_numpy(),
                         (-(ptend) / 16).to_numpy(), (anom / 3).to_numpy()])
    intensity = np.clip(np.max(np.nan_to_num(drivers, nan=0.0), axis=0), 0, 1)
    intensity = np.where(storm, intensity, 0.0)
    noreaster = storm & wdir.between(0, 100)

    hours = np.arange(len(df))
    storm_idx = hours[storm.to_numpy()]
    since_end = np.full(len(df), np.nan)
    until_start = np.full(len(df), np.nan)
    for n in range(len(df)):
        past = storm_idx[storm_idx <= n]
        fut = storm_idx[storm_idx >= n]
        if len(past) and not storm.iloc[n]:
            since_end[n] = n - past[-1]
        if len(fut) and not storm.iloc[n]:
            until_start[n] = fut[0] - n
    phase = np.where(storm, "during", "none").astype(object)
    phase[(~storm.to_numpy()) & (since_end <= post_h)] = "post"
    phase[(~storm.to_numpy()) & (until_start <= pre_h) & (phase == "none")] = "pre"

    # wave trend over 6 h — falling seas after a blow = recovery
    wave_trend = wave - wave.shift(6)
    return pd.DataFrame({
        "storm_active": storm.to_numpy(), "storm_phase": phase, "storm_intensity": intensity,
        "noreaster": noreaster.to_numpy(), "hours_since_storm": since_end,
        "hours_until_storm": until_start, "wave_trend_6h": wave_trend.to_numpy(),
    }, index=df.index)


def storm_multiplier(phase: np.ndarray, hours_since: np.ndarray, effects: dict) -> np.ndarray:
    """Species multiplier. Post-storm effect fades out over 12-72 h after the storm."""
    out = np.ones(len(phase))
    out[phase == "pre"] = effects.get("pre", 1.0)
    out[phase == "during"] = effects.get("during", 1.0)
    post = phase == "post"
    if post.any():
        h = np.nan_to_num(hours_since[post], nan=72.0)
        during, post_eff = effects.get("during", 1.0), effects.get("post", 1.0)
        # 0-12 h after: conditions recover from 'during' toward the post-storm effect;
        # 12-72 h: post-storm effect fades back to neutral.
        recovering = during + (post_eff - during) * np.clip(h / 12, 0, 1)
        fading = 1 + (post_eff - 1) * np.clip((72 - h) / 60, 0, 1)
        out[post] = np.where(h < 12, recovering, fading)
    return out
