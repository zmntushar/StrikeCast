"""Baitfish Habitat Suitability Index (BHSI), 0-100, per bait group.

This is an ENVIRONMENTAL SUITABILITY score, not a calibrated probability.
Different bait groups respond differently: e.g. peanut bunker and anchovies are
flushed out of bays on the outgoing tide in fall, sand eels favor sandy ocean
beaches with moving water, adult bunker get pushed against the beach by
sustained onshore wind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import species_config
from app.features.seasonal import seasonal_factor, trapezoid
from app.models.common import col, fill_neutral
from app.models.storms import storm_multiplier
from app.schemas.env import Spot

BAIT_STORM = {"pre": 1.0, "during": 0.6, "post": 0.9}


def bait_scores(df: pd.DataFrame, spot: Spot) -> pd.DataFrame:
    cfg = species_config()["baitfish"]
    doy = df.index.tz_convert(spot.timezone).day_of_year.to_numpy()
    water_t = col(df, "water_temp_f")
    onshore_h = col(df, "onshore_hours_12h", 0)
    offshore_h = col(df, "offshore_hours_12h", 0)
    cur = col(df, "current_norm")
    waves = col(df, "nearshore_wave_ft")
    tide_dir = df.get("tide_dir", pd.Series(None, index=df.index)).to_numpy()
    phase = df.get("storm_phase", pd.Series("none", index=df.index)).to_numpy()
    hsince = col(df, "hours_since_storm")

    out = {}
    for g, b in cfg.items():
        season = seasonal_factor(b["season"], doy, spot.lat, spot.water_body)
        temp = fill_neutral(trapezoid(water_t, *b["temp_f"], floor=0.03), 0.6)
        habitat = b["habitat"].get(spot.water_body, 0.6) * b["environment"].get(spot.environment, 0.7)
        push = b.get("wind_push", 0.5)
        wind = np.clip(0.55 + push * 0.45 * np.clip(onshore_h / 8, 0, 1) - 0.15 * np.clip(offshore_h / 10, 0, 1), 0, 1)
        pref = b.get("tide_pref", "any")
        if pref == "outgoing":
            tide = np.where(tide_dir == "outgoing", 1.0, np.where(tide_dir == "incoming", 0.6, 0.7))
            tide = tide * fill_neutral(0.6 + 0.4 * np.clip(cur / 0.5, 0, 1), 0.85)
        elif pref == "moving":
            tide = fill_neutral(0.45 + 0.55 * np.clip(cur / 0.6, 0, 1), 0.7)
        else:
            tide = np.full(len(df), 0.8)
        seas = fill_neutral(trapezoid(waves, -1, 0, 5, 11, floor=0.2), 0.8)
        storm = storm_multiplier(phase, hsince, BAIT_STORM)
        local = 0.45 * habitat + 0.2 * wind + 0.2 * tide + 0.15 * seas
        idx = 100 * np.sqrt(season * temp) * (0.35 + 0.65 * local) * storm
        out[f"bait_{g}"] = np.clip(idx, 0, 100)
        out[f"bf_{g}_season"] = season
        out[f"bf_{g}_temp"] = temp
    res = pd.DataFrame(out, index=df.index)
    groups = list(cfg)
    res["bait_index"] = res[[f"bait_{g}" for g in groups]].max(axis=1)
    res["bait_top"] = res[[f"bait_{g}" for g in groups]].idxmax(axis=1).str.replace("bait_", "", regex=False)
    return res
