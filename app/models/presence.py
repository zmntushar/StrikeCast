"""Species Habitat Suitability Index (presence), 0-100.

Season (location-specific migration curve) and water temperature set the
ceiling; shoreline structure (config tags confirmed/boosted by bathymetry) and
the availability of the species' preferred bait refine it. Used for striped bass
(see ``striped_bass.py``) and the other target species.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.features.seasonal import seasonal_factor, trapezoid
from app.models.common import col, fill_neutral, species_meta
from app.schemas.env import Spot


def structure_score(sp_cfg: dict, spot: Spot, bathy_bonus: dict[str, float] | None = None) -> tuple[float, str]:
    aff = sp_cfg["structure"]
    tags = {t: 1.0 for t in spot.structure}
    tags.setdefault(spot.environment, 1.0)
    for t, conf in (bathy_bonus or {}).items():
        tags[t] = max(tags.get(t, 0), conf)
    scored = sorted(((aff.get(t, 0.3) * conf, t) for t, conf in tags.items()), reverse=True)
    if not scored:
        return 0.4, "no structure info"
    best, best_tag = scored[0]
    rest = [s for s, _ in scored[1:3]]
    score = min(1.0, best + 0.08 * sum(rest))
    if sp_cfg.get("structure_required") and best < 0.5:
        score *= 0.35
    return score, best_tag.replace("_", " ")


def bait_availability(sp_cfg: dict, bait: pd.DataFrame) -> np.ndarray | None:
    diet = sp_cfg.get("diet") or {}
    if not diet:
        return None
    tot = sum(diet.values())
    acc = np.zeros(len(bait))
    for g, w in diet.items():
        if f"bait_{g}" in bait:
            acc += w * bait[f"bait_{g}"].to_numpy() / 100
    # a strong showing of any favored bait counts more than the average
    best = np.max(np.vstack([bait[f"bait_{g}"].to_numpy() / 100 for g in diet if f"bait_{g}" in bait]), axis=0)
    return np.clip(0.6 * acc / tot + 0.4 * best, 0, 1)


def presence_scores(df: pd.DataFrame, spot: Spot, species: str, bait: pd.DataFrame,
                    bathy_bonus: dict[str, float] | None = None) -> pd.DataFrame:
    cfg = species_meta(species)
    doy = df.index.tz_convert(spot.timezone).day_of_year.to_numpy()
    season = seasonal_factor(cfg["season"], doy, spot.lat, spot.water_body)
    temp_raw = trapezoid(col(df, "water_temp_f"), *cfg["temp_f"], floor=0.02)
    temp = fill_neutral(temp_raw, 0.55)
    struct, struct_tag = structure_score(cfg, spot, bathy_bonus)
    bait_av = bait_availability(cfg, bait)
    w = cfg["presence_weights"]
    if bait_av is None or w.get("bait", 0) == 0:
        local = np.full(len(df), struct)
        bait_col = np.full(len(df), np.nan)
    else:
        local = (w["structure"] * struct + w["bait"] * bait_av) / (w["structure"] + w["bait"])
        bait_col = bait_av
    p = 100 * np.sqrt(season * temp) * (0.3 + 0.7 * local)
    p = np.clip(p, 0, 100)
    return pd.DataFrame({
        f"presence_{species}": p,
        f"pf_{species}_season": season,
        f"pf_{species}_temp": temp,
        f"pf_{species}_structure": np.full(len(df), struct),
        f"pf_{species}_bait": bait_col,
        f"pf_{species}_structure_tag": struct_tag,
    }, index=df.index)
