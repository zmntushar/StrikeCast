"""Species feeding profiles for the 90-minute window optimizer.

Each species declares behavioral categories (diurnal / nocturnal / crepuscular /
flexible, several allowed), a solar-suitability curve over sun elevation, current
and tide preferences, and its own component weights (``configs/species.yaml``,
``window:`` block). A new species only needs ``categories`` — every other value
falls back to category templates or the species' existing hourly-model settings.
These are heuristic, configurable profiles, not calibrated biology.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.models.common import species_meta

CATEGORIES = ("diurnal", "nocturnal", "crepuscular", "flexible")
COMPONENTS = ("solar", "current", "tide", "habitat", "seasonal", "temperature", "conditions")

# Default solar curves [[sun elevation deg, suitability]] per behavioral category.
CATEGORY_SOLAR = {
    "diurnal": [[-90, 0.08], [-18, 0.08], [-12, 0.15], [-6, 0.35], [0, 0.6], [8, 0.85], [20, 1.0], [90, 1.0]],
    "nocturnal": [[-90, 1.0], [-18, 1.0], [-12, 0.9], [-6, 0.75], [0, 0.5], [10, 0.25], [90, 0.15]],
    "crepuscular": [[-90, 0.35], [-18, 0.4], [-12, 0.7], [-6, 0.95], [-3, 1.0], [3, 1.0], [8, 0.85], [15, 0.55],
                    [30, 0.4], [90, 0.35]],
    "flexible": [[-90, 0.65], [90, 0.65]],
}
DEFAULT_WEIGHTS = {"solar": 0.22, "current": 0.22, "tide": 0.1, "habitat": 0.1, "seasonal": 0.1,
                   "temperature": 0.1, "conditions": 0.16}


@dataclass(frozen=True)
class FeedingProfile:
    species: str
    name: str
    categories: tuple[str, ...]
    solar_curve: tuple[tuple[float, float], ...]
    overcast_bonus: float
    weights: dict[str, float]              # normalized to sum to 1
    limiting: dict[str, float]
    current_opt: tuple[float, float]       # preferred share of the local peak current
    slack_score: float                     # suitability at slack water
    manageable_current_kt: float | None
    preferred_tide: dict[str, list[str]]   # by spot environment
    condition_weights: dict[str, float]    # weights over the hourly environmental factors
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def category_label(self) -> str:
        return " · ".join(c.title() for c in self.categories)

    def tide_prefs(self, environment: str) -> list[str]:
        return self.preferred_tide.get(environment, ["incoming", "outgoing"])


def _normalize(w: dict[str, float]) -> dict[str, float]:
    w = {k: max(float(w.get(k, 0.0)), 0.0) for k in COMPONENTS}
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()} if tot > 0 else dict(DEFAULT_WEIGHTS)


def _blend_curves(cats: tuple[str, ...]) -> list[list[float]]:
    xs = np.unique(np.concatenate([[p[0] for p in CATEGORY_SOLAR[c]] for c in cats]))
    ys = np.mean([np.interp(xs, *zip(*CATEGORY_SOLAR[c])) for c in cats], axis=0)
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def load_profile(species: str) -> FeedingProfile:
    meta = species_meta(species)
    win = meta.get("window") or {}
    cats = tuple(c for c in win.get("categories", ["flexible"]) if c in CATEGORIES) or ("flexible",)
    curve = win.get("solar_curve") or _blend_curves(cats)
    curve = sorted(([float(a), float(b)] for a, b in curve), key=lambda p: p[0])
    cond = {k: v for k, v in meta.get("feeding_weights", {}).items()
            if k in ("wind", "waves", "pressure", "bait", "clarity", "temp_trend") and v > 0}
    return FeedingProfile(
        species=species, name=meta.get("name", species), categories=cats,
        solar_curve=tuple((a, b) for a, b in curve),
        overcast_bonus=float(win.get("overcast_bonus", meta.get("overcast_day_bonus", 0.0))),
        weights=_normalize(win.get("weights") or DEFAULT_WEIGHTS),
        limiting={k: float(v) for k, v in (win.get("limiting") or {}).items() if k in COMPONENTS and v > 0},
        current_opt=tuple(meta.get("current_opt", [0.3, 0.9])),  # type: ignore[arg-type]
        slack_score=float(meta.get("slack_score", 0.4)),
        manageable_current_kt=win.get("manageable_current_kt"),
        preferred_tide=meta.get("preferred_tide", {}),
        condition_weights=cond,
        notes=tuple(win.get("notes", [])),
    )


def solar_suitability(profile: FeedingProfile, elev: np.ndarray, cloud_pct: np.ndarray | None = None) -> np.ndarray:
    """Continuous 0-1 light suitability from solar elevation; overcast helps low-light feeders by day."""
    e = np.asarray(elev, dtype=float)
    xs, ys = zip(*profile.solar_curve)
    s = np.interp(e, xs, ys)
    if profile.overcast_bonus > 0 and cloud_pct is not None:
        c = np.nan_to_num(np.asarray(cloud_pct, dtype=float), nan=30.0) / 100
        s = np.where(e > 0, s + profile.overcast_bonus * c * (1 - s), s)
    return np.clip(s, 0, 1)
