"""Shared scoring primitives. All factor scores are 0-1 (1 = most favorable)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import species_config


def col(df: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), default, dtype=float)


def fill_neutral(x: np.ndarray, neutral: float = 0.6) -> np.ndarray:
    """Missing inputs score as a neutral value (and lower the data-quality rating)."""
    return np.where(np.isnan(x), neutral, x)


def ramp_band(x: np.ndarray, flat_val: float, lo: float, hi: float, bad: float, bad_val: float) -> np.ndarray:
    """flat_val at 0 -> 1 at lo; 1 on [lo, hi]; falls to bad_val at ``bad`` and beyond."""
    x = np.asarray(x, dtype=float)
    below = flat_val + (1 - flat_val) * np.clip(x / max(lo, 1e-6), 0, 1)
    above = 1 - (1 - bad_val) * np.clip((x - hi) / max(bad - hi, 1e-6), 0, 1)
    out = np.where(x < lo, below, np.where(x <= hi, 1.0, above))
    return np.where(np.isnan(x), np.nan, out)


def species_list() -> list[str]:
    return list(species_config()["species"].keys())


def species_meta(sp: str) -> dict:
    return species_config()["species"][sp]


def bait_groups() -> list[str]:
    return list(species_config()["baitfish"].keys())
