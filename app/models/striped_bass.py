"""Striped bass convenience wrappers (the flagship species).

The generic engines in ``presence.py`` / ``feeding.py`` are parameterized per
species from configs/species.yaml; this module exposes the three striped bass
indices the platform was designed around:

* Baitfish Habitat Suitability Index  -> ``bait_index`` (baitfish.py)
* Striped Bass Habitat Suitability Index -> ``presence_striped_bass``
* Striped Bass Feeding Activity Index  -> ``feeding_striped_bass``
"""
from __future__ import annotations

import pandas as pd

SPECIES = "striped_bass"


def bass_indices(scored: pd.DataFrame) -> pd.DataFrame:
    cols = {"bait_index": "baitfish_hsi", f"presence_{SPECIES}": "bass_hsi",
            f"feeding_{SPECIES}": "bass_feeding", f"bite_{SPECIES}": "bass_bite"}
    return scored[[c for c in cols if c in scored]].rename(columns=cols)
