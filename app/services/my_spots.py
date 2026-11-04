"""The user's saved spots ("Your spots").

Stored in data/my_spots.json so they survive restarts and stay out of git:

    {"ids": ["jones_beach_field_10", ...],          # order = order on the dashboard
     "custom": [{...Spot fields...}, ...],          # searched places the user saved
     "stations": {"ndbc": {...}, "coops": {...}}}   # station metadata those places use

Until the user changes anything, the `favorite: true` spots in configs/spots.yaml are used.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.core.config import ROOT_DIR, spots_config, stations_config

STORE = Path(os.environ.get("STRIKECAST_MY_SPOTS_FILE", ROOT_DIR / "data" / "my_spots.json"))


def _default() -> dict[str, Any]:
    return {"ids": [s["id"] for s in spots_config()["spots"] if s.get("favorite")], "custom": [], "stations": {}}


def _read() -> dict[str, Any]:
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _default()
    data.setdefault("ids", [])
    data.setdefault("custom", [])
    data.setdefault("stations", {})
    return data


def _write(data: dict[str, Any]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(STORE)


def saved_ids() -> list[str]:
    return list(_read()["ids"])


def saved_custom() -> list[dict[str, Any]]:
    """Saved searched places, re-registering their stations so names/distances resolve."""
    data = _read()
    sc = stations_config()
    for kind, stations in data["stations"].items():
        for sid, meta in stations.items():
            sc.setdefault(kind, {}).setdefault(sid, meta)
    keep = set(data["ids"])
    return [c for c in data["custom"] if c.get("id") in keep]


def set_ids(ids: list[str]) -> None:
    data = _read()
    data["ids"] = list(dict.fromkeys(ids))
    data["custom"] = [c for c in data["custom"] if c.get("id") in set(data["ids"])]
    _write(data)


def add(spot_id: str) -> None:
    ids = saved_ids()
    if spot_id not in ids:
        set_ids(ids + [spot_id])


def remove(spot_id: str) -> None:
    set_ids([i for i in saved_ids() if i != spot_id])


def move(spot_id: str, offset: int) -> None:
    """Move a saved spot up (offset < 0) or down (offset > 0); the dashboard shows spots in this order."""
    ids = saved_ids()
    if spot_id not in ids:
        return
    i = ids.index(spot_id)
    j = max(0, min(len(ids) - 1, i + offset))
    if i != j:
        ids.insert(j, ids.pop(i))
        set_ids(ids)


def save_custom(spot: Any, name: str | None = None) -> str:
    """Save a searched place (a custom Spot) to Your spots. Returns its id."""
    fields = spot.model_dump(exclude={"favorite"})
    if name:
        fields["name"] = name
        fields["short"] = name[:14]
    data = _read()
    data["custom"] = [c for c in data["custom"] if c.get("id") != spot.id] + [fields]
    srcs = list(spot.temp_sources) + list(spot.wave_sources)
    srcs += [s for s in (spot.tide_station, spot.current_station) if s]
    sc = stations_config()
    for src in srcs:
        # Same lookup rules as station_meta: "ndbc:44025" / "coops:8516945" / bare current or tide id.
        kind, sid = src.split(":", 1) if ":" in src else ("currents" if src in sc.get("currents", {}) else "coops", src)
        meta = (sc.get(kind) or {}).get(sid)
        if meta and meta.get("lat") is not None:
            data["stations"].setdefault(kind, {})[sid] = dict(meta)
    if spot.id not in data["ids"]:
        data["ids"].append(spot.id)
    _write(data)
    return spot.id
