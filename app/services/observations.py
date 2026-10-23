"""Fishing log: record trips / sightings, CSV import & export.

Each record stores an environmental feature snapshot for its time and spot so
models can later learn from real outcomes. Confidence is tracked explicitly
(verified / user_reported / inferred). Remember: no reported catch does NOT
mean no fish were present (observation bias).
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import select

from app.database.db import Observation, session

FIELDS = ["spot_id", "lat", "lon", "start_utc", "duration_h", "species", "fish_caught", "largest_in", "lure",
          "bait_seen", "birds_diving", "blitz", "anglers", "water_notes", "notes", "confidence", "source"]


def add_observation(data: dict[str, Any], features: dict[str, Any] | None = None) -> int:
    clean = {k: data.get(k) for k in FIELDS if k in data}
    st = clean.get("start_utc")
    if isinstance(st, str):
        st = pd.Timestamp(st)
    if isinstance(st, pd.Timestamp):
        st = (st.tz_localize("UTC") if st.tzinfo is None else st.tz_convert("UTC")).to_pydatetime()
    clean["start_utc"] = st
    with session() as db:
        obs = Observation(**clean, features_json=json.dumps(features) if features else None)
        db.add(obs)
        db.commit()
        return obs.id


def list_observations() -> pd.DataFrame:
    with session() as db:
        rows = db.execute(select(Observation).order_by(Observation.start_utc.desc())).scalars().all()
    recs = []
    for r in rows:
        d = {k: getattr(r, k) for k in ["id", "created_at"] + FIELDS}
        d["has_features"] = bool(r.features_json)
        recs.append(d)
    return pd.DataFrame(recs)


def delete_observation(obs_id: int) -> None:
    with session() as db:
        obj = db.get(Observation, obs_id)
        if obj:
            db.delete(obj)
            db.commit()


def export_csv() -> bytes:
    with session() as db:
        rows = db.execute(select(Observation)).scalars().all()
    df = pd.DataFrame([{**{k: getattr(r, k) for k in FIELDS}, "features_json": r.features_json} for r in rows],
                      columns=FIELDS + ["features_json"])
    return df.to_csv(index=False).encode("utf-8")


def import_csv(content: bytes) -> int:
    df = pd.read_csv(io.BytesIO(content))
    missing = {"spot_id", "start_utc", "species"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    n = 0
    for rec in df.to_dict(orient="records"):
        feats = rec.pop("features_json", None)
        rec = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in rec.items()}
        for b in ("birds_diving", "blitz"):
            if b in rec and rec[b] is not None:
                rec[b] = str(rec[b]).lower() in ("1", "true", "yes", "y")
        rec.setdefault("source", "csv_import")
        add_observation(rec, json.loads(feats) if isinstance(feats, str) and feats else None)
        n += 1
    return n


def training_frame() -> pd.DataFrame:
    """One row per observation with flattened environmental features."""
    with session() as db:
        rows = db.execute(select(Observation)).scalars().all()
    recs = []
    for r in rows:
        if not r.features_json:
            continue
        f = json.loads(r.features_json)
        recs.append({"id": r.id, "start_utc": r.start_utc, "spot_id": r.spot_id, "species": r.species,
                     "caught": int((r.fish_caught or 0) > 0), "blitz": bool(r.blitz),
                     "birds": bool(r.birds_diving), "confidence": r.confidence, **f})
    return pd.DataFrame(recs)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
