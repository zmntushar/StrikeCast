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

FIELDS = ["spot_id", "lat", "lon", "start_utc", "end_utc", "duration_h", "species", "bites", "fish_caught", "largest_in",
          "lure", "bait_seen", "birds_diving", "blitz", "anglers", "water_notes", "notes", "confidence", "source"]


def _utc(v: Any) -> datetime | None:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    ts = pd.Timestamp(v)
    return (ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).to_pydatetime()


def add_observation(data: dict[str, Any], features: dict[str, Any] | None = None,
                    prediction: dict[str, Any] | None = None) -> int:
    """Save a trip. ``prediction`` = the window optimizer's output for that spot/date/species."""
    clean = {k: data.get(k) for k in FIELDS if k in data}
    clean["start_utc"] = _utc(clean.get("start_utc"))
    if "end_utc" in clean:
        clean["end_utc"] = _utc(clean["end_utc"])
    if clean.get("end_utc") and clean.get("start_utc") and not clean.get("duration_h"):
        clean["duration_h"] = (clean["end_utc"] - clean["start_utc"]).total_seconds() / 3600
    if not clean.get("duration_h"):
        clean["duration_h"] = 2.0
    with session() as db:
        obs = Observation(**clean, features_json=json.dumps(features) if features else None,
                          prediction_json=json.dumps(prediction, default=str) if prediction else None)
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
        d["has_prediction"] = bool(r.prediction_json)
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
    df = pd.DataFrame([{**{k: getattr(r, k) for k in FIELDS}, "features_json": r.features_json,
                        "prediction_json": r.prediction_json} for r in rows],
                      columns=FIELDS + ["features_json", "prediction_json"])
    return df.to_csv(index=False).encode("utf-8")


def import_csv(content: bytes) -> int:
    df = pd.read_csv(io.BytesIO(content))
    missing = {"spot_id", "start_utc", "species"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    n = 0
    for rec in df.to_dict(orient="records"):
        feats = rec.pop("features_json", None)
        pred = rec.pop("prediction_json", None)
        rec = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in rec.items()}
        for b in ("birds_diving", "blitz"):
            if b in rec and rec[b] is not None:
                rec[b] = str(rec[b]).lower() in ("1", "true", "yes", "y")
        rec.setdefault("source", "csv_import")
        add_observation(rec, json.loads(feats) if isinstance(feats, str) and feats else None,
                        json.loads(pred) if isinstance(pred, str) and pred else None)
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


def sessions_with_predictions(species: str | None = None) -> list[dict[str, Any]]:
    """Trips that carry a saved window prediction (for forecast evaluation / calibration)."""
    with session() as db:
        q = select(Observation).where(Observation.prediction_json.is_not(None))
        if species:
            q = q.where(Observation.species == species)
        rows = db.execute(q.order_by(Observation.start_utc)).scalars().all()
    return [{**{k: getattr(r, k) for k in ["id"] + FIELDS}, "prediction": json.loads(r.prediction_json)} for r in rows]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
