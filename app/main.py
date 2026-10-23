"""StrikeCast REST API (FastAPI).

Run:  uvicorn app.main:app --reload     ->  http://127.0.0.1:8000/docs

The Streamlit dashboard calls the same service layer in-process; this API makes
the platform usable from a future React/mobile frontend or other tools.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import get_settings, regulations_config
from app.core.http import HttpClient
from app.core.logging import get_logger
from app.models.common import species_list
from app.providers.geocode import geocode_name, geocode_zip
from app.schemas.env import ApiSpotForecast, Spot
from app.services.compare import compare_dates
from app.services.forecast import SpotForecast, collect, forecast_frame, get_spot, load_spots
from app.services.observations import add_observation, list_observations

log = get_logger("api")
_scheduler = None


def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def to_api(fc: SpotForecast, hourly_cols: list[str] | None = None) -> ApiSpotForecast:
    d = fc.display
    cols = hourly_cols or [c for c in d.columns if not c.startswith(("pf_", "ff_", "bf_"))]
    hourly = [{"time": ts.isoformat(), **{c: _json_safe(row[c]) for c in cols}} for ts, row in d[cols].iterrows()]
    now = {k: _json_safe(v) for k, v in fc.now_row().items() if not str(k).startswith(("pf_", "ff_", "bf_"))}
    return ApiSpotForecast(spot=fc.spot, generated_at=fc.generated_at, timezone=fc.spot.timezone,
                           quality=fc.quality, current=now,
                           windows=[w for ws in fc.windows.values() for w in ws], alerts=fc.alerts,
                           sources=fc.sources, hourly=hourly,
                           bathymetry=fc.bathy.to_dict() if fc.bathy else None)


async def refresh_favorites() -> None:
    spots = [s for s in load_spots() if s.favorite]
    await collect(spots, forecast_frame())
    log.info("scheduled refresh complete")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _scheduler
    s = get_settings()
    if s.scheduler_enabled:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        _scheduler = AsyncIOScheduler()
        _scheduler.add_job(refresh_favorites, "interval", minutes=s.scheduler_interval_min, id="refresh",
                           next_run_time=None)
        _scheduler.start()
    yield
    if _scheduler:
        _scheduler.shutdown(wait=False)


app = FastAPI(title="StrikeCast API", version="1.0",
              description="Free, open-data fishing intelligence for striped bass, bluefish, fluke, blackfish and porgy.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/spots", response_model=list[Spot])
def spots() -> list[Spot]:
    return load_spots()


@app.get("/species")
def species() -> list[str]:
    return species_list()


@app.get("/regulations")
def regulations() -> dict[str, Any]:
    return regulations_config()


@app.get("/forecast/{spot_id}", response_model=ApiSpotForecast)
async def forecast(spot_id: str, days: int = Query(7, ge=1, le=7)) -> ApiSpotForecast:
    try:
        spot = get_spot(spot_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown spot {spot_id}") from exc
    res = await collect([spot], forecast_frame(days))
    return to_api(res[spot.id])


@app.get("/rank")
async def rank(species: str = Query("striped_bass"), favorites_only: bool = False) -> list[dict[str, Any]]:
    if species not in species_list():
        raise HTTPException(400, f"species must be one of {species_list()}")
    sp = [s for s in load_spots() if s.favorite or not favorites_only]
    res = await collect(sp, forecast_frame(3))
    out = []
    for fc in res.values():
        w = fc.windows.get(species) or []
        out.append({"spot_id": fc.spot.id, "name": fc.spot.name, "now": _json_safe(fc.bite_now(species)),
                    "best_today": _json_safe(fc.best_today(species)),
                    "next_window": w[0].model_dump() if w else None, "quality": fc.quality})
    return sorted(out, key=lambda r: r["best_today"] or 0, reverse=True)


@app.get("/compare")
async def compare(spot_id: str = "robert_moses_democrat_point", date_a: date = date(2026, 10, 10),
                  date_b: date = date(2026, 10, 22)) -> dict[str, Any]:
    c = await compare_dates(get_spot(spot_id), date_a, date_b)
    return {"spot": c.spot.model_dump(), "table": [{k: _json_safe(v) for k, v in r.items()} for r in c.table.to_dict("records")],
            "explanation": c.explanation}


@app.get("/geocode")
async def geocode(q: str) -> list[dict[str, Any]]:
    async with HttpClient() as http:
        if q.strip().isdigit():
            return [await geocode_zip(http, q)]
        return await geocode_name(http, q)


class ObservationIn(BaseModel):
    spot_id: str
    start_utc: str
    species: str
    fish_caught: int = 0
    duration_h: float = 2.0
    largest_in: float | None = None
    lure: str | None = None
    bait_seen: str | None = None
    birds_diving: bool = False
    blitz: bool = False
    anglers: int = 1
    notes: str | None = None
    confidence: str = "user_reported"


@app.post("/observations")
def post_observation(o: ObservationIn) -> dict[str, int]:
    return {"id": add_observation(o.model_dump())}


@app.get("/observations")
def get_observations() -> list[dict[str, Any]]:
    df = list_observations()
    return [{k: _json_safe(v) for k, v in r.items()} for r in df.to_dict("records")]
