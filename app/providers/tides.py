"""NOAA CO-OPS Tides & Currents (public domain).

API docs: https://api.tidesandcurrents.noaa.gov/api/prod/
* predictions (interval=hilo) — works for harmonic AND subordinate stations
* currents_predictions (interval=MAX_SLACK) — official max flood/ebb + slack times
* water_temperature, water_level (observed) and hourly predictions at reference
  gauges for water-level anomaly (surge) estimates.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.core.http import FetchError, HttpClient
from app.schemas.env import CurrentEvent, TideEvent

API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def _base(station: str, product: str, begin: date, end: date, **extra: Any) -> dict[str, Any]:
    return {"product": product, "station": station, "begin_date": begin.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"), "units": "english", "time_zone": "gmt",
            "format": "json", "application": "StrikeCast", **extra}


def _t(s: str) -> datetime:
    return pd.Timestamp(s, tz="UTC").to_pydatetime()


async def tide_hilo(http: HttpClient, station: str, begin: date, end: date) -> tuple[list[TideEvent], str]:
    res = await http.request(API, provider="NOAA CO-OPS", ttl=get_settings().ttl_coops_predictions,
                             params=_base(station, "predictions", begin, end, datum="MLLW", interval="hilo"))
    if "error" in res.data:
        raise FetchError(res.data["error"].get("message", "CO-OPS error"))
    events = [TideEvent(time=_t(p["t"]), height_ft=float(p["v"]), type=p["type"])
              for p in res.data.get("predictions", [])]
    return events, res.url


async def current_events(http: HttpClient, station: str, begin: date, end: date, bin_: int = 1) -> tuple[list[CurrentEvent], str]:
    res = await http.request(API, provider="NOAA CO-OPS", ttl=get_settings().ttl_coops_predictions,
                             params=_base(station, "currents_predictions", begin, end, interval="MAX_SLACK", bin=bin_))
    if "error" in res.data:
        raise FetchError(res.data["error"].get("message", "CO-OPS error"))
    cps = (res.data.get("current_predictions") or {}).get("cp", [])
    out = []
    for c in cps:
        v = float(c.get("Velocity_Major") or 0.0)
        typ = c.get("Type", "").lower()
        d = c.get("meanFloodDir") if v >= 0 else c.get("meanEbbDir")
        out.append(CurrentEvent(time=_t(c["Time"]), velocity_kt=v, type=typ,
                                direction_deg=float(d) if d is not None else None))
    return out, res.url


async def water_temperature(http: HttpClient, station: str, begin: date, end: date) -> tuple[pd.DataFrame, str]:
    res = await http.request(API, provider="NOAA CO-OPS", ttl=get_settings().ttl_coops_observations,
                             params=_base(station, "water_temperature", begin, end))
    if "error" in res.data:
        raise FetchError(res.data["error"].get("message", "CO-OPS error"))
    rows = [(pd.Timestamp(d["t"], tz="UTC"), float(d["v"])) for d in res.data.get("data", []) if d.get("v")]
    if not rows:
        raise FetchError("no water temperature data")
    df = pd.DataFrame(rows, columns=["time", "water_temp_f"]).set_index("time")
    return df, res.url


async def water_level_anomaly(http: HttpClient, station: str, begin: date, end: date) -> tuple[pd.Series, str]:
    """Observed minus predicted water level (ft), hourly. Positive = surge."""
    s = get_settings()
    obs = await http.request(API, provider="NOAA CO-OPS", ttl=s.ttl_coops_observations,
                             params=_base(station, "hourly_height", begin, end, datum="MLLW"))
    if "error" in obs.data:
        # hourly_height lags a few days; use 6-minute water_level instead.
        obs = await http.request(API, provider="NOAA CO-OPS", ttl=s.ttl_coops_observations,
                                 params=_base(station, "water_level", begin, end, datum="MLLW"))
    pred = await http.request(API, provider="NOAA CO-OPS", ttl=s.ttl_coops_predictions,
                              params=_base(station, "predictions", begin, end, datum="MLLW", interval="h"))
    if "error" in obs.data or "error" in pred.data:
        raise FetchError("water level unavailable")
    o = pd.Series({pd.Timestamp(d["t"], tz="UTC"): float(d["v"]) for d in obs.data.get("data", []) if d.get("v")})
    p = pd.Series({pd.Timestamp(d["t"], tz="UTC"): float(d["v"]) for d in pred.data.get("predictions", []) if d.get("v")})
    if o.empty or p.empty:
        raise FetchError("water level unavailable")
    o = o.resample("1h").mean()
    anom = (o - p.reindex(o.index)).dropna()
    return anom, obs.url


def hilo_frame(events: list[TideEvent]) -> pd.DataFrame:
    return pd.DataFrame([{"time": e.time, "height_ft": e.height_ft, "type": e.type} for e in events])


def current_frame(events: list[CurrentEvent]) -> pd.DataFrame:
    return pd.DataFrame([{"time": e.time, "velocity_kt": e.velocity_kt, "type": e.type} for e in events])

