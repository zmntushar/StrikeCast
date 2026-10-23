"""Weather providers.

* Open-Meteo forecast / historical-forecast API (free, no key, CC-BY 4.0):
  https://open-meteo.com/en/docs — hourly temperature, wind, gusts, MSL pressure,
  precipitation, cloud cover, visibility. Multi-location requests are batched.
* NOAA / National Weather Service API (public domain): https://www.weather.gov/documentation/services-web-api
  /points -> hourly forecast (official wind/temperature/PoP/short text) and
  /alerts/active?point= (Small Craft Advisory, Gale Warning, Coastal Flood ...).
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.geo import compass_to_deg
from app.core.http import FetchError, HttpClient
from app.schemas.env import Alert, DataKind, SourceInfo

OM_FORECAST = "https://api.open-meteo.com/v1/forecast"
OM_HISTORICAL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
NWS = "https://api.weather.gov"

OM_VARS = ["temperature_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
           "pressure_msl", "precipitation", "cloud_cover", "visibility", "weather_code"]
OM_RENAME = {"temperature_2m": "air_temp_f", "wind_speed_10m": "wind_mph",
             "wind_direction_10m": "wind_dir", "wind_gusts_10m": "gust_mph",
             "pressure_msl": "pressure_hpa", "precipitation": "precip_in",
             "cloud_cover": "cloud_pct", "visibility": "visibility_mi",
             "weather_code": "weather_code"}


def _om_frame(block: dict[str, Any]) -> pd.DataFrame:
    hourly = block.get("hourly") or {}
    if not hourly.get("time"):
        return pd.DataFrame()
    idx = pd.to_datetime(hourly["time"], utc=True)
    df = pd.DataFrame({OM_RENAME.get(k, k): hourly[k] for k in hourly if k != "time"}, index=idx)
    df = df.apply(pd.to_numeric, errors="coerce")
    if "visibility_mi" in df:
        df["visibility_mi"] = df["visibility_mi"] / 1609.34
    return df


async def openmeteo_weather(
    http: HttpClient,
    points: list[tuple[float, float]],
    *,
    past_days: int = 2,
    forecast_days: int = 8,
    start: date | None = None,
    end: date | None = None,
) -> list[tuple[pd.DataFrame, SourceInfo]]:
    """Hourly weather for many points in ONE request. ``start/end`` -> historical mode."""
    s = get_settings()
    params: dict[str, Any] = {
        "latitude": ",".join(f"{p[0]:.4f}" for p in points),
        "longitude": ",".join(f"{p[1]:.4f}" for p in points),
        "hourly": ",".join(OM_VARS),
        "wind_speed_unit": "mph", "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch", "timezone": "GMT",
    }
    if start and end:
        url, ttl, product = OM_HISTORICAL, s.ttl_openmeteo_history, "historical forecast archive"
        params.update(start_date=start.isoformat(), end_date=end.isoformat())
    else:
        url, ttl, product = OM_FORECAST, s.ttl_openmeteo_forecast, "best-match forecast (NOAA GFS/HRRR blend)"
        params.update(past_days=past_days, forecast_days=forecast_days)
    try:
        res = await http.request(url, provider="Open-Meteo", params=params, ttl=ttl)
    except FetchError:
        if not (start and end):
            raise
        # Older dates: fall back to ERA5 reanalysis archive.
        res = await http.request(OM_ARCHIVE, provider="Open-Meteo", params=params, ttl=ttl)
        product = "ERA5 reanalysis archive"
    blocks = res.data if isinstance(res.data, list) else [res.data]
    out = []
    for block in blocks:
        df = _om_frame(block)
        out.append((df, SourceInfo(
            feature="weather (wind, pressure, clouds, rain)", provider="Open-Meteo", product=product,
            kind=DataKind.model, fetched_at=res.fetched_at, ok=not df.empty, stale=res.stale,
            distance_km=None, url=res.url[:300],
            note=f"grid cell {block.get('latitude')}, {block.get('longitude')}")))
    return out


# --------------------------------------------------------------------- NWS
_MPH = re.compile(r"(\d+)")


def _parse_nws_speed(text: str | None) -> float | None:
    if not text:
        return None
    nums = [int(n) for n in _MPH.findall(text)]
    return float(max(nums)) if nums else None


async def nws_point(http: HttpClient, lat: float, lon: float) -> dict[str, Any]:
    res = await http.request(f"{NWS}/points/{lat:.4f},{lon:.4f}", provider="NOAA NWS",
                             ttl=get_settings().ttl_nws_points)
    return res.data.get("properties", {})


async def nws_hourly(http: HttpClient, forecast_hourly_url: str) -> tuple[pd.DataFrame, SourceInfo]:
    res = await http.request(forecast_hourly_url, provider="NOAA NWS", ttl=get_settings().ttl_nws_hourly)
    periods = res.data.get("properties", {}).get("periods", [])
    rows = []
    for p in periods:
        rows.append({
            "time": pd.Timestamp(p["startTime"]).tz_convert("UTC"),
            "nws_temp_f": float(p["temperature"]) if p.get("temperature") is not None else np.nan,
            "nws_wind_mph": _parse_nws_speed(p.get("windSpeed")),
            "nws_wind_dir": compass_to_deg(p.get("windDirection")),
            "pop_pct": (p.get("probabilityOfPrecipitation") or {}).get("value"),
            "short_forecast": p.get("shortForecast", ""),
        })
    df = pd.DataFrame(rows).set_index("time") if rows else pd.DataFrame()
    if not df.empty:
        for c in ("nws_wind_mph", "nws_wind_dir", "pop_pct"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    upd = res.data.get("properties", {}).get("updateTime")
    return df, SourceInfo(feature="official hourly forecast (wind, PoP, sky)", provider="NOAA NWS",
                          product="forecastHourly", kind=DataKind.predicted, fetched_at=res.fetched_at,
                          observed_at=pd.Timestamp(upd).to_pydatetime() if upd else None,
                          ok=not df.empty, stale=res.stale, url=forecast_hourly_url)


async def nws_alerts(http: HttpClient, lat: float, lon: float) -> list[Alert]:
    res = await http.request(f"{NWS}/alerts/active", provider="NOAA NWS",
                             params={"point": f"{lat:.4f},{lon:.4f}"}, ttl=get_settings().ttl_nws_alerts)
    alerts = []
    for f in res.data.get("features", []):
        p = f.get("properties", {})
        alerts.append(Alert(
            event=p.get("event", ""), headline=p.get("headline") or p.get("event", ""),
            severity=p.get("severity", ""),
            onset=pd.Timestamp(p["onset"]).to_pydatetime() if p.get("onset") else None,
            ends=pd.Timestamp(p["ends"]).to_pydatetime() if p.get("ends") else None))
    return alerts


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
