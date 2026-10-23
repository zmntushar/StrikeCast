"""Station discovery for any US coastal point (custom ZIP / lat-lon locations).

Uses the CO-OPS metadata API and NDBC's active station list (both free) and
selects stations by distance AND product availability. Configured spots use the
hand-checked station choices in configs/spots.yaml instead.
"""
from __future__ import annotations

import re
from typing import Any

from app.core.config import get_settings, stations_config
from app.core.geo import bearing_deg, haversine_km
from app.core.http import HttpClient
from app.providers.geography import analyze, fetch_grid
from app.providers.weather import nws_point
from app.schemas.env import Spot

MDAPI = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
NDBC_ACTIVE = "https://www.ndbc.noaa.gov/activestations.xml"


async def _coops_list(http: HttpClient, kind: str) -> list[dict[str, Any]]:
    res = await http.request(MDAPI, provider="NOAA CO-OPS metadata", params={"type": kind},
                             ttl=30 * 86400)
    return res.data.get("stations", [])


async def _ndbc_list(http: HttpClient) -> list[dict[str, Any]]:
    res = await http.request(NDBC_ACTIVE, provider="NOAA NDBC", ttl=7 * 86400, as_json=False)
    out = []
    for m in re.finditer(r"<station ([^>]*)/?>", res.data):
        a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        try:
            out.append({"id": a["id"], "name": a.get("name", ""), "lat": float(a["lat"]), "lon": float(a["lon"]),
                        "met": a.get("met") == "y", "type": a.get("type", "")})
        except (KeyError, ValueError):
            continue
    return out


def _nearest(items: list[dict], lat: float, lon: float, *, lat_key="lat", lon_key="lng", max_km: float = 1e9, n: int = 1):
    scored = []
    for s in items:
        try:
            d = haversine_km(lat, lon, float(s[lat_key]), float(s[lon_key]))
        except (KeyError, TypeError, ValueError):
            continue
        if d <= max_km:
            scored.append((d, s))
    scored.sort(key=lambda x: x[0])
    return scored[:n]


async def build_custom_spot(http: HttpClient, lat: float, lon: float, environment: str = "beach",
                            label: str = "Custom location") -> tuple[Spot, list[str]]:
    """Create a Spot for an arbitrary coastal point, with automatic station choice."""
    notes: list[str] = []
    tz = get_settings().default_timezone
    try:
        props = await nws_point(http, lat, lon)
        tz = props.get("timeZone") or tz
    except Exception:  # noqa: BLE001 — NWS optional here
        notes.append("NWS point lookup failed; timezone defaulted")

    tide = _nearest(await _coops_list(http, "tidepredictions"), lat, lon, max_km=60)
    cur = _nearest([s for s in await _coops_list(http, "currentpredictions") if str(s.get("currbin", "1")) in ("1", "None")],
                   lat, lon, max_km=25)
    wt = _nearest(await _coops_list(http, "watertemp"), lat, lon, max_km=80, n=2)
    buoys = [b for b in await _ndbc_list(http) if b["met"]]
    nb = _nearest(buoys, lat, lon, lon_key="lon", max_km=150, n=2)

    # register discovered stations so distances / names resolve in the UI
    sc = stations_config()
    for d, s in tide + cur + wt:
        sc.setdefault("coops" if not str(s["id"]).startswith(("ACT", "LIS", "NYH", "HUR")) else "currents", {})[str(s["id"])] = {
            "name": s.get("name", ""), "lat": float(s["lat"]), "lon": float(s["lng"]), "bin": 1}
    for d, b in nb:
        sc.setdefault("ndbc", {})[b["id"]] = {"name": b["name"], "lat": b["lat"], "lon": b["lon"], "kind": "offshore_buoy"}

    facing = 180.0
    try:
        grid = await fetch_grid(http, lat, lon)
        an = analyze(grid, facing)
        if an.derived_facing_deg is not None:
            facing = float(an.derived_facing_deg)
            notes.append(f"shoreline orientation derived from bathymetry: {facing:.0f}°")
    except Exception:  # noqa: BLE001
        notes.append("bathymetry unavailable; facing defaulted to 180°")
    if not tide:
        notes.append("no NOAA tide prediction station within 60 km")

    spot = Spot(
        id=f"custom_{lat:.4f}_{lon:.4f}", name=label, short=label[:14], lat=lat, lon=lon,
        environment=environment, water_body="ocean" if environment in ("beach", "jetty", "inlet") else "bay",
        facing_deg=facing, exposure=1.0 if environment == "beach" else 0.6, structure=[environment],
        tide_station=str(tide[0][1]["id"]) if tide else None,
        current_station=str(cur[0][1]["id"]) if cur else None, current_is_proxy=not cur or cur[0][0] > 5,
        temp_sources=[f"coops:{s['id']}" for _, s in wt] + [f"ndbc:{b['id']}" for _, b in nb],
        wave_sources=[f"ndbc:{b['id']}" for _, b in nb], timezone=tz, custom=True,
        notes="Custom location — stations chosen automatically by distance and data availability.",
    )
    return spot, notes


def station_meta(source: str) -> dict[str, Any]:
    """Resolve 'ndbc:44025' / 'coops:8516945' / 'ACT3541' to name + coordinates."""
    sc = stations_config()
    if ":" in source:
        kind, sid = source.split(":", 1)
        return {"id": sid, "kind": kind, **(sc.get(kind, {}).get(sid) or {})}
    if source in sc.get("currents", {}):
        return {"id": source, "kind": "currents", **sc["currents"][source]}
    return {"id": source, "kind": "coops", **(sc.get("coops", {}).get(source) or {})}


def distance_to(spot: Spot, meta: dict[str, Any]) -> tuple[float | None, float | None]:
    if "lat" not in meta:
        return None, None
    return (round(haversine_km(spot.lat, spot.lon, meta["lat"], meta["lon"]), 1),
            round(bearing_deg(spot.lat, spot.lon, meta["lat"], meta["lon"])))
