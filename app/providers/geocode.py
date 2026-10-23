"""Free geocoding: Zippopotam.us for ZIP codes, OpenStreetMap Nominatim for names."""
from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.core.http import FetchError, HttpClient


async def geocode_zip(http: HttpClient, zip_code: str) -> dict[str, Any]:
    z = zip_code.strip()[:5]
    if not z.isdigit():
        raise ValueError("ZIP code must be 5 digits")
    res = await http.request(f"https://api.zippopotam.us/us/{z}", provider="Zippopotam.us",
                             ttl=get_settings().ttl_geocode)
    places = res.data.get("places") or []
    if not places:
        raise FetchError(f"ZIP {z} not found")
    p = places[0]
    return {"lat": float(p["latitude"]), "lon": float(p["longitude"]),
            "label": f"{p['place name']}, {p['state abbreviation']} {z}"}


async def geocode_name(http: HttpClient, query: str) -> list[dict[str, Any]]:
    res = await http.request("https://nominatim.openstreetmap.org/search", provider="OSM Nominatim",
                             params={"q": query, "format": "jsonv2", "limit": 5, "countrycodes": "us"},
                             ttl=get_settings().ttl_geocode)
    return [{"lat": float(r["lat"]), "lon": float(r["lon"]), "label": r.get("display_name", "")[:120]}
            for r in res.data]
