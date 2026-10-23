"""Geographic fishing intelligence from free NOAA bathymetry.

Source: NOAA NCEI "DEM_all" mosaic ImageServer (public domain), which serves the
best available DEM at each point — on Long Island this is the Continuously
Updated DEM (CUDEM) at 1/9 arc-second (~3 m). ``getSamples`` lets us pull a grid
of depths around each spot in a handful of requests.

What we compute (and what we deliberately do NOT claim)
-------------------------------------------------------
* depth, slope (gradient) and local relief within casting range of shore
* candidate drop-offs / channel edges, troughs behind bars, shallow bars
* shoreline orientation (direction to deeper water) and distance to inlets

Elevations are relative to NAVD88 (≈ 0.7 m above MLLW on the south shore).
Survey dates vary by area and sandbars move after storms, so these are labeled
"candidate" structures — confirm by reading the water.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.config import get_settings, spots_config
from app.core.geo import haversine_km
from app.core.http import HttpClient

DEM_URL = "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_all/ImageServer/getSamples"
CHUNK = 500


@dataclass
class BathyGrid:
    lat0: float
    lon0: float
    step_m: float
    n: int
    z: np.ndarray          # (n, n) elevation in meters, row 0 = south, col 0 = west
    resolution_m: float | None
    fetched_ok: bool = True

    def cell_latlon(self, j: int, i: int) -> tuple[float, float]:
        dlat = self.step_m / 111_320
        dlon = self.step_m / (111_320 * math.cos(math.radians(self.lat0)))
        return self.lat0 + (j - self.n // 2) * dlat, self.lon0 + (i - self.n // 2) * dlon


@dataclass
class BathyAnalysis:
    ok: bool
    source: str = "NOAA NCEI DEM mosaic (CUDEM 1/9 arc-sec where available)"
    resolution_m: float | None = None
    datum: str = "NAVD88 (≈0.7 m above MLLW)"
    max_depth_cast_m: float | None = None
    max_slope_pct: float | None = None
    dist_to_3m_m: float | None = None
    derived_facing_deg: float | None = None
    structures: dict[str, int] = field(default_factory=dict)
    targets: list[dict[str, Any]] = field(default_factory=list)   # ranked casting targets
    cells: list[dict[str, Any]] = field(default_factory=list)     # classified cells for the map
    nearest_inlet: dict[str, Any] | None = None
    nearest_river: dict[str, Any] | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


async def fetch_grid(http: HttpClient, lat: float, lon: float, half_m: float = 400, step_m: float = 20) -> BathyGrid:
    n = int(2 * half_m / step_m) + 1
    dlat = step_m / 111_320
    dlon = step_m / (111_320 * math.cos(math.radians(lat)))
    pts = [[round(lon + (i - n // 2) * dlon, 6), round(lat + (j - n // 2) * dlat, 6)]
           for j in range(n) for i in range(n)]
    z = np.full(n * n, np.nan)
    res_deg = None
    for b in range(0, len(pts), CHUNK):
        chunk = pts[b:b + CHUNK]
        geom = json.dumps({"points": chunk, "spatialReference": {"wkid": 4326}})
        res = await http.request(DEM_URL, provider="NOAA NCEI DEM", method="POST",
                                 data={"geometry": geom, "geometryType": "esriGeometryMultipoint",
                                       "returnFirstValueOnly": "true", "f": "json"},
                                 ttl=get_settings().ttl_bathymetry)
        for smp in res.data.get("samples", []):
            try:
                z[b + int(smp["locationId"])] = float(smp["value"])
                res_deg = res_deg or smp.get("resolution")
            except (KeyError, TypeError, ValueError):
                continue
    res_m = float(res_deg) * 111_320 if res_deg else None
    return BathyGrid(lat, lon, step_m, n, z.reshape(n, n), res_m, fetched_ok=bool(np.isfinite(z).any()))


def nearest_feature(lat: float, lon: float, kind: str) -> dict[str, Any] | None:
    feats = spots_config().get("features", {}).get(kind, [])
    best = None
    for f in feats:
        d = haversine_km(lat, lon, f["lat"], f["lon"])
        if best is None or d < best["distance_km"]:
            best = {"name": f["name"], "distance_km": round(d, 2)}
    return best


def analyze(grid: BathyGrid, facing_deg: float, *, walk_m: float = 350, cast_m: float = 90) -> BathyAnalysis:
    """Classify water cells within casting distance of shoreline reachable from the spot."""
    z = grid.z
    n, step = grid.n, grid.step_m
    if not grid.fetched_ok or np.isnan(z).all():
        return BathyAnalysis(ok=False, note="bathymetry unavailable")
    water = z < -0.3
    land = ~water & np.isfinite(z)
    jj, ii = np.mgrid[0:n, 0:n]
    c = n // 2
    dist_center = np.hypot(ii - c, jj - c) * step

    # shoreline = land cells touching water, within walking distance of the spot
    pad = np.pad(water, 1, constant_values=False)
    touches = np.zeros_like(water)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            touches |= pad[1 + dj:1 + dj + n, 1 + di:1 + di + n]
    shore = land & touches & (dist_center <= walk_m)
    if not shore.any():
        return BathyAnalysis(ok=False, resolution_m=grid.resolution_m,
                             note="no shoreline found near spot in DEM")
    sj, si = np.nonzero(shore)

    # distance from every cell to the nearest reachable shoreline cell
    d_shore = np.full(z.shape, np.inf)
    for a, b in zip(sj, si):
        d_shore = np.minimum(d_shore, np.hypot(ii - b, jj - a) * step)
    cast = water & (d_shore <= cast_m)

    gy, gx = np.gradient(np.where(np.isfinite(z), z, np.nanmean(z)), step)
    slope = np.hypot(gx, gy)  # m/m

    # facing derived from the mean direction toward water around the spot
    near = water & (dist_center <= 160)
    derived = None
    if near.any():
        e = float(np.mean(ii[near] - c)); nn = float(np.mean(jj[near] - c))
        derived = (math.degrees(math.atan2(e, nn)) + 360) % 360

    # trough test: deeper than the cell ~40 m further seaward (water behind a bar)
    k = max(1, int(round(40 / step)))
    fe, fn = math.sin(math.radians(facing_deg)), math.cos(math.radians(facing_deg))
    oj = np.clip(jj + int(round(fn * k)), 0, n - 1)
    oi = np.clip(ii + int(round(fe * k)), 0, n - 1)
    seaward = z[oj, oi]

    depth = -z
    kinds = np.full(z.shape, "", dtype=object)
    kinds[cast & (slope >= 0.06) & (depth >= 1.2)] = "drop-off"
    kinds[cast & (depth >= 4.0)] = "deep channel"
    kinds[cast & (kinds == "") & (depth >= 0.8) & (seaward > z + 0.35)] = "trough"
    kinds[cast & (kinds == "") & (depth < 0.8) & (seaward < z - 0.3)] = "bar / shoal"

    # target score: slope + depth + trough, with a preference for shorter casts
    tscore = (np.clip(slope / 0.12, 0, 1) * 0.45 + np.clip(depth / 6, 0, 1) * 0.35
              + (kinds == "trough") * 0.25 + (kinds == "drop-off") * 0.1
              - np.clip(d_shore / cast_m, 0, 1) * 0.15)
    tscore = np.where(cast, tscore, -np.inf)

    cells = []
    for kind_name in ("drop-off", "deep channel", "trough", "bar / shoal"):
        mj, mi = np.nonzero(kinds == kind_name)
        order = np.argsort(-tscore[mj, mi])[:70]
        for a, b in zip(mj[order], mi[order]):
            la, lo = grid.cell_latlon(int(a), int(b))
            cells.append({"lat": la, "lon": lo, "kind": kind_name, "depth_m": round(float(depth[a, b]), 1),
                          "slope_pct": round(float(slope[a, b] * 100), 1)})

    # top targets, spaced >= 60 m apart
    targets: list[dict[str, Any]] = []
    flat = np.argsort(-tscore, axis=None)
    for idx in flat[:2000]:
        a, b = divmod(int(idx), n)
        if not np.isfinite(tscore[a, b]):
            break
        if any(math.hypot(a - t["_j"], b - t["_i"]) * step < 60 for t in targets):
            continue
        la, lo = grid.cell_latlon(a, b)
        kind_name = kinds[a, b] or "edge"
        targets.append({"_j": a, "_i": b, "lat": la, "lon": lo, "kind": kind_name,
                        "depth_m": round(float(depth[a, b]), 1), "slope_pct": round(float(slope[a, b] * 100), 1),
                        "cast_m": round(float(d_shore[a, b])), "score": round(float(tscore[a, b]), 3)})
        if len(targets) >= 3:
            break
    for t in targets:
        t.pop("_j"); t.pop("_i")

    deep3 = water & (depth >= 3)
    dist3 = float(np.min(dist_center[deep3])) if deep3.any() else None
    structures = {k_: int((kinds == k_).sum()) for k_ in ("drop-off", "deep channel", "trough", "bar / shoal")}
    return BathyAnalysis(
        ok=True, resolution_m=round(grid.resolution_m, 1) if grid.resolution_m else None,
        max_depth_cast_m=round(float(np.nanmax(np.where(cast, depth, np.nan))), 1) if cast.any() else None,
        max_slope_pct=round(float(np.nanmax(np.where(cast, slope, np.nan)) * 100), 1) if cast.any() else None,
        dist_to_3m_m=round(dist3) if dist3 is not None else None,
        derived_facing_deg=round(derived) if derived is not None else None,
        structures=structures, targets=targets, cells=cells,
        note=f"Grid {n}x{n} @ {step:.0f} m sampling of a ~{grid.resolution_m or 0:.0f} m DEM. Casting reach {cast_m:.0f} m, walking range {walk_m:.0f} m.",
    )


def structure_bonus(analysis: BathyAnalysis | None) -> dict[str, float]:
    """0-1 boosts for structure tags confirmed by bathymetry."""
    if not analysis or not analysis.ok:
        return {}
    s = analysis.structures
    out = {}
    if s.get("drop-off", 0) >= 3 or (analysis.max_slope_pct or 0) >= 8:
        out["channel_edge"] = min(1.0, 0.6 + s.get("drop-off", 0) / 40)
    if s.get("trough", 0) >= 3:
        out["trough"] = min(1.0, 0.6 + s["trough"] / 40)
    if s.get("bar / shoal", 0) >= 3:
        out["sandbar"] = 0.75
    if s.get("deep channel", 0) >= 3:
        out["channel_edge"] = max(out.get("channel_edge", 0), 0.85)
    return out
