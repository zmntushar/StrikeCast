"""Normalized schemas shared by every provider and model.

Conventions
-----------
* All timestamps are timezone-aware UTC internally; convert only for display.
* Units are US-angler units and are stated in ``UNITS``.
* Every value carries (via its column/source) a ``DataKind`` so measured,
  predicted, model, interpolated and heuristic values are never mixed up.
* Missing values are explicit ``None`` / NaN — never filled with fake numbers.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DataKind(str, Enum):
    measured = "measured"          # instrument observation (buoy, tide gauge)
    predicted = "predicted"        # official prediction (NOAA tide/current tables, NWS forecast)
    model = "model"                # numerical weather/ocean model or reanalysis
    interpolated = "interpolated"  # derived between official values (e.g. hourly tide curve)
    computed = "computed"          # deterministic astronomy / geometry
    heuristic = "heuristic"        # rule-based estimate or index
    missing = "missing"


UNITS: dict[str, str] = {
    "air_temp_f": "°F", "wind_mph": "mph", "gust_mph": "mph", "wind_dir": "° from",
    "pressure_hpa": "hPa", "precip_in": "in", "cloud_pct": "%", "visibility_mi": "mi",
    "pop_pct": "%", "wave_ft": "ft", "wave_period_s": "s", "wave_dir": "° from",
    "swell_ft": "ft", "swell_period_s": "s", "swell_dir": "° from",
    "nearshore_wave_ft": "ft", "sst_model_f": "°F", "water_temp_f": "°F",
    "tide_ft": "ft MLLW", "tide_rate_ft_h": "ft/h", "current_kt": "kt (+flood/−ebb)",
    "sun_alt": "°", "moon_alt": "°", "moon_illum": "%", "onshore_mph": "mph",
    "alongshore_mph": "mph", "pressure_tend_3h": "hPa/3h", "water_level_anom_ft": "ft",
}


class Spot(BaseModel):
    id: str
    name: str
    short: str
    favorite: bool = False
    region: str = ""
    lat: float
    lon: float
    environment: str
    water_body: str
    facing_deg: float
    # Deprecated: the old 0-1 surf multiplier. Kept so older configs still load;
    # surf is now computed from geometry (configs/exposure.yaml) and wave physics.
    exposure: float | None = None
    # open_ocean | sheltered_bay | sound. Derived from water_body when omitted.
    wave_exposure: str | None = None
    # Offshore wave-model point [lat, lon] (default: 12 km out along facing_deg).
    wave_point: list[float] | None = None
    marine_zone: str | None = None     # NWS coastal-waters zone, e.g. ANZ353
    surf_zone: str | None = None       # NWS surf-zone forecast zone, e.g. NYZ080
    bottom: str = "sand"
    structure: list[str] = Field(default_factory=list)
    tide_station: str | None = None
    current_station: str | None = None
    current_is_proxy: bool = True
    temp_sources: list[str] = Field(default_factory=list)
    wave_sources: list[str] = Field(default_factory=list)
    access: str = ""
    notes: str = ""
    timezone: str = "America/New_York"
    custom: bool = False

    @property
    def exposure_class(self) -> str:
        if self.wave_exposure:
            return self.wave_exposure
        return {"ocean": "open_ocean", "bay": "sheltered_bay", "sound": "sound"}.get(self.water_body, "sheltered_bay")

    @property
    def is_open_ocean(self) -> bool:
        return self.exposure_class == "open_ocean"

    def boundary_point(self, dist_km: float = 12.0) -> tuple[float, float]:
        """Where offshore wave conditions are taken for the nearshore transformation."""
        if self.wave_point:
            return float(self.wave_point[0]), float(self.wave_point[1])
        from app.core.geo import destination
        return destination(self.lat, self.lon, self.facing_deg, dist_km)


class SourceInfo(BaseModel):
    """Provenance record for one dataset feeding a forecast."""

    feature: str                     # e.g. "water temperature"
    provider: str                    # e.g. "NOAA NDBC"
    product: str = ""
    station_id: str | None = None
    station_name: str | None = None
    distance_km: float | None = None
    kind: DataKind = DataKind.measured
    fetched_at: datetime | None = None
    observed_at: datetime | None = None
    age_hours: float | None = None
    ok: bool = True
    stale: bool = False
    note: str = ""
    url: str = ""


class TideEvent(BaseModel):
    time: datetime
    height_ft: float
    type: str  # "H" | "L"


class CurrentEvent(BaseModel):
    time: datetime
    velocity_kt: float          # signed: + flood, - ebb, 0 slack
    type: str                   # flood | ebb | slack
    direction_deg: float | None = None


class FactorScore(BaseModel):
    """One explainable component of an index."""

    factor: str
    score: float                 # 0..1 (1 = most favorable)
    weight: float
    detail: str = ""

    @property
    def contribution(self) -> float:
        return self.weight * (self.score - 0.5)


class Window(BaseModel):
    species: str
    start: datetime
    end: datetime
    peak: datetime
    peak_score: float
    mean_score: float
    summary: str = ""


class Alert(BaseModel):
    event: str
    headline: str
    severity: str = ""
    onset: datetime | None = None
    ends: datetime | None = None


class ApiSpotForecast(BaseModel):
    """JSON-friendly forecast payload served by the FastAPI layer."""

    spot: Spot
    generated_at: datetime
    timezone: str
    quality: dict[str, Any]
    current: dict[str, Any]
    windows: list[Window]
    alerts: list[Alert]
    sources: list[SourceInfo]
    hourly: list[dict[str, Any]]
    bathymetry: dict[str, Any] | None = None
    surf: dict[str, Any] | None = None       # exposure class, buoy anchoring, geometry summary, latest buoy
    marine: dict[str, Any] | None = None     # NWS zone forecast headlines / advisories / surf-zone forecast
