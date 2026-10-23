"""Application settings (environment variables / .env) and YAML config loading."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration. Every field can be overridden with STRIKECAST_<NAME>."""

    model_config = SettingsConfigDict(
        env_prefix="STRIKECAST_", env_file=ROOT_DIR / ".env", extra="ignore"
    )

    app_name: str = "StrikeCast"
    # NWS and Nominatim ask for a User-Agent that identifies the app and a contact.
    contact: str = Field(default="https://github.com/strikecast/strikecast")
    database_url: str = f"sqlite:///{(ROOT_DIR / 'data' / 'strikecast.db').as_posix()}"
    configs_dir: Path = ROOT_DIR / "configs"
    default_timezone: str = "America/New_York"
    log_level: str = "INFO"

    http_timeout_s: float = 25.0
    http_max_retries: int = 3
    http_max_concurrency: int = 8
    cache_enabled: bool = True

    # Cache lifetimes (seconds) per provider product — tuned to update intervals.
    ttl_openmeteo_forecast: int = 3600
    ttl_openmeteo_marine: int = 3600
    ttl_openmeteo_history: int = 7 * 86400
    ttl_nws_points: int = 30 * 86400
    ttl_nws_hourly: int = 3600
    ttl_nws_alerts: int = 600
    ttl_coops_predictions: int = 24 * 3600
    ttl_coops_observations: int = 1800
    ttl_ndbc_realtime: int = 1800
    ttl_bathymetry: int = 90 * 86400
    ttl_geocode: int = 90 * 86400

    # Scheduler (FastAPI process): refresh favorites in the background.
    scheduler_enabled: bool = True
    scheduler_interval_min: int = 60

    forecast_days: int = 7
    history_hours: int = 48


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _load_yaml(name: str) -> dict[str, Any]:
    path = get_settings().configs_dir / name
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache
def spots_config() -> dict[str, Any]:
    return _load_yaml("spots.yaml")


@lru_cache
def stations_config() -> dict[str, Any]:
    return _load_yaml("stations.yaml")


@lru_cache
def species_config() -> dict[str, Any]:
    return _load_yaml("species.yaml")


@lru_cache
def regulations_config() -> dict[str, Any]:
    return _load_yaml("regulations.yaml")


def user_agent() -> str:
    s = get_settings()
    return f"{s.app_name}/1.0 ({s.contact})"
