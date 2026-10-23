"""Forecast orchestration: fetch every provider concurrently, build hourly
features per spot, then run the baitfish / presence / feeding / bite engines.

The same pipeline powers the live 7-day forecast and historical "hindcasts"
used by the date-comparison feature (``mode="historical"``).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd

from app.core.config import exposure_config, get_settings, spots_config, stations_config
from app.core.geo import destination
from app.core.http import HttpClient
from app.core.logging import get_logger, log_kv
from app.features.tide_features import current_features, current_proxy_from_tide, tide_features
from app.features import surf as surf_phys
from app.features.weather_features import clarity_index, pressure_tendency, weather_flags, wind_features
from app.models import bite as bite_mod
from app.models.baitfish import bait_scores
from app.models.common import species_list
from app.models.feeding import feeding_scores
from app.models.presence import bait_availability, presence_scores
from app.models.storms import detect_storms
from app.providers import geography, lunar, marine, nws_marine, tides, weather
from app.schemas.env import Alert, DataKind, SourceInfo, Spot, TideEvent, Window
from app.services import surf as surf_svc
from app.services.stations import distance_to, station_meta

log = get_logger("forecast")
ProgressCb = Callable[[float, str], None]

NUMERIC_COLS = ["air_temp_f", "wind_mph", "wind_dir", "gust_mph", "pressure_hpa", "precip_in", "cloud_pct",
                "visibility_mi", "pop_pct", "wave_ft", "wave_period_s", "wave_dir", "swell_ft", "swell_period_s",
                "swell_dir", "sst_model_f", "tide_ft", "tide_rate_ft_h", "tide_frac", "hours_since_turn",
                "hours_to_turn", "tide_range_ft", "current_kt", "current_norm", "water_level_anom_ft", "buoy_wave_ft",
                "windwave_ft", "windwave_period_s", "windwave_dir", "swell_peak_period_s", "wave_peak_period_s"]
TEXT_COLS = ["short_forecast", "tide_dir", "tide_stage", "tide_stage_tag", "next_turn_type", "current_phase"]


# ------------------------------------------------------------------ spots
def load_spots() -> list[Spot]:
    cfg = spots_config()
    tz = cfg.get("timezone", get_settings().default_timezone)
    return [Spot(**{**s, "timezone": tz}) for s in cfg["spots"]]


def get_spot(spot_id: str) -> Spot:
    for s in load_spots():
        if s.id == spot_id:
            return s
    raise KeyError(spot_id)


# --------------------------------------------------------------- framing
@dataclass
class TimeFrame:
    mode: str                      # "forecast" | "historical"
    start: pd.Timestamp            # first hour fetched (includes history for trends)
    end: pd.Timestamp
    display_start: pd.Timestamp
    display_end: pd.Timestamp
    target_date: date | None = None

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end, freq="1h", tz="UTC")


def forecast_frame(days: int | None = None, now: datetime | None = None) -> TimeFrame:
    s = get_settings()
    days = days or s.forecast_days
    now_h = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC").floor("h")
    return TimeFrame("forecast", now_h - pd.Timedelta(hours=96), now_h + pd.Timedelta(days=days),
                     now_h, now_h + pd.Timedelta(days=days))


def historical_frame(day: date, tz: str) -> TimeFrame:
    d0 = pd.Timestamp(day).tz_localize(tz).tz_convert("UTC")
    # next LOCAL midnight (23 or 25 hours later on daylight-saving change days)
    d1 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC")
    return TimeFrame("historical", d0 - pd.Timedelta(hours=96), d1, d0, d1 - pd.Timedelta(hours=1), day)


# ---------------------------------------------------------------- results
@dataclass
class SpotForecast:
    spot: Spot
    frame: TimeFrame
    hourly: pd.DataFrame                       # full index incl. history (UTC)
    sources: list[SourceInfo]
    alerts: list[Alert]
    bathy: geography.BathyAnalysis | None
    tide_events: list[TideEvent]
    daily: list[dict[str, Any]]
    windows: dict[str, list[Window]]
    quality: dict[str, Any]
    missing: list[str] = field(default_factory=list)
    surf: dict[str, Any] = field(default_factory=dict)        # surf model context / validation summary
    marine: dict[str, Any] = field(default_factory=dict)      # NWS zone forecast, advisories, surf-zone forecast
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def display(self) -> pd.DataFrame:
        return self.hourly.loc[self.frame.display_start:self.frame.display_end]

    def now_row(self) -> pd.Series:
        d = self.display
        return d.iloc[0] if not d.empty else self.hourly.iloc[-1]

    def bite_now(self, species: str) -> float:
        return float(self.now_row().get(f"bite_{species}", np.nan))

    def latest_buoy(self, max_age_h: float = 6.0) -> dict[str, Any] | None:
        """Most recent measured offshore buoy reading (the current hour often has none yet)."""
        h = self.hourly
        if "buoy_wave_ft" not in h:
            return None
        upto = min(pd.Timestamp.now(tz="UTC"), self.frame.display_end)
        s = h["buoy_wave_ft"].loc[:upto].dropna()
        if self.frame.mode == "historical":
            s = s.loc[self.frame.display_start:]
        if s.empty:
            return None
        ts = s.index[-1]
        age = (pd.Timestamp.now(tz="UTC") - ts) / pd.Timedelta(hours=1)
        if self.frame.mode == "forecast" and age > max_age_h:
            return None
        r = h.loc[ts]
        src = next((x for x in self.sources if x.feature == "offshore waves (measured)"), None)
        g = lambda k: float(r[k]) if k in r and pd.notna(r[k]) else np.nan  # noqa: E731
        return {"time": ts, "age_h": round(age, 1), "station": src.station_id if src else None,
                "station_name": src.station_name if src else None, "distance_km": src.distance_km if src else None,
                "hs_ft": g("buoy_wave_ft"), "period_s": g("buoy_period_s"), "dir": g("buoy_wave_dir"),
                "swell_ft": g("buoy_swell_ft"), "swell_s": g("buoy_swell_period_s"), "swell_dir": g("buoy_swell_dir"),
                "ww_ft": g("buoy_windwave_ft"), "ww_s": g("buoy_windwave_period_s"), "ww_dir": g("buoy_windwave_dir")}

    def best_today(self, species: str) -> float:
        d = self.display
        local = d.index.tz_convert(self.spot.timezone)
        today = local.normalize() == local[0].normalize()
        col = f"bite_{species}"
        return float(d.loc[today, col].max()) if col in d and today.any() else np.nan


class Progress:
    def __init__(self, cb: ProgressCb | None, total: int) -> None:
        self.cb, self.total, self.done = cb, max(total, 1), 0

    def step(self, label: str) -> None:
        self.done += 1
        if self.cb:
            try:
                self.cb(min(self.done / self.total, 1.0), label)
            except Exception:  # noqa: BLE001 — UI callbacks must never break data collection
                pass


# ----------------------------------------------------------------- fetch
async def _fetch_all(http: HttpClient, spots: list[Spot], frame: TimeFrame, prog: Progress,
                     extra_steps: int) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    tasks: list[tuple[str, str, Awaitable[Any]]] = []
    hist = frame.mode == "historical"
    d_begin = (frame.start - pd.Timedelta(days=1)).date()
    d_end = (frame.end + pd.Timedelta(days=1)).date()
    obs_begin = frame.start.date()
    obs_end = min(frame.end, pd.Timestamp.now(tz="UTC")).date()

    wx_points = [(s.lat, s.lon) for s in spots]
    # Ocean spots: offshore point ~12 km out (20-30 m deep), the boundary for the surf
    # transformation. Sheltered spots: the nearest water cell (SST / model context only).
    mar_points = [s.boundary_point() if s.is_open_ocean else destination(s.lat, s.lon, s.facing_deg, 1.0) for s in spots]
    # The same request also returns the model at each wave buoy, for buoy-vs-model anchoring.
    wave_buoys = sorted({src for s in spots for src in s.wave_sources if src.startswith("ndbc:")})
    raw["_marine_buoys"] = {}
    for src in wave_buoys:
        meta = station_meta(src)
        if "lat" in meta:
            raw["_marine_buoys"][src] = len(mar_points)
            mar_points.append((float(meta["lat"]), float(meta["lon"])))
    if hist:
        a, b = frame.start.date(), frame.end.date()
        tasks.append(("wx", "Open-Meteo weather archive", weather.openmeteo_weather(http, wx_points, start=a, end=b)))
        tasks.append(("marine", "Open-Meteo marine archive", marine.openmeteo_marine(http, mar_points, start=a, end=b)))
    else:
        days = int((frame.end - frame.display_start) / pd.Timedelta(days=1)) + 1
        tasks.append(("wx", "Open-Meteo weather forecast", weather.openmeteo_weather(http, wx_points, past_days=4, forecast_days=min(days, 16))))
        tasks.append(("marine", "Open-Meteo marine forecast", marine.openmeteo_marine(http, mar_points, past_days=4, forecast_days=min(days, 16))))
        hourly_memo: dict[str, asyncio.Task] = {}

        async def nws_bundle(sp: Spot):
            props = await weather.nws_point(http, sp.lat, sp.lon)
            url = props.get("forecastHourly")
            hourly = None
            if url:
                if url not in hourly_memo:
                    hourly_memo[url] = asyncio.ensure_future(weather.nws_hourly(http, url))
                hourly = await hourly_memo[url]
            alerts = await weather.nws_alerts(http, sp.lat, sp.lon)
            return hourly, alerts, props

        for sp in spots:
            tasks.append((f"nws:{sp.id}", f"NWS forecast & alerts · {sp.short}", nws_bundle(sp)))
        mz = sorted({s.marine_zone for s in spots if s.marine_zone})
        if mz:
            tasks.append(("cwf", "NWS coastal waters forecast", nws_marine.latest_cwf(http, mz)))
        if any(s.surf_zone for s in spots):
            tasks.append(("srf", "NWS surf zone forecast", nws_marine.surf_zone_forecasts(http)))
    if hist:
        # archived NWS products for past dates (never today's forecast)
        if any(s.marine_zone for s in spots):
            tasks.append(("cwf_arch", "NWS coastal waters forecast (archive)",
                          nws_marine.archived_products(http, "CWFOKX", (frame.display_start - pd.Timedelta(hours=18)).to_pydatetime(),
                                                       frame.end.to_pydatetime())))
        if any(s.surf_zone for s in spots):
            tasks.append(("srf", "NWS surf zone forecast (archive)",
                          nws_marine.surf_zone_forecasts(http, (frame.display_start - pd.Timedelta(hours=36)).to_pydatetime(),
                                                         frame.end.to_pydatetime())))
        zones = sorted({z for s in spots for z in (s.marine_zone, s.surf_zone) if z})
        for z in zones:
            tasks.append((f"vtec:{z}", f"NWS advisories archive · {z}",
                          # advisories issued days earlier can still be in effect on the target date
                          nws_marine.advisories(http, z, (frame.display_start - pd.Timedelta(days=10)).date(),
                                                (frame.display_end + pd.Timedelta(days=1)).date())))

    for st in sorted({s.tide_station for s in spots if s.tide_station}):
        name = station_meta(st).get("name", st)
        tasks.append((f"tide:{st}", f"NOAA tide predictions · {name}", tides.tide_hilo(http, st, d_begin, d_end)))
    for st in sorted({s.current_station for s in spots if s.current_station}):
        meta = station_meta(st)
        tasks.append((f"cur:{st}", f"NOAA current predictions · {meta.get('name', st)}",
                      tides.current_events(http, st, d_begin, d_end, int(meta.get("bin", 1) or 1))))
    obs_sources = sorted({src for s in spots for src in (s.temp_sources + s.wave_sources)})
    for src in obs_sources:
        kind, sid = src.split(":", 1)
        name = station_meta(src).get("name", sid)
        if kind == "ndbc":
            coro = marine.ndbc_observations(http, sid, start=frame.start.to_pydatetime(), end=frame.end.to_pydatetime())
            tasks.append((src, f"NDBC buoy {sid} · {name}", coro))
        else:
            tasks.append((src, f"NOAA water temperature · {name}", tides.water_temperature(http, sid, obs_begin, obs_end)))
    for src in wave_buoys:
        sid = src.split(":", 1)[1]
        if frame.start.to_pydatetime() > datetime.now(timezone.utc) - pd.Timedelta(days=44):
            tasks.append((f"spec:{src}", f"NDBC swell / wind-wave split · {sid}",
                          marine.ndbc_spectral(http, sid, start=frame.start.to_pydatetime(), end=frame.end.to_pydatetime())))
    expo = exposure_config()
    for sp in spots:
        if sp.id not in expo:
            tasks.append((f"expo:{sp.id}", f"Wave exposure rays · {sp.short}",
                          geography.exposure_rays(http, sp.lat, sp.lon, sp.boundary_point() if sp.is_open_ocean else None)))
    refs = stations_config().get("water_level_reference", {})
    for st in sorted({refs.get(s.water_body) for s in spots if refs.get(s.water_body)}):
        tasks.append((f"wl:{st}", f"NOAA water level anomaly · {station_meta(st).get('name', st)}",
                      tides.water_level_anomaly(http, st, obs_begin, obs_end)))
    for sp in spots:
        tasks.append((f"bathy:{sp.id}", f"NOAA bathymetry · {sp.short}", geography.fetch_grid(http, sp.lat, sp.lon)))

    prog.total = len(tasks) + extra_steps

    async def run(key: str, label: str, coro: Awaitable[Any]) -> None:
        try:
            raw[key] = await coro
        except Exception as exc:  # noqa: BLE001 — record the failure, keep going
            raw[key] = exc
            log_kv(log, logging.WARNING, "provider failed", key=key, error=str(exc)[:200])
        finally:
            prog.step(label)

    await asyncio.gather(*(run(k, lbl, c) for k, lbl, c in tasks))
    return raw


# --------------------------------------------------------------- assemble
def _ok(raw: dict, key: str) -> Any | None:
    v = raw.get(key)
    return None if v is None or isinstance(v, Exception) else v


def _err(raw: dict, key: str) -> str:
    v = raw.get(key)
    return str(v)[:160] if isinstance(v, Exception) else "not requested"


def _floatify(df: pd.DataFrame) -> pd.DataFrame:
    """Cast integer columns (e.g. API degree values) to float so measured values can be merged."""
    ints = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    return df.astype({c: "float64" for c in ints}) if ints else df


def _hourly(obs: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Hourly means; direction columns (*_dir) use a circular mean so 350 and 10 deg average to 0."""
    keep = [c for c in cols if c in obs]
    if not keep:
        return pd.DataFrame()
    lin = [c for c in keep if not c.endswith("_dir")]
    out = obs[lin].astype(float).resample("1h").mean() if lin else pd.DataFrame(index=obs.resample("1h").mean().index)
    for c in keep:
        if c.endswith("_dir"):
            r = np.radians(obs[c].astype(float))
            s, co = np.sin(r).resample("1h").mean(), np.cos(r).resample("1h").mean()
            ang = (np.degrees(np.arctan2(s, co)) + 360) % 360
            out[c] = ang.where(s.notna() & co.notna())
    return out


def assemble(spot: Spot, frame: TimeFrame, raw: dict[str, Any], spot_pos: int,
             species: list[str] | None = None) -> SpotForecast:
    species = species or species_list()
    idx = frame.index
    now = pd.Timestamp.now(tz="UTC").floor("h")
    df = pd.DataFrame(index=idx)
    sources: list[SourceInfo] = []
    missing: list[str] = []
    alerts: list[Alert] = []

    # ---- weather (model) ------------------------------------------------
    wx = _ok(raw, "wx")
    if wx and spot_pos < len(wx) and not wx[spot_pos][0].empty:
        wdf, src = wx[spot_pos]
        df = df.join(wdf.reindex(idx))
        sources.append(src)
    else:
        missing.append("weather")
        sources.append(SourceInfo(feature="weather", provider="Open-Meteo", kind=DataKind.missing, ok=False, note=_err(raw, "wx")))

    # ---- NWS official forecast (overrides model wind for future hours) --
    nws = _ok(raw, f"nws:{spot.id}")
    if frame.mode == "forecast":
        if nws and nws[0] is not None and not nws[0][0].empty:
            ndf, nsrc = nws[0]
            ndf = ndf[~ndf.index.duplicated()].reindex(idx)
            fut = idx >= now
            for c_n, c_m in (("nws_wind_mph", "wind_mph"), ("nws_wind_dir", "wind_dir")):
                if c_m in df:
                    df.loc[fut, c_m] = ndf.loc[fut, c_n].combine_first(df.loc[fut, c_m])
            df["pop_pct"] = ndf["pop_pct"]
            df["short_forecast"] = ndf["short_forecast"]
            if "gust_mph" in df and "wind_mph" in df:
                df["gust_mph"] = np.fmax(df["gust_mph"], df["wind_mph"])
            nsrc.note = "NWS wind speed/direction used for forecast hours; gusts, pressure, clouds from Open-Meteo"
            sources.append(nsrc)
            alerts = nws[1]
        else:
            missing.append("NWS forecast")
            sources.append(SourceInfo(feature="official hourly forecast", provider="NOAA NWS", kind=DataKind.missing,
                                      ok=False, note=_err(raw, f"nws:{spot.id}")))

    # ---- marine model ---------------------------------------------------
    mar = _ok(raw, "marine")
    if mar and spot_pos < len(mar) and not mar[spot_pos][0].empty:
        mdf, msrc = mar[spot_pos]
        df = df.join(mdf.reindex(idx))
        sources.append(msrc)
    else:
        missing.append("wave model")
        sources.append(SourceInfo(feature="waves / SST model", provider="Open-Meteo Marine", kind=DataKind.missing, ok=False))
    df = _floatify(df)
    df["wave_kind"] = "model"

    # ---- wave buoy (measured, offshore) ----------------------------------
    picked_wave = False
    wave_anchor: dict[str, Any] = {}
    for src in spot.wave_sources:
        obs = _ok(raw, src)
        if obs is None:
            continue
        odf, url = obs
        h = _hourly(odf, ["wave_ft", "wave_period_s", "wave_dir"]).reindex(idx)
        if "wave_ft" not in h or h["wave_ft"].notna().sum() < 3:
            continue
        meta = station_meta(src)
        dist, _ = distance_to(spot, meta)
        last_t = odf["wave_ft"].dropna().index.max()
        age = (min(now, frame.end) - last_t).total_seconds() / 3600 if last_t is not None else None
        if frame.mode == "forecast" and age is not None and age > 6:
            continue          # stale: try the next buoy
        if frame.mode == "historical" and h.loc[frame.display_start:frame.display_end, "wave_ft"].notna().sum() < 3:
            continue          # no measurements on that date: never substitute other days
        # Offshore buoys are kept separate from the nearshore estimate: they are
        # an offshore reference (and a check on the model), not the surf.
        df["buoy_wave_ft"] = h["wave_ft"]
        df["buoy_period_s"] = h.get("wave_period_s")
        df["buoy_wave_dir"] = h.get("wave_dir")
        spec = _ok(raw, f"spec:{src}")
        if spec is not None:
            sh = _hourly(spec[0], ["swell_ft", "swell_period_s", "swell_dir", "windwave_ft", "windwave_period_s",
                                   "windwave_dir"]).reindex(idx)
            for c in sh:
                df[f"buoy_{c}"] = sh[c]
        bi = raw.get("_marine_buoys", {}).get(src)
        mar_all = _ok(raw, "marine")
        if spot.is_open_ocean and bi is not None and mar_all and bi < len(mar_all) and not mar_all[bi][0].empty:
            mb = mar_all[bi][0].reindex(idx)
            wave_anchor = surf_svc.anchor_to_buoy(df["buoy_wave_ft"], mb["wave_ft"],
                                                  now if frame.mode == "forecast" else frame.end, frame.mode,
                                                  buoy_dir=df["buoy_wave_dir"], model_dir=mb.get("wave_dir"))
            wave_anchor.update(buoy=meta["id"], buoy_name=meta.get("name"), distance_km=dist)
        if wave_anchor.get("applied"):
            diff = (f" Model at the buoy {wave_anchor['bias_ft']:+.1f} ft vs measured (mean abs. error {wave_anchor['mae_ft']:.1f} ft);"
                    f" model heights scaled x{wave_anchor['ratio']:.2f}"
                    + (f", directions rotated {wave_anchor['dir_offset_deg']:+.0f} deg" if wave_anchor.get("dir_offset") is not None else "")
                    + f", relaxing to the {wave_anchor.get('ratio_longterm', 1):.2f} multi-day bias over ~12 h.")
        elif wave_anchor.get("reason"):
            diff = f" Not used to anchor the model: {wave_anchor['reason']}."
        else:
            diff = ""
        sources.append(SourceInfo(
            feature="offshore waves (measured)", provider="NOAA NDBC",
            product="realtime stdmet" + (" + spectral summary (swell / wind-wave split)" if spec is not None else ""),
            station_id=meta["id"], station_name=meta.get("name"), distance_km=dist, kind=DataKind.measured,
            observed_at=last_t.to_pydatetime(), age_hours=round(age, 1) if age is not None else None, url=url,
            stale=bool(frame.mode == "forecast" and age is not None and age > 3),
            note="Offshore buoy reference, not nearshore surf." + diff))
        picked_wave = True
        break
    if not picked_wave and spot.wave_sources:
        missing.append("measured waves")

    # ---- NWS marine zone forecast / surf zone forecast / advisories --------
    marine_info: dict[str, Any] = {"zone": spot.marine_zone, "surf_zone": spot.surf_zone, "advisories": [], "srf": []}
    if spot.marine_zone:
        zf, zurl, znote = None, "", ""
        if frame.mode == "forecast":
            cwf = _ok(raw, "cwf")
            if cwf:
                zf, zurl = cwf[0].get(spot.marine_zone), cwf[1]
            if zf:
                df = df.join(nws_marine.zone_hourly(zf, idx))
                marine_info.update(name=zf.name, issued=zf.issued, expires=zf.expires, headlines=zf.headlines,
                                   stale=zf.stale(now), periods=[(p.name, p.text) for p in zf.periods[:4]])
                znote = (" | ".join(zf.headlines) + " | " if zf.headlines else "") + f"issued {zf.issued:%Y-%m-%d %H:%MZ}"
        else:
            arch = _ok(raw, "cwf_arch")
            if arch:
                try:
                    zdf, heads, znote = nws_marine.cwf_hourly_from_products(arch, spot.marine_zone, idx)
                    df = df.join(zdf)
                    marine_info.update(headlines=heads, archived=True)
                    zf = True
                except Exception as exc:  # noqa: BLE001 - archive gaps are normal
                    znote = str(exc)[:120]
        if zf:
            sources.append(SourceInfo(feature="NWS marine zone forecast (seas, wave detail)", provider="NOAA NWS",
                                      product=f"Coastal Waters Forecast {spot.marine_zone}"
                                              + (" (archived)" if frame.mode == "historical" else ""),
                                      station_id=spot.marine_zone, kind=DataKind.predicted, url=zurl,
                                      stale=bool(marine_info.get("stale")),
                                      note=znote + (" | EXPIRED product" if marine_info.get("stale") else "")))
        else:
            sources.append(SourceInfo(feature="NWS marine zone forecast (seas, wave detail)", provider="NOAA NWS",
                                      station_id=spot.marine_zone, kind=DataKind.missing, ok=False,
                                      note=znote or _err(raw, "cwf" if frame.mode == "forecast" else "cwf_arch")))
    srf = _ok(raw, "srf")
    if srf and spot.surf_zone:
        days = {}
        for sd in srf:
            if sd.zone != spot.surf_zone:
                continue
            if frame.mode == "historical":
                day_start = pd.Timestamp(sd.day).tz_localize(spot.timezone).tz_convert("UTC")
                if sd.issued > day_start + pd.Timedelta(hours=8):
                    continue          # use forecasts issued before that morning only
            prev = days.get(sd.day)
            if prev is None or sd.issued > prev.issued:
                days[sd.day] = sd
        marine_info["srf"] = [{"day": d, "lo": v.surf_lo_ft, "hi": v.surf_hi_ft, "rip": v.rip_risk, "issued": v.issued}
                              for d, v in sorted(days.items())]
    for z in (spot.marine_zone, spot.surf_zone):
        if not z:
            continue
        for a in (_ok(raw, f"vtec:{z}") or []):
            if a["expire"] >= frame.display_start and a["issue"] <= frame.display_end + pd.Timedelta(hours=1):
                marine_info["advisories"].append(a)

    # ---- water temperature (measured anchor + model trend) ---------------
    picked_temp = None
    for src in spot.temp_sources:
        obs = _ok(raw, src)
        if obs is None:
            continue
        odf, url = obs
        if "water_temp_f" not in odf or odf["water_temp_f"].dropna().empty:
            continue
        series = odf["water_temp_f"].dropna()
        last_t = series.index.max()
        age = (min(now, frame.end) - last_t).total_seconds() / 3600
        if frame.mode == "forecast" and age > 12:
            continue
        if frame.mode == "historical" and series.loc[frame.display_start:frame.display_end].count() < 3:
            continue
        picked_temp = (src, series, last_t, age, url)
        break
    sst = df["sst_model_f"] if "sst_model_f" in df else pd.Series(np.nan, index=idx)
    if picked_temp:
        src, series, last_t, age, url = picked_temp
        hourly_t = series.resample("1h").mean().reindex(idx).interpolate(limit=6, limit_area="inside")
        anchor = float(series.loc[last_t - pd.Timedelta(hours=2):last_t].mean())
        sst_anchor = sst.loc[:last_t].dropna()
        trend = (sst - float(sst_anchor.iloc[-1])) if not sst_anchor.empty else 0.0
        fut = idx > last_t
        wt = hourly_t.copy()
        wt[fut] = anchor + (trend[fut] if isinstance(trend, pd.Series) else 0.0)
        wt = wt.ffill(limit=3)
        df["water_temp_f"] = wt
        df["water_temp_kind"] = np.where(fut, "measured + model trend", "measured")
        meta = station_meta(src)
        dist, _ = distance_to(spot, meta)
        sources.append(SourceInfo(
            feature="water temperature", provider="NOAA NDBC" if src.startswith("ndbc") else "NOAA CO-OPS",
            station_id=meta["id"], station_name=meta.get("name"), distance_km=dist, kind=DataKind.measured,
            observed_at=last_t.to_pydatetime(), age_hours=round(age, 1), url=url,
            note="Forecast hours = latest measurement + Open-Meteo SST trend"
                 + ("" if src == spot.temp_sources[0] else f" · substituted for {spot.temp_sources[0]} (unavailable)")))
    elif sst.notna().any():
        df["water_temp_f"] = sst
        df["water_temp_kind"] = "model"
        missing.append("measured water temperature")
        sources.append(SourceInfo(feature="water temperature", provider="Open-Meteo Marine", kind=DataKind.model,
                                  note="No fresh buoy/gauge reading — model sea-surface temperature used"))
    else:
        df["water_temp_f"] = np.nan
        df["water_temp_kind"] = "missing"
        missing.append("water temperature")
    df["water_temp_trend_3d"] = df["water_temp_f"] - df["water_temp_f"].shift(72)

    # ---- tides --------------------------------------------------------------
    tide_events: list[TideEvent] = []
    tr = _ok(raw, f"tide:{spot.tide_station}") if spot.tide_station else None
    if tr:
        tide_events, url = tr
        df = df.join(tide_features(tide_events, idx))
        meta = station_meta(spot.tide_station)
        dist, _ = distance_to(spot, meta)
        sources.append(SourceInfo(feature="tide times & heights", provider="NOAA CO-OPS", product="predictions (hilo)",
                                  station_id=spot.tide_station, station_name=meta.get("name"), distance_km=dist,
                                  kind=DataKind.predicted, url=url,
                                  note="Hourly curve interpolated between official highs/lows (cosine method)"))
    else:
        missing.append("tides")
        sources.append(SourceInfo(feature="tides", provider="NOAA CO-OPS", kind=DataKind.missing, ok=False,
                                  note=_err(raw, f"tide:{spot.tide_station}")))

    # ---- currents ----------------------------------------------------------
    cr = _ok(raw, f"cur:{spot.current_station}") if spot.current_station else None
    if cr and len(cr[0]) >= 2:
        events, url = cr
        df = df.join(current_features(events, idx))
        df["current_is_proxy"] = spot.current_is_proxy
        meta = station_meta(spot.current_station)
        dist, _ = distance_to(spot, meta)
        sources.append(SourceInfo(feature="tidal current", provider="NOAA CO-OPS", product="currents_predictions (max/slack)",
                                  station_id=spot.current_station, station_name=meta.get("name"), distance_km=dist,
                                  kind=DataKind.predicted, url=url,
                                  note=("Nearby inlet station used as a timing/strength PROXY for this shoreline. "
                                        if spot.current_is_proxy else "") + "Hourly values interpolated between official max/slack events."))
    elif "tide_rate_ft_h" in df:
        df["current_norm"] = current_proxy_from_tide(df)
        df["current_kt"] = np.nan
        df["current_phase"] = np.where(df["tide_dir"] == "incoming", "flood", "ebb")
        df["current_is_proxy"] = True
        sources.append(SourceInfo(feature="tidal current", provider="StrikeCast", kind=DataKind.heuristic,
                                  note="No current prediction station — strength estimated from rate of tide change (proxy)"))
    else:
        missing.append("currents")

    # ---- water level anomaly ---------------------------------------------
    ref = stations_config().get("water_level_reference", {}).get(spot.water_body)
    wl = _ok(raw, f"wl:{ref}") if ref else None
    if wl is not None:
        anom, url = wl
        df["water_level_anom_ft"] = anom.reindex(idx)
        meta = station_meta(ref)
        dist, _ = distance_to(spot, meta)
        last = anom.dropna().index.max() if not anom.dropna().empty else None
        sources.append(SourceInfo(feature="water-level anomaly (surge)", provider="NOAA CO-OPS", station_id=ref,
                                  station_name=meta.get("name"), distance_km=dist, kind=DataKind.measured,
                                  observed_at=last.to_pydatetime() if last is not None else None, url=url,
                                  note="Observed minus predicted at a reference gauge (regional indicator)"))

    # ---- astronomy -----------------------------------------------------------
    astro = lunar.hourly_astro(spot.lat, spot.lon, idx)
    df = df.join(astro)
    df["light_phase"] = lunar.light_phase(df["sun_alt"].to_numpy(), df["sun_alt"].shift(1).bfill().to_numpy())
    local_days = sorted({ts.date() for ts in frame.display_start.tz_convert(spot.timezone) +
                         pd.to_timedelta(np.arange(0, max(1, int((frame.display_end - frame.display_start) / pd.Timedelta(hours=1)) + 1), 24), unit="h")})
    daily = [lunar.daily_events(spot.lat, spot.lon, d, spot.timezone) for d in local_days]
    sources.append(SourceInfo(feature="sun & moon", provider="PyEphem (local computation)", kind=DataKind.computed))

    # ---- derived features ------------------------------------------------------
    # Missing providers leave explicit empty columns (never invented values).
    for c in NUMERIC_COLS:
        if c not in df:
            df[c] = np.nan
    for c in TEXT_COLS:
        if c not in df:
            df[c] = None
    if "current_is_proxy" not in df:
        df["current_is_proxy"] = True
    # ---- bathymetry ------------------------------------------------------------
    bathy = None
    grid = _ok(raw, f"bathy:{spot.id}")
    if grid is not None:
        bathy = geography.analyze(grid, spot.facing_deg)
        bathy.nearest_inlet = geography.nearest_feature(spot.lat, spot.lon, "inlets")
        bathy.nearest_river = geography.nearest_feature(spot.lat, spot.lon, "river_mouths")
        sources.append(SourceInfo(feature="bathymetry / structure", provider="NOAA NCEI DEM", kind=DataKind.measured,
                                  note=f"{bathy.resolution_m or '?'} m resolution; {bathy.note}"))
    else:
        missing.append("bathymetry")
    bonus = geography.structure_bonus(bathy)

    df = df.join(pressure_tendency(df["pressure_hpa"]))
    df = df.join(wind_features(df["wind_mph"].astype(float), df["wind_dir"].astype(float), spot.facing_deg))

    # ---- surf (offshore waves -> nearshore, or wind chop in sheltered water) --
    expo = exposure_config().get(spot.id)
    if expo is None and _ok(raw, f"expo:{spot.id}") is not None:
        expo = raw[f"expo:{spot.id}"].to_dict()
    geom = surf_phys.Geometry.from_dict(expo) if expo else None
    profile: dict[str, Any] = {}
    if grid is not None:
        prof = geography.cross_shore_profile(grid, spot.facing_deg)
        if prof is not None:
            profile = geography.profile_features(*prof)
    mar_src = next((s for s in sources if s.feature.startswith("waves / swell")), None)
    ctx = surf_svc.SurfContext(spot=spot, geometry=geom, boundary_depth_m=(expo or {}).get("boundary_depth_m"),
                               profile=profile, anchor=wave_anchor, now=now, mode=frame.mode,
                               model_stale=bool(mar_src and mar_src.stale))
    surf_df, surf_meta = surf_svc.compute_surf(df, ctx)
    df = df.join(surf_df)
    surf_meta.update(geometry=expo, boundary_point=spot.boundary_point() if spot.is_open_ocean else None)
    if df.loc[frame.display_start:frame.display_end, "surf_ft"].isna().all():
        missing.append("surf estimate")
    sources.append(SourceInfo(
        feature="nearshore surf (computed)", provider="StrikeCast surf model", kind=DataKind.model,
        ok=bool(df["surf_ft"].notna().any()),
        product=("shoaling + refraction + sheltering + depth-limited breaking" if spot.is_open_ocean
                 else "SPM fetch-limited wind waves (sheltered water)"),
        note=("Offshore input: " + ", ".join(surf_meta.get("sources_used", [])) if surf_meta.get("sources_used") else "No usable input")
             + (f" | anchored to NDBC {wave_anchor.get('buoy')} (height x{wave_anchor.get('ratio')}"
                + (f", direction {wave_anchor.get('dir_offset_deg'):+.0f} deg" if wave_anchor.get("dir_offset") is not None else "")
                + ")" if wave_anchor.get("applied") else "")
             + (" | geometry from NOAA DEM ray-cast" if geom is not None else " | no exposure geometry")))
    df["thunder"] = weather_flags(df)
    df = df.join(detect_storms(df))
    df["clarity_index"] = clarity_index(df, sheltered=not spot.is_open_ocean)

    # ---- scores ----------------------------------------------------------------
    df = df.copy()   # de-fragment after the many feature joins
    bait = bait_scores(df, spot)
    df = df.join(bait)
    windows: dict[str, list[Window]] = {}
    for sp in species:
        pres = presence_scores(df, spot, sp, bait, bonus)
        df = df.join(pres)
        bait_av = bait_availability(bite_mod.species_meta(sp), bait)
        df = df.join(feeding_scores(df, spot, sp, bait_av))
        df = df.join(bite_mod.fishability(df, spot, sp))
        df[f"bite_{sp}"] = bite_mod.combine(df, sp)
        disp = df.loc[frame.display_start:frame.display_end, f"bite_{sp}"]
        windows[sp] = bite_mod.best_windows(disp, sp, spot.timezone)
    df["bite_best"] = df[[f"bite_{sp}" for sp in species]].max(axis=1)
    df["bite_best_species"] = df[[f"bite_{sp}" for sp in species]].idxmax(axis=1).str.replace("bite_", "", regex=False)

    quality = data_quality(sources, missing, frame.mode)
    return SpotForecast(spot=spot, frame=frame, hourly=df, sources=sources, alerts=alerts, bathy=bathy,
                        tide_events=tide_events, daily=daily, windows=windows, quality=quality, missing=missing,
                        surf=surf_meta, marine=marine_info)


def data_quality(sources: list[SourceInfo], missing: list[str], mode: str) -> dict[str, Any]:
    weights = {"weather": 15, "official hourly forecast": 10, "waves / swell / SST (model)": 10,
               "offshore waves (measured)": 10, "tide times & heights": 20, "tidal current": 10,
               "water temperature": 15, "bathymetry / structure": 5, "water-level anomaly (surge)": 5}
    if mode == "historical":
        weights.pop("official hourly forecast")
    got = 0.0
    for s in sources:
        key = next((k for k in weights if s.feature.startswith(k.split(" (")[0])), None)
        if not key or not s.ok:
            continue
        w = weights[key]
        if s.kind == DataKind.heuristic or (key == "tidal current" and "PROXY" in s.note):
            w *= 0.6
        if key == "water temperature" and s.kind == DataKind.model:
            w *= 0.35
        if s.age_hours is not None and s.age_hours > 6:
            w *= 0.6
        if s.stale:
            w *= 0.7
        got += w
        weights[key] = 0  # count each feature once
    total = 100 if mode == "forecast" else 90
    score = round(100 * got / total)
    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"
    return {"score": score, "grade": grade, "missing": missing}


# ------------------------------------------------------------------ public
async def collect(spots: list[Spot], frame: TimeFrame, progress: ProgressCb | None = None,
                  species: list[str] | None = None) -> dict[str, SpotForecast]:
    prog = Progress(progress, 1)
    async with HttpClient() as http:
        raw = await _fetch_all(http, spots, frame, prog, extra_steps=len(spots))
    out: dict[str, SpotForecast] = {}
    for i, sp in enumerate(spots):
        try:
            out[sp.id] = assemble(sp, frame, raw, i, species)
        except Exception as exc:  # noqa: BLE001
            log.exception("assemble failed for %s", sp.id)
            raise RuntimeError(f"Could not build forecast for {sp.name}: {exc}") from exc
        prog.step(f"Scoring {sp.short}")
    return out


def run_sync(coro):
    """Run a coroutine from sync code (Streamlit) even if a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


def feature_snapshot(fc: SpotForecast, when_utc: pd.Timestamp) -> dict[str, Any]:
    """Numeric features at a time (for the observation log / ML training)."""
    ts = pd.Timestamp(when_utc).tz_convert("UTC").floor("h")
    if ts not in fc.hourly.index:
        return {}
    row = fc.hourly.loc[ts]
    keep = {}
    for k, v in row.items():
        if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and np.isfinite(v):
            keep[k] = float(v)
        elif isinstance(v, str) and k in ("tide_stage_tag", "light_phase", "storm_phase", "current_phase"):
            keep[k] = v
    return keep
