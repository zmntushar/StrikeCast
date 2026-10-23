"""Marine providers.

* NOAA NDBC buoys (public domain, measured): https://www.ndbc.noaa.gov/faq/measdes.shtml
  realtime2/<id>.txt holds the last 45 days; older data come from the stdmet archives.
* Open-Meteo Marine API (free, model): https://open-meteo.com/en/docs/marine-weather-api
  waves, swell, sea-surface temperature and surface currents (MeteoFrance/ECMWF/
  NOAA WW3 blend). ``cell_selection=sea`` keeps grid cells over water.

Buoys are OFFSHORE measurements; nearshore surf is computed separately in
app/features/surf.py from these offshore conditions and each spot's geometry.

All heights leave this module in FEET, periods in seconds, directions in degrees
the waves come FROM (true). Every frame passes ``qc_waves`` first.
"""
from __future__ import annotations

import io
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.geo import c_to_f, m_to_ft, ms_to_mph
from app.core.http import FetchError, HttpClient
from app.schemas.env import DataKind, SourceInfo

OM_MARINE = "https://marine-api.open-meteo.com/v1/marine"
MARINE_VARS = ["wave_height", "wave_period", "wave_peak_period", "wave_direction",
               "swell_wave_height", "swell_wave_period", "swell_wave_peak_period", "swell_wave_direction",
               "secondary_swell_wave_height", "secondary_swell_wave_period", "secondary_swell_wave_direction",
               "wind_wave_height", "wind_wave_period", "wind_wave_peak_period", "wind_wave_direction",
               "sea_surface_temperature", "ocean_current_velocity", "ocean_current_direction"]
MARINE_RENAME = {"wave_height": "wave_ft", "wave_period": "wave_period_s", "wave_peak_period": "wave_peak_period_s",
                 "wave_direction": "wave_dir",
                 "swell_wave_height": "swell_ft", "swell_wave_period": "swell_period_s",
                 "swell_wave_peak_period": "swell_peak_period_s", "swell_wave_direction": "swell_dir",
                 "secondary_swell_wave_height": "swell2_ft", "secondary_swell_wave_period": "swell2_period_s",
                 "secondary_swell_wave_direction": "swell2_dir",
                 "wind_wave_height": "windwave_ft", "wind_wave_period": "windwave_period_s",
                 "wind_wave_peak_period": "windwave_peak_period_s", "wind_wave_direction": "windwave_dir",
                 "sea_surface_temperature": "sst_model_f",
                 "ocean_current_velocity": "model_current_kt", "ocean_current_direction": "model_current_dir"}
HEIGHT_COLS = ("wave_ft", "swell_ft", "swell2_ft", "windwave_ft")
PERIOD_COLS = ("wave_period_s", "wave_peak_period_s", "swell_period_s", "swell_peak_period_s",
               "windwave_period_s", "windwave_peak_period_s", "avg_period_s", "swell2_period_s")
DIR_COLS = ("wave_dir", "swell_dir", "swell2_dir", "windwave_dir")

NDBC_RT = "https://www.ndbc.noaa.gov/data/realtime2/{sid}.txt"
NDBC_HIST_YEAR = "https://www.ndbc.noaa.gov/view_text_file.php"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


async def openmeteo_marine(
    http: HttpClient,
    points: list[tuple[float, float]],
    *,
    past_days: int = 2,
    forecast_days: int = 8,
    start: date | None = None,
    end: date | None = None,
) -> list[tuple[pd.DataFrame, SourceInfo]]:
    s = get_settings()
    params: dict[str, Any] = {
        "latitude": ",".join(f"{p[0]:.4f}" for p in points),
        "longitude": ",".join(f"{p[1]:.4f}" for p in points),
        "hourly": ",".join(MARINE_VARS), "timezone": "GMT", "cell_selection": "sea",
    }
    if start and end:
        params.update(start_date=start.isoformat(), end_date=end.isoformat())
        ttl = s.ttl_openmeteo_history
    else:
        params.update(past_days=past_days, forecast_days=forecast_days)
        ttl = s.ttl_openmeteo_marine
    res = await http.request(OM_MARINE, provider="Open-Meteo Marine", params=params, ttl=ttl)
    blocks = res.data if isinstance(res.data, list) else [res.data]
    out = []
    for block in blocks:
        hourly = block.get("hourly") or {}
        if not hourly.get("time"):
            out.append((pd.DataFrame(), SourceInfo(feature="waves / swell / SST (model)", provider="Open-Meteo Marine",
                                                   kind=DataKind.model, ok=False, note="no data")))
            continue
        idx = pd.to_datetime(hourly["time"], utc=True)
        df = pd.DataFrame({MARINE_RENAME[k]: hourly[k] for k in hourly if k in MARINE_RENAME}, index=idx)
        df = df.apply(pd.to_numeric, errors="coerce")
        for c in HEIGHT_COLS:
            if c in df:
                df[c] = m_to_ft(df[c])
        if "sst_model_f" in df:
            df["sst_model_f"] = c_to_f(df["sst_model_f"])
        if "model_current_kt" in df:  # km/h -> kt
            df["model_current_kt"] = df["model_current_kt"] * 0.539957
        df, flags = qc_waves(df)
        out.append((df, SourceInfo(
            feature="waves / swell / SST (model)", provider="Open-Meteo Marine",
            product="wave + SST model (offshore grid cell)", kind=DataKind.model,
            fetched_at=res.fetched_at, ok=not df.dropna(how="all").empty, stale=res.stale, url=res.url[:300],
            note=f"grid cell {block.get('latitude')}, {block.get('longitude')}"
                 + (f" · QC: {'; '.join(flags)}" if flags else ""))))
    return out


# ------------------------------------------------------------------ QC
MAX_HS_FT = 60.0          # well above any recorded Long Island sea state
PERIOD_RANGE_S = (1.0, 30.0)


def qc_waves(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove physically impossible values and isolated spikes; report what was removed.

    * heights must be 0..60 ft, periods 1..30 s, directions 0..360 deg
    * a spike is a jump of > max(6 ft, 150 %) above both neighbours that agree with each other
    * a stuck sensor repeats the identical non-zero height for 36+ consecutive readings
    Nothing is ever filled in: rejected values become NaN.
    """
    if df.empty:
        return df, []
    df = df.copy()
    flags: list[str] = []
    for c in HEIGHT_COLS:
        if c not in df:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        bad = (s < 0) | (s > MAX_HS_FT)
        prev, nxt = s.shift(1), s.shift(-1)
        base = (prev + nxt) / 2
        spike = (((s - base).abs() > np.maximum(6.0, 1.5 * base.abs())) & prev.notna() & nxt.notna()
                 & ((prev - nxt).abs() < np.maximum(2.0, 0.5 * base.abs())))
        run = s.groupby((s != s.shift()).cumsum()).transform("size")
        stuck = (run >= 36) & (s > 0) & s.notna()
        rej = bad | spike | stuck
        if rej.any():
            flags.append(f"{c}: {int(bad.sum())} out-of-range, {int(spike.sum())} spikes, {int(stuck.sum())} stuck")
            df[c] = s.mask(rej)
    for c in PERIOD_COLS:
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce")
            bad = (s < PERIOD_RANGE_S[0]) | (s > PERIOD_RANGE_S[1])
            if bad.any():
                flags.append(f"{c}: {int(bad.sum())} invalid periods")
                df[c] = s.mask(bad)
    for c in DIR_COLS:
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce")
            bad = (s < 0) | (s > 360)
            if bad.any():
                flags.append(f"{c}: {int(bad.sum())} invalid directions")
                df[c] = s.mask(bad)
    return df, flags


# --------------------------------------------------------------------- NDBC
def parse_ndbc_stdmet(text: str) -> pd.DataFrame:
    """Parse an NDBC standard meteorological text file into US units (UTC index)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("#"):
        return pd.DataFrame()
    header = lines[0].lstrip("#").split()
    body = "\n".join(ln for ln in lines if not ln.startswith("#"))
    if not body:
        return pd.DataFrame()
    raw = pd.read_csv(io.StringIO(body), sep=r"\s+", header=None, names=header, na_values=["MM", "99.0", "999", "9999.0", "999.0", "99.00"], dtype=str)
    num = raw.apply(pd.to_numeric, errors="coerce")
    yy = num[header[0]].astype("Int64")
    yy = yy.where(yy > 100, yy + 1900)
    ts = pd.to_datetime(dict(year=yy, month=num["MM"], day=num["DD"], hour=num["hh"],
                             minute=num["mm"] if "mm" in num else 0), utc=True, errors="coerce")
    out = pd.DataFrame(index=ts)
    def col(name):
        return num[name].to_numpy() if name in num else np.full(len(num), np.nan)
    out["wind_dir"] = col("WDIR")
    out["wind_mph"] = ms_to_mph(col("WSPD"))
    out["gust_mph"] = ms_to_mph(col("GST"))
    out["wave_ft"] = m_to_ft(col("WVHT"))
    out["wave_period_s"] = col("DPD")
    out["avg_period_s"] = col("APD")
    out["wave_dir"] = col("MWD")
    out["pressure_hpa"] = col("PRES")
    out["air_temp_f"] = c_to_f(col("ATMP"))
    out["water_temp_f"] = c_to_f(col("WTMP"))
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out, _ = qc_waves(out)
    return out


COMPASS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _dir_val(v) -> float:
    v = str(v).strip().upper()
    if v in COMPASS16:
        return COMPASS16.index(v) * 22.5
    try:
        return float(v)
    except ValueError:
        return np.nan


def parse_ndbc_spec(text: str) -> pd.DataFrame:
    """Parse an NDBC spectral wave summary (.spec): the measured swell / wind-wave split.

    Output (US units): wave_ft, swell_ft, swell_period_s, swell_dir, windwave_ft,
    windwave_period_s, windwave_dir, avg_period_s, wave_dir, steepness.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("#"):
        return pd.DataFrame()
    header = lines[0].lstrip("#").split()
    rows = [ln.split() for ln in lines if not ln.startswith("#")]
    rows = [r for r in rows if len(r) == len(header)]
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows, columns=header).replace("MM", np.nan)
    num = raw.drop(columns=[c for c in ("SwD", "WWD", "STEEPNESS") if c in raw]).apply(pd.to_numeric, errors="coerce")
    yy = num[header[0]].astype("Int64")
    ts = pd.to_datetime(dict(year=yy, month=num["MM"], day=num["DD"], hour=num["hh"], minute=num["mm"]),
                        utc=True, errors="coerce")
    out = pd.DataFrame(index=ts)
    out["wave_ft"] = m_to_ft(num["WVHT"].to_numpy())
    out["swell_ft"] = m_to_ft(num["SwH"].to_numpy())
    out["swell_period_s"] = num["SwP"].to_numpy()
    out["swell_dir"] = raw["SwD"].map(_dir_val).to_numpy() if "SwD" in raw else np.nan
    out["windwave_ft"] = m_to_ft(num["WWH"].to_numpy())
    out["windwave_period_s"] = num["WWP"].to_numpy()
    out["windwave_dir"] = raw["WWD"].map(_dir_val).to_numpy() if "WWD" in raw else np.nan
    out["avg_period_s"] = num["APD"].to_numpy()
    out["wave_dir"] = num["MWD"].to_numpy()
    out["steepness"] = raw["STEEPNESS"].to_numpy() if "STEEPNESS" in raw else None
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out, _ = qc_waves(out)
    return out


NDBC_SPEC = "https://www.ndbc.noaa.gov/data/realtime2/{sid}.spec"


async def ndbc_spectral(http: HttpClient, station: str, *, start: datetime | None = None,
                        end: datetime | None = None) -> tuple[pd.DataFrame, str]:
    """Measured swell / wind-wave partitions (the realtime file covers the last 45 days only)."""
    now = datetime.now(timezone.utc)
    if start is not None and start < now - timedelta(days=44):
        raise FetchError(f"NDBC {station}: spectral summaries are only online for the last 45 days")
    res = await http.request(NDBC_SPEC.format(sid=station), provider="NOAA NDBC",
                             ttl=get_settings().ttl_ndbc_realtime, as_json=False)
    df = parse_ndbc_spec(res.data)
    if start is not None and not df.empty:
        df = df[(df.index >= start) & (df.index <= (end or now))]
    if df.empty:
        raise FetchError(f"NDBC {station}: no spectral data for window")
    return df, res.url


async def ndbc_observations(
    http: HttpClient, station: str, *, start: datetime | None = None, end: datetime | None = None
) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, url). Uses realtime2 when the window is within 44 days."""
    s = get_settings()
    now = datetime.now(timezone.utc)
    if start is None or start > now - timedelta(days=44):
        res = await http.request(NDBC_RT.format(sid=station), provider="NOAA NDBC",
                                 ttl=s.ttl_ndbc_realtime, as_json=False)
        df = parse_ndbc_stdmet(res.data)
        url = res.url
    else:
        if start.year < now.year:
            params = {"filename": f"{station}h{start.year}.txt.gz", "dir": "data/historical/stdmet/"}
        else:
            m = start.month
            params = {"filename": f"{station}{m}{start.year}.txt.gz", "dir": f"data/stdmet/{MONTHS[m-1]}/"}
        res = await http.request(NDBC_HIST_YEAR, provider="NOAA NDBC", params=params,
                                 ttl=s.ttl_openmeteo_history, as_json=False)
        df = parse_ndbc_stdmet(res.data)
        url = res.url
    if start is not None and not df.empty:
        df = df[(df.index >= start) & (df.index <= (end or now))]
    if df.empty:
        raise FetchError(f"NDBC {station}: no data for window")
    return df, url
