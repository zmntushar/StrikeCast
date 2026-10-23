"""Historical comparison of two dates at the same spot.

Each date is rebuilt with the hindcast pipeline (``mode="historical"``):
measured buoy / gauge data where available, NOAA tide & current predictions,
archived model weather and waves, and computed astronomy. Every metric carries
its data kind so observed values are never confused with model hindcasts.

The explanation describes how conditions DIFFERED and how that may influence
bait and fish. It never claims a difference caused a catch outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from app.core.geo import angle_diff, compass
from app.models.common import species_list, species_meta
from app.providers import lunar
from app.services.forecast import ProgressCb, SpotForecast, collect, historical_frame
from app.schemas.env import Spot


@dataclass
class DayComparison:
    spot: Spot
    a: SpotForecast
    b: SpotForecast
    table: pd.DataFrame          # metric | unit | A | B | diff | kind_a | kind_b
    explanation: list[str]


def _circ_mean(deg: pd.Series, weights: pd.Series | None = None) -> float:
    d = np.radians(deg.dropna().astype(float))
    if d.empty:
        return np.nan
    w = np.ones(len(d)) if weights is None else weights.reindex(deg.dropna().index).fillna(0).to_numpy()
    return float((np.degrees(np.arctan2((np.sin(d) * w).sum(), (np.cos(d) * w).sum())) + 360) % 360)


def day_metrics(fc: SpotForecast) -> dict[str, tuple[Any, str]]:
    d = fc.display
    m: dict[str, tuple[Any, str]] = {}

    def put(key, val, kind):
        m[key] = (val, kind)

    kinds = {s.feature: s.kind.value for s in fc.sources}
    wx_kind = kinds.get("weather (wind, pressure, clouds, rain)", "missing")
    put("Wind speed (mean)", d["wind_mph"].mean(), wx_kind)
    put("Wind gusts (max)", d["gust_mph"].max(), wx_kind)
    put("Wind direction (dominant)", _circ_mean(d["wind_dir"], d["wind_mph"]), wx_kind)
    put("Onshore wind (mean)", d["onshore_mph"].mean(), "computed")
    put("Pressure (mean)", d["pressure_hpa"].mean(), wx_kind)
    put("Pressure change (24 h)", d["pressure_hpa"].iloc[-1] - d["pressure_hpa"].iloc[0] if d["pressure_hpa"].notna().sum() > 1 else np.nan, wx_kind)
    put("Air temperature (mean)", d["air_temp_f"].mean(), wx_kind)
    put("Cloud cover (mean)", d["cloud_pct"].mean(), wx_kind)
    put("Precipitation (total)", d["precip_in"].sum(min_count=1), wx_kind)
    nb = int(d["buoy_wave_ft"].notna().sum()) if "buoy_wave_ft" in d else 0
    bkind = f"measured {nb}/{len(d)} h" if nb else "missing"
    put("Offshore Hs, buoy (mean)", d["buoy_wave_ft"].mean() if nb else np.nan, bkind)
    put("Offshore Hs, buoy (max)", d["buoy_wave_ft"].max() if nb else np.nan, bkind)
    put("Buoy dominant period (mean)", d["buoy_period_s"].mean() if nb and "buoy_period_s" in d else np.nan, bkind)
    put("Buoy wave direction (mean)", _circ_mean(d["buoy_wave_dir"]) if nb and "buoy_wave_dir" in d else np.nan, bkind)
    ns = int(d["buoy_swell_ft"].notna().sum()) if "buoy_swell_ft" in d else 0
    skind = f"measured {ns}/{len(d)} h" if ns else "missing"
    put("Buoy swell height (mean)", d["buoy_swell_ft"].mean() if ns else np.nan, skind)
    put("Buoy swell period (mean)", d["buoy_swell_period_s"].mean() if ns else np.nan, skind)
    mk = "model" if d["wave_ft"].notna().any() else "missing"
    put("Offshore Hs, model (mean)", d["wave_ft"].mean(), mk)
    if fc.spot.is_open_ocean:
        used = ", ".join(sorted({str(x) for x in d.get("surf_source", pd.Series(dtype=object)).dropna().unique()})) or "missing"
        put("Swell height, used (mean)", d["bc_swell_ft"].mean() if "bc_swell_ft" in d else np.nan, used)
        put("Swell period, used (mean)", d["bc_swell_tp_s"].mean() if "bc_swell_tp_s" in d else np.nan, used)
        put("Swell direction, used (mean)", _circ_mean(d["bc_swell_dir"], d["bc_swell_ft"]) if "bc_swell_dir" in d else np.nan, used)
        put("Wind-wave height, used (mean)", d["bc_windwave_ft"].mean() if "bc_windwave_ft" in d else np.nan, used)
        put("Wind-wave period, used (mean)", d["bc_windwave_tp_s"].mean() if "bc_windwave_tp_s" in d else np.nan, used)
        put("Wave direction (dominant)", _circ_mean(d["offshore_dir"], d["offshore_hs_ft"]) if "offshore_dir" in d else np.nan, used)
    surf_kind = "model" if d["surf_ft"].notna().any() else "missing"
    lab = "Surf, breaking est." if fc.spot.is_open_ocean else "Wind chop est."
    put(f"{lab} (mean)", d["surf_ft"].mean(), surf_kind)
    put(f"{lab} (max)", d["surf_ft"].max(), surf_kind)
    conf = d["surf_confidence"].dropna().mode() if "surf_confidence" in d else pd.Series(dtype=object)
    put("Surf estimate confidence", str(conf.iloc[0]) if len(conf) else "—", surf_kind)
    if "nws_seas_hi_ft" in d and d["nws_seas_hi_ft"].notna().any():
        put("NWS zone seas/waves (forecast range)", f"{d['nws_seas_lo_ft'].min():.0f}–{d['nws_seas_hi_ft'].max():.0f} ft", "predicted (archived)")
    else:
        put("NWS zone seas/waves (forecast range)", "—", "missing")
    srf = [x for x in fc.marine.get("srf", []) if x["day"] == fc.frame.target_date]
    put("NWS surf zone forecast", f"{srf[0]['lo']:.0f}–{srf[0]['hi']:.0f} ft · rip {srf[0]['rip'] or '—'}" if srf else "—",
        "predicted (archived)" if srf else "missing (issued in beach season)")
    adv = sorted({a["name"] for a in fc.marine.get("advisories", [])})
    put("Marine advisories", "; ".join(adv) if adv else ("none" if fc.marine.get("advisories") is not None else "—"),
        "NWS (archived VTEC)" if fc.spot.marine_zone else "missing")
    wkind = str(d["water_temp_kind"].mode().iloc[0]) if "water_temp_kind" in d and d["water_temp_kind"].notna().any() else "missing"
    if wkind == "measured":
        wkind = f"measured {int(d['water_temp_f'].notna().sum())}/{len(d)} h"
    put("Water temperature (mean)", d["water_temp_f"].mean(), wkind)
    put("Water temp change (3 days)", d["water_temp_trend_3d"].mean(), wkind)
    put("Tidal range (max)", d["tide_range_ft"].max(), "predicted")
    if fc.daily and fc.daily[0].get("sunrise") is not None:
        sr = pd.Timestamp(fc.daily[0]["sunrise"]).tz_convert("UTC").floor("h")
        put("Tide stage at sunrise", str(fc.hourly.loc[sr, "tide_stage"]) if sr in fc.hourly.index and fc.hourly.loc[sr, "tide_stage"] else "—",
            "predicted")
    put("Peak current", d["current_kt"].abs().max() if d["current_kt"].notna().any() else np.nan,
        "predicted" if d["current_kt"].notna().any() else "heuristic")
    put("Water-level anomaly (max)", d["water_level_anom_ft"].max() if d["water_level_anom_ft"].notna().any() else np.nan,
        "measured" if d["water_level_anom_ft"].notna().any() else "missing")
    noon = fc.frame.display_start + pd.Timedelta(hours=12)
    mp = lunar.moon_phase_info(noon.to_pydatetime())
    put("Moon phase", mp["phase_name"], "computed")
    put("Moon illumination", mp["illumination_pct"], "computed")
    put("Spring-tide index", mp["spring_index"], "computed")
    put("Storm hours", int(d["storm_active"].sum()), "heuristic")
    put("Storm phase", ", ".join(sorted(set(d["storm_phase"].astype(str)) - {"none"})) or "none", "heuristic")
    put("Baitfish index (mean)", d["bait_index"].mean(), "heuristic")
    for sp in species_list():
        if f"bite_{sp}" in d:
            put(f"{species_meta(sp)['short']} Bite Index (peak)", d[f"bite_{sp}"].max(), "heuristic")
            put(f"{species_meta(sp)['short']} season factor", d[f"pf_{sp}_season"].mean(), "heuristic")
    return m


UNITS = {"Wind speed (mean)": "mph", "Wind gusts (max)": "mph", "Wind direction (dominant)": "°",
         "Onshore wind (mean)": "mph", "Pressure (mean)": "hPa", "Pressure change (24 h)": "hPa",
         "Air temperature (mean)": "°F", "Cloud cover (mean)": "%", "Precipitation (total)": "in",
         "Offshore Hs, buoy (mean)": "ft", "Offshore Hs, buoy (max)": "ft", "Buoy dominant period (mean)": "s",
         "Buoy wave direction (mean)": "°", "Buoy swell height (mean)": "ft", "Buoy swell period (mean)": "s",
         "Offshore Hs, model (mean)": "ft", "Swell height, used (mean)": "ft", "Swell period, used (mean)": "s",
         "Swell direction, used (mean)": "°", "Wind-wave height, used (mean)": "ft", "Wind-wave period, used (mean)": "s",
         "Wave direction (dominant)": "°", "Surf, breaking est. (mean)": "ft", "Surf, breaking est. (max)": "ft",
         "Wind chop est. (mean)": "ft", "Wind chop est. (max)": "ft", "Water temperature (mean)": "°F",
         "Water temp change (3 days)": "°F", "Tidal range (max)": "ft", "Peak current": "kt",
         "Water-level anomaly (max)": "ft", "Moon illumination": "%"}


def build_table(ma: dict, mb: dict) -> pd.DataFrame:
    rows = []
    for k in ma:
        va, ka = ma[k]
        vb, kb = mb.get(k, (np.nan, "missing"))
        numeric = isinstance(va, (int, float, np.floating)) and isinstance(vb, (int, float, np.floating))
        diff = (angle_diff(vb, va) if "direction" in k.lower() else vb - va) if numeric else None
        rows.append({"metric": k, "unit": UNITS.get(k, ""), "A": va, "B": vb, "diff": diff, "kind_a": ka, "kind_b": kb})
    return pd.DataFrame(rows)


def explain(table: pd.DataFrame, fa: SpotForecast, fb: SpotForecast) -> list[str]:
    t = table.set_index("metric")
    la = fa.frame.target_date.strftime("%b %d")
    lb = fb.frame.target_date.strftime("%b %d")
    out: list[str] = []

    def v(metric, side):
        try:
            x = t.loc[metric, side]
            return None if x is None or (isinstance(x, float) and np.isnan(x)) else x
        except KeyError:
            return None

    wa, wb = v("Wind speed (mean)", "A"), v("Wind speed (mean)", "B")
    if wa is not None and wb is not None and abs(wb - wa) >= 4:
        windier, calmer = (lb, la) if wb > wa else (la, lb)
        ob_w = v("Onshore wind (mean)", "B" if wb > wa else "A") or 0
        effect = ("Onshore wind like that can stack bait against the beach and color the water."
                  if ob_w > 3 else "Offshore wind knocks down local chop, grooms any incoming swell and can push bait off the beach."
                  if ob_w < -3 else "Mostly side-shore, so it moved water along the beach more than onto it.")
        out.append(f"Wind: {windier} was noticeably windier ({max(wa, wb):.0f} vs {min(wa, wb):.0f} mph average). "
                   f"Direction {compass(v('Wind direction (dominant)', 'A'))} on {la} vs {compass(v('Wind direction (dominant)', 'B'))} on {lb}. "
                   + effect + " Strong wind also limits shore fishing.")
    oa, ob = v("Onshore wind (mean)", "A"), v("Onshore wind (mean)", "B")
    if oa is not None and ob is not None and np.sign(oa) != np.sign(ob) and max(abs(oa), abs(ob)) > 4:
        out.append(f"Wind orientation flipped: {'onshore' if oa > 0 else 'offshore'} on {la} vs {'onshore' if ob > 0 else 'offshore'} on {lb} for this shoreline.")
    pa, pb = v("Pressure change (24 h)", "A"), v("Pressure change (24 h)", "B")
    if pa is not None and pb is not None and abs(pb - pa) >= 4:
        out.append(f"Pressure trend: {pa:+.0f} hPa over {la} vs {pb:+.0f} hPa over {lb}. Falling pressure ahead of fronts is often linked to active feeding; a sharp post-front rise is often linked to a slower bite.")
    lab = "Surf, breaking est. (mean)" if fa.spot.is_open_ocean else "Wind chop est. (mean)"
    sa, sb = v(lab, "A"), v(lab, "B")
    if sa is not None and sb is not None and abs(sb - sa) >= 0.8:
        word = "Surf" if fa.spot.is_open_ocean else "Wind chop (sheltered water)"
        out.append(f"{word}: about {sa:.1f} ft on {la} vs {sb:.1f} ft on {lb} (physical estimate from archived waves/wind). "
                   "Moderate white water favors bass and blues; rough or dirty water hurts fluke, porgies and tog, and jetty safety.")
    pa_, pb_ = v("Swell period, used (mean)", "A"), v("Swell period, used (mean)", "B")
    da_, db_ = v("Swell direction, used (mean)", "A"), v("Swell direction, used (mean)", "B")
    if pa_ is not None and pb_ is not None and (abs(pb_ - pa_) >= 2 or (da_ is not None and db_ is not None and abs(angle_diff(db_, da_)) >= 30)):
        out.append(f"Swell: {pa_:.0f} s from {compass(da_)} on {la} vs {pb_:.0f} s from {compass(db_)} on {lb}. "
                   "Longer-period swell from a direction facing the beach breaks harder (more power, stronger rips) than short wind chop running along it.")
    for side, lab_d in (("A", la), ("B", lb)):
        adv = v("Marine advisories", side)
        if adv and adv not in ("none", "—"):
            out.append(f"NWS marine advisories in effect on {lab_d}: {adv}.")
    ya, yb = v("Offshore Hs, buoy (mean)", "A"), v("Offshore Hs, buoy (mean)", "B")
    if ya is not None and yb is not None and abs(yb - ya) >= 2:
        out.append(f"Offshore seas (measured at the buoy): {ya:.1f} ft on {la} vs {yb:.1f} ft on {lb}. "
                   "Big offshore seas usually mean a churned-up, dirty nearshore and dangerous jetties.")
    ta, tb = v("Water temperature (mean)", "A"), v("Water temperature (mean)", "B")
    if ta is not None and tb is not None and abs(tb - ta) >= 1:
        out.append(f"Water temperature: {ta:.1f}°F on {la} vs {tb:.1f}°F on {lb} ({tb - ta:+.1f}°F). "
                   "In the fall, dropping water temperatures push bait out of the bays and trigger migration movement.")
    ra, rb = v("Tidal range (max)", "A"), v("Tidal range (max)", "B")
    if ra is not None and rb is not None and abs(rb - ra) >= 0.4:
        out.append(f"Tides: range {ra:.1f} ft vs {rb:.1f} ft — the {'bigger' if rb > ra else 'smaller'} tide on {lb} means "
                   f"{'stronger' if rb > ra else 'weaker'} currents through inlets and rips.")
    out.append(f"Moon: {v('Moon phase', 'A')} ({v('Moon illumination', 'A'):.0f}% lit) on {la} vs {v('Moon phase', 'B')} ({v('Moon illumination', 'B'):.0f}% lit) on {lb}. Moon phase is shown separately from tidal range because the phase alone does not set local tide strength.")
    ha, hb = v("Storm hours", "A"), v("Storm hours", "B")
    if (ha or 0) or (hb or 0):
        out.append(f"Storms: {ha} storm hours on {la} ({v('Storm phase', 'A')}) vs {hb} on {lb} ({v('Storm phase', 'B')}). Post-storm periods can produce strong bass/blues bites once water clears, but effects vary.")
    ba, bb = v("Bass Bite Index (peak)", "A"), v("Bass Bite Index (peak)", "B")
    if ba is not None and bb is not None:
        out.append(f"Overall, the striped bass Bite Index peaked at {ba:.0f} on {la} vs {bb:.0f} on {lb}. "
                   "These are environmental suitability scores — they describe conditions, not what was actually caught.")
    for lab, fc in ((la, fa), (lb, fb)):
        n = int(fc.display["buoy_wave_ft"].notna().sum()) if "buoy_wave_ft" in fc.display else 0
        if n < len(fc.display) * 0.5:
            out.append(f"Data note: the offshore buoy reported waves for only {n} of {len(fc.display)} hours on {lab} "
                       "(station outage) — treat that day's buoy average with caution; model hindcasts fill the charts.")
    return out


async def compare_dates(spot: Spot, date_a: date, date_b: date, progress: ProgressCb | None = None) -> DayComparison:
    def sub(offset: float):
        return (lambda p, label: progress(offset + p / 2, label)) if progress else None

    fa = (await collect([spot], historical_frame(date_a, spot.timezone), sub(0.0)))[spot.id]
    fb = (await collect([spot], historical_frame(date_b, spot.timezone), sub(0.5)))[spot.id]
    table = build_table(day_metrics(fa), day_metrics(fb))
    return DayComparison(spot=spot, a=fa, b=fb, table=table, explanation=explain(table, fa, fb))
