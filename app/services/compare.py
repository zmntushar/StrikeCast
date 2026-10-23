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
    put("Waves at spot, model (mean)", d["wave_ft"].mean(), "model")
    put("Wave period (mean)", d["wave_period_s"].mean(), "model")
    nb = int(d["buoy_wave_ft"].notna().sum()) if "buoy_wave_ft" in d else 0
    put("Offshore buoy waves (mean)", d["buoy_wave_ft"].mean() if nb else np.nan,
        f"measured {nb}/{len(d)} h" if nb else "missing")
    put("Nearshore surf, est. (mean)", d["nearshore_wave_ft"].mean(), "heuristic")
    wkind = str(d["water_temp_kind"].mode().iloc[0]) if "water_temp_kind" in d and d["water_temp_kind"].notna().any() else "missing"
    if wkind == "measured":
        wkind = f"measured {int(d['water_temp_f'].notna().sum())}/{len(d)} h"
    put("Water temperature (mean)", d["water_temp_f"].mean(), wkind)
    put("Water temp change (3 days)", d["water_temp_trend_3d"].mean(), wkind)
    put("Tidal range (max)", d["tide_range_ft"].max(), "predicted")
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
         "Waves at spot, model (mean)": "ft", "Wave period (mean)": "s", "Offshore buoy waves (mean)": "ft",
         "Nearshore surf, est. (mean)": "ft", "Water temperature (mean)": "°F",
         "Water temp change (3 days)": "°F", "Tidal range (max)": "ft", "Peak current": "kt",
         "Water-level anomaly (max)": "ft", "Moon illumination": "%"}


def build_table(ma: dict, mb: dict) -> pd.DataFrame:
    rows = []
    for k in ma:
        va, ka = ma[k]
        vb, kb = mb.get(k, (np.nan, "missing"))
        numeric = isinstance(va, (int, float, np.floating)) and isinstance(vb, (int, float, np.floating))
        diff = (angle_diff(vb, va) if "direction" in k else vb - va) if numeric else None
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
                  if ob_w > 3 else "Offshore wind flattens the surf and can push bait off the beach."
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
    sa, sb = v("Nearshore surf, est. (mean)", "A"), v("Nearshore surf, est. (mean)", "B")
    if sa is not None and sb is not None and abs(sb - sa) >= 1:
        out.append(f"Surf: about {sa:.1f} ft on {la} vs {sb:.1f} ft on {lb} (estimate). Moderate white water favors bass and blues; rough or dirty water hurts fluke, porgies and tog, and jetty safety.")
    ya, yb = v("Offshore buoy waves (mean)", "A"), v("Offshore buoy waves (mean)", "B")
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
