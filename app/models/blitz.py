"""Blitz Potential (0-100): when bait, migrating fish and conditions line up for a surface blitz.

A shore blitz is striped bass or bluefish driving dense bait to the surface within casting range,
usually under diving birds. It needs ALL of these at once:

* migration   — fish moving through (species migration calendar for the spot)
* temperature — water in the surface-feeding band of the run (fall bass ~50-58 F)
* bait        — dense bait near shore. In the fall that is above all the exodus of juvenile bunker,
                spearing and anchovies from the bays, which happens as the bays cool below the ocean:
                measured from a bay buoy vs an ocean buoy (``bait_flush``), plus the bait calendar and the
                spot type (inlets, jetties and ocean beaches concentrate bait; bays far less)
* access      — bait held within casting range and feeding visible: light-to-moderate wind, no strong
                offshore wind (it pushes bait out of range), manageable surf, clean-enough water, no storm
* tide        — moving water; at inlets and river mouths the ebb flushes bait out to waiting fish
* light       — first light to ~1.5 h after sunrise, and the last ~1.5 h of daylight into dusk
* front       — steady or slowly falling barometer; not the clear, rising-pressure day after a cold front

The score is a weighted GEOMETRIC mean, so a single weak factor pulls it down hard. Levels are strict on
purpose (the user prefers missed calls over false alarms):

* ``alert`` — score above the alert threshold AND every factor above its own floor, a measured water
  temperature, tide predictions and a forecast at most ``max_alert_lead_hours`` ahead.
* ``watch`` — the setup is there but one ingredient is marginal or unverified.

Birds — the best real-time sign — have no free data feed. A "birds diving" / "surface blitz" report in the
Fishing Log near the spot in the last day and a half confirms the bait and fish and lifts a watch to an alert.

This is a setup indicator, not a guarantee: fish can blitz out of casting range, or not at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import species_config
from app.core.geo import compass
from app.features.seasonal import trapezoid
from app.models.common import col, species_meta
from app.schemas.env import Spot

COMPONENTS = ("migration", "temperature", "bait", "access", "tide", "light", "front")
LABELS = {"migration": "Migrating fish", "temperature": "Water temperature", "bait": "Bait near shore",
          "access": "Wind & surf (bait in range)", "tide": "Moving water", "light": "Light (dawn / dusk)",
          "front": "Barometer / front"}
LEVEL_RANK = {"": 0, "watch": 1, "alert": 2}
LEVEL_TEXT = {"alert": "Blitz alert", "watch": "Blitz watch"}
FALL = (245, 355)          # day-of-year span of the fall bait exodus


def blitz_config() -> dict[str, Any]:
    return species_config().get("blitz", {}) or {}


def blitz_species() -> list[str]:
    sp = species_config()["species"]
    return [s for s in blitz_config().get("species", []) if s in sp and sp[s].get("blitz")]


def _ramp(x, a: float, b: float) -> np.ndarray:
    return np.clip((np.asarray(x, dtype=float) - a) / (b - a), 0, 1)


# ------------------------------------------------------------------ bait flush (regional)
def bait_flush(doy: np.ndarray, bay_t: np.ndarray, ocean_t: np.ndarray, bay_trend: np.ndarray,
               own_t: np.ndarray | None = None, own_trend: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fall exodus of juvenile bait from the bays, 0-1, and whether it came from measured bay/ocean gauges.

    Juvenile menhaden ("peanut bunker"), silversides and bay anchovies leave the bays through the inlets as
    the bays cool — strongest when the bay is clearly colder than the ocean or dropping fast, while the bay
    is still between ~47 and ~60 F (below that most bait has already left). Without a bay/ocean pair (Sound
    spots) the spot's own measured cooling is used as a weaker, labeled estimate. Outside the fall: NaN.
    """
    doy = np.asarray(doy, dtype=float)
    fall = (doy >= FALL[0]) & (doy <= FALL[1])
    bay_t, ocean_t, bay_trend = (np.asarray(a, dtype=float) for a in (bay_t, ocean_t, bay_trend))
    band = trapezoid(bay_t, 42, 47, 60, 66)
    trigger = np.fmax(_ramp(-bay_trend, 0.5, 4.0), _ramp(ocean_t - bay_t, 1.0, 7.0))
    gauged = np.where(np.isnan(band) | np.isnan(ocean_t), np.nan, band * (0.3 + 0.7 * np.nan_to_num(trigger)))
    measured = ~np.isnan(gauged)
    if own_t is not None:
        own_band = trapezoid(np.asarray(own_t, dtype=float), 44, 48, 62, 68)
        own_trig = _ramp(-np.asarray(own_trend if own_trend is not None else np.nan, dtype=float), 0.5, 3.0)
        est = own_band * (0.35 + 0.5 * np.nan_to_num(own_trig))
        gauged = np.where(measured, gauged, est)
    return np.where(fall, gauged, np.nan), measured & fall


# ------------------------------------------------------------------------- components
def light_component(sun_alt: np.ndarray, cloud_pct: np.ndarray) -> np.ndarray:
    """First light (sun -9..+12 deg, rising) = 1; last light (+10..-6, setting) = 0.95; midday low, night lowest."""
    e = np.asarray(sun_alt, dtype=float)
    rising = np.gradient(e) > 0 if len(e) > 1 else np.ones(len(e), dtype=bool)
    day = 0.45 + 0.2 * np.nan_to_num(np.asarray(cloud_pct, dtype=float), nan=30.0) / 100
    night = 0.25
    up = np.where(e < -12, night, np.where(e < -9, night + (1 - night) * (e + 12) / 3,
                  np.where(e <= 12, 1.0, np.where(e < 20, 1.0 - (1.0 - day) * (e - 12) / 8, day))))
    down = np.where(e < -10, night, np.where(e < -6, night + (0.95 - night) * (e + 10) / 4,
                    np.where(e <= 10, 0.95, np.where(e < 18, 0.95 - (0.95 - day) * (e - 10) / 8, day))))
    return np.where(np.isnan(e), np.nan, np.where(rising, up, down))


def tide_component(df: pd.DataFrame, spot: Spot) -> np.ndarray:
    cn = col(df, "current_norm")
    move = np.where(np.isnan(cn), 0.7, 0.45 + 0.55 * _ramp(cn, 0.1, 0.45))
    phase = df.get("current_phase", pd.Series(None, index=df.index)).astype(object).to_numpy()
    tdir = df.get("tide_dir", pd.Series(None, index=df.index)).astype(object).to_numpy()
    ebb = np.array([(p == "ebb") or (p not in ("ebb", "flood") and t == "outgoing") for p, t in zip(phase, tdir)])
    flushing = spot.environment in ("inlet", "jetty", "river_mouth") or bool({"inlet", "river_mouth"} & set(spot.structure))
    direction = np.where(ebb, 1.0, 0.82) if flushing else np.full(len(df), 0.95)
    return move * direction


def access_component(df: pd.DataFrame, spot: Spot) -> np.ndarray:
    """Is the bait held within casting range, and can fish and birds work it on the surface?"""
    wind = col(df, "wind_mph")
    gust = col(df, "gust_mph")
    on = col(df, "onshore_mph")
    eff = np.fmax(wind, 0.75 * gust)          # gusty wind chops the surface and scatters bait like a stronger wind
    speed = np.where(np.isnan(eff), 1.0, np.interp(np.nan_to_num(eff), [0, 12, 20, 26, 60], [1, 1, 0.35, 0.1, 0.05]))
    offshore = np.nan_to_num(-on)
    off_pen = (1 - 0.55 * _ramp(offshore, 6, 16)) * (1 - 0.15 * _ramp(col(df, "offshore_hours_12h", 0), 6, 12))
    on_pen = 1 - 0.4 * _ramp(np.nan_to_num(on), 15, 24)
    surf = col(df, "nearshore_wave_ft")
    if spot.is_open_ocean:
        s = np.interp(np.nan_to_num(surf), [0, 3, 5, 8, 30], [1, 1, 0.55, 0.15, 0.1])
    else:
        s = np.interp(np.nan_to_num(surf), [0, 2, 4, 30], [1, 1, 0.5, 0.4])
    s = np.where(np.isnan(surf), 1.0, s)
    clarity = 1 - 0.45 * _ramp(np.nan_to_num(col(df, "clarity_index")), 0.5, 0.85)
    phase = df.get("storm_phase", pd.Series("none", index=df.index)).astype(str).to_numpy()
    since = np.nan_to_num(col(df, "hours_since_storm"), nan=99)
    # Ahead of a storm the surface goes to pieces within hours; after one the water stays churned for a day or so.
    storm = np.where(phase == "during", 0.1, np.where(phase == "pre", 0.8,
                     np.interp(since, [0, 12, 24, 36, 48], [0.4, 0.4, 0.7, 0.85, 1.0])))
    storm = np.where((phase == "post") | (phase == "during") | (phase == "pre"), storm, 1.0)
    thunder = df.get("thunder", pd.Series(False, index=df.index)).fillna(False).to_numpy(dtype=bool)
    rain = col(df, "precip_in", 0)
    wx = np.where(thunder, 0.15, 1.0) * np.where(np.nan_to_num(rain) > 0.15, 0.7, 1.0)
    vis = col(df, "visibility_mi")
    fog = np.where(np.nan_to_num(vis, nan=10) < 0.5, 0.7, 1.0)
    return np.clip(speed * off_pen * on_pen * s * clarity * storm * wx * fog, 0, 1)


def front_component(df: pd.DataFrame) -> np.ndarray:
    """Steady / slowly falling pressure = good; a clear, rising post-frontal high = bad; a crashing low = rough."""
    p24 = col(df, "pressure_tend_24h")
    cloud = np.nan_to_num(col(df, "cloud_pct"), nan=40) / 100
    post = _ramp(p24, 2.5, 8) * (0.6 + 0.4 * (1 - cloud))
    v = 0.85 + 0.15 * _ramp(-p24, 1, 5) - 0.45 * post - 0.3 * _ramp(-p24, 6, 12)
    return np.where(np.isnan(p24), 0.8, np.clip(v, 0.2, 1))


def species_components(df: pd.DataFrame, spot: Spot, species: str, bait: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg = blitz_config()
    bz = species_meta(species).get("blitz") or {}
    doy = df.index.tz_convert(spot.timezone).day_of_year.to_numpy()
    season = col(df, f"pf_{species}_season")
    smin = float(bz.get("season_min", 0.55))
    trend = col(df, "water_temp_trend_3d")
    fall = (doy >= FALL[0]) & (doy <= FALL[1])
    push = np.where(fall & ~np.isnan(trend), 0.92 + 0.08 * _ramp(-trend, 0, 2), 1.0)   # cooling sets the run moving
    migration = np.where(np.isnan(season), np.nan, _ramp(season, smin, smin + 0.35) * push)
    temp = trapezoid(col(df, "water_temp_f"), *bz.get("temp_f", [45, 50, 60, 66]), floor=0.02)
    temp = np.where(np.isnan(temp), 0.6, temp)

    src = bait if bait is not None else df
    groups = [g for g in bz.get("bait", []) if f"bait_{g}" in src]
    clim = (np.max(np.vstack([src[f"bait_{g}"].to_numpy(dtype=float) / 100 for g in groups]), axis=0)
            if groups else np.full(len(df), 0.6))
    flush = col(df, "bait_flush")
    place = (cfg.get("environment", {}).get(spot.environment, 0.8) * cfg.get("water_body", {}).get(spot.water_body, 0.85))
    bait_v = np.where(np.isnan(flush), 0.9 * clim, 0.3 * clim + 0.7 * flush) * place

    return pd.DataFrame({"migration": migration, "temperature": temp, "bait": np.clip(bait_v, 0, 1),
                         "access": access_component(df, spot), "tide": tide_component(df, spot),
                         "light": light_component(col(df, "sun_alt"), col(df, "cloud_pct")),
                         "front": front_component(df)}, index=df.index)


def geo_score(comp: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    w = np.array([weights.get(c, 0.0) for c in COMPONENTS], dtype=float)
    v = np.clip(comp[list(COMPONENTS)].to_numpy(dtype=float), 0.01, 1)
    ok = ~np.isnan(v)
    lw = np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.exp(np.sum(np.where(ok, np.log(np.nan_to_num(v, nan=1.0)), 0) * lw, axis=1) / lw.sum(axis=1))
    return np.where(lw.sum(axis=1) > 0, 100 * g, np.nan)


def _floors_ok(comp: pd.DataFrame, floors: dict[str, float]) -> np.ndarray:
    ok = np.ones(len(comp), dtype=bool)
    for c, lo in floors.items():
        if c in comp:
            ok &= np.nan_to_num(comp[c].to_numpy(dtype=float), nan=0.0) >= lo
    return ok


# ----------------------------------------------------------------------------- scoring
def blitz_scores(df: pd.DataFrame, spot: Spot, bait: pd.DataFrame | None = None, now: pd.Timestamp | None = None,
                 mode: str = "forecast") -> pd.DataFrame:
    """Hourly blitz columns: blitz_score, blitz_level, blitz_species, bz_<component>, blitz_cap (why not an alert)."""
    cfg = blitz_config()
    sps = blitz_species()
    n = len(df)
    empty = pd.DataFrame({"blitz_score": np.full(n, np.nan), "blitz_level": "", "blitz_species": None,
                          "blitz_cap": "", **{f"bz_{c}": np.full(n, np.nan) for c in COMPONENTS}}, index=df.index)
    if not sps or n == 0:
        return empty
    weights = cfg.get("weights", {})
    best_score = np.full(n, -1.0)
    best_sp = np.full(n, None, dtype=object)
    best_vals = np.full((n, len(COMPONENTS)), np.nan)
    per_sp = {}
    for sp in sps:
        comp = species_components(df, spot, sp, bait)
        sc = np.nan_to_num(geo_score(comp, weights), nan=-1.0)
        per_sp[f"blitz_{sp}"] = np.where(sc >= 0, sc, np.nan)
        take = sc > best_score
        best_score = np.where(take, sc, best_score)
        best_sp = np.where(take, sp, best_sp)
        best_vals = np.where(take[:, None], comp[list(COMPONENTS)].to_numpy(dtype=float), best_vals)
    best_comp = pd.DataFrame(best_vals, index=df.index, columns=list(COMPONENTS))
    score = np.where(best_score >= 0, best_score, np.nan)

    a, w = cfg.get("alert", {}), cfg.get("watch", {})
    is_alert = (np.nan_to_num(score) >= a.get("score", 74)) & _floors_ok(best_comp, a.get("min", {}))
    is_watch = (np.nan_to_num(score) >= w.get("score", 62)) & _floors_ok(best_comp, w.get("min", {}))

    # Alerts need verified inputs; otherwise they are held at "watch" with the reason.
    caps = np.full(n, "", dtype=object)
    wt_kind = df.get("water_temp_kind", pd.Series("missing", index=df.index)).astype(str).to_numpy()
    no_temp = ~np.char.startswith(wt_kind.astype(str), "measured")
    no_tide = np.isnan(col(df, "tide_ft"))
    flush_measured = df.get("bait_flush_measured", pd.Series(False, index=df.index)).fillna(False).to_numpy(dtype=bool)
    doy = df.index.tz_convert(spot.timezone).day_of_year.to_numpy()
    fall = (doy >= FALL[0]) & (doy <= FALL[1])
    no_flush = fall & ~flush_measured
    lead_cap = np.zeros(n, dtype=bool)
    if mode == "forecast" and now is not None:
        lead_h = (df.index - pd.Timestamp(now)) / pd.Timedelta(hours=1)
        lead_cap = np.asarray(lead_h > float(cfg.get("max_alert_lead_hours", 72)))
    for mask, why in ((no_temp, "no measured water temperature"), (no_tide, "no tide predictions"),
                      (no_flush, "bay/ocean bait-flush gauges unavailable (estimated)"),
                      (lead_cap, "more than 3 days out — wind forecast too uncertain")):
        hit = is_alert & mask
        caps = np.where(hit, np.where(caps == "", why, caps + "; " + why), caps)
    alert_ok = is_alert & (caps == "")
    level = np.where(alert_ok, "alert", np.where(is_watch | is_alert, "watch", ""))

    out = pd.DataFrame({"blitz_score": score, "blitz_level": level, "blitz_species": best_sp, "blitz_cap": caps,
                        **{f"bz_{c}": best_comp[c].to_numpy() for c in COMPONENTS}, **per_sp}, index=df.index)
    return out


def apply_sightings(bl: pd.DataFrame, sightings: list[dict[str, Any]], max_age_h: float = 36) -> pd.DataFrame:
    """A logged birds-diving / blitz sighting nearby confirms bait + fish: lift 'watch' hours to 'alert'
    for ``max_age_h`` after the sighting. Never creates a signal where the setup is absent."""
    if not sightings or bl.empty:
        return bl
    bl = bl.copy()
    if "blitz_confirmed" not in bl:
        bl["blitz_confirmed"] = ""
    for s in sightings:
        t0 = pd.Timestamp(s["time"])
        t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
        win = (bl.index >= t0.floor("h")) & (bl.index <= t0 + pd.Timedelta(hours=max_age_h))
        lift = win & (bl["blitz_level"] == "watch").to_numpy()
        bl.loc[lift, "blitz_level"] = "alert"
        note = s.get("label") or "birds / blitz reported nearby in your Fishing Log"
        bl.loc[win & (bl["blitz_level"] != "").to_numpy(), "blitz_confirmed"] = note
    return bl


# ------------------------------------------------------------------------------ windows
@dataclass
class BlitzWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    peak: pd.Timestamp
    score: float
    level: str                     # alert | watch
    species: str
    reasons: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)
    cap: str = ""
    confirmed: str = ""

    @property
    def label(self) -> str:
        return LEVEL_TEXT.get(self.level, "")


def blitz_windows(hourly: pd.DataFrame, spot: Spot, start: pd.Timestamp | None = None,
                  end: pd.Timestamp | None = None) -> list[BlitzWindow]:
    """Contiguous hours at 'watch' or 'alert' between start and end, strongest first."""
    if "blitz_level" not in hourly:
        return []
    d = hourly.loc[start:end] if (start is not None or end is not None) else hourly
    lv = d["blitz_level"].fillna("").astype(str).to_numpy()
    out: list[BlitzWindow] = []
    i = 0
    while i < len(d):
        if not lv[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(d) and lv[j + 1] and (d.index[j + 1] - d.index[j]) <= pd.Timedelta(hours=1):
            j += 1
        seg = d.iloc[i:j + 1]
        k = int(np.nanargmax(seg["blitz_score"].to_numpy(dtype=float)))
        row = seg.iloc[k]
        level = "alert" if (seg["blitz_level"] == "alert").any() else "watch"
        caps = [str(c) for c in seg.get("blitz_cap", pd.Series(dtype=object)) if isinstance(c, str) and c]
        conf = [str(c) for c in seg.get("blitz_confirmed", pd.Series(dtype=object)) if isinstance(c, str) and c]
        out.append(BlitzWindow(start=seg.index[0], end=seg.index[-1] + pd.Timedelta(hours=1), peak=seg.index[k],
                               score=round(float(row["blitz_score"]), 1), level=level,
                               species=str(row.get("blitz_species") or ""), reasons=reasons(row, spot),
                               limits=limits(row), cap=caps[0] if caps and level != "alert" else "",
                               confirmed=conf[0] if conf else ""))
        i = j + 1
    return sorted(out, key=lambda w: (LEVEL_RANK[w.level], w.score), reverse=True)


def _f(v, spec="{:.0f}") -> str:
    try:
        return spec.format(v) if v is not None and np.isfinite(float(v)) else "?"
    except (TypeError, ValueError):
        return "?"


def reasons(row: pd.Series, spot: Spot) -> list[str]:
    """Plain-English drivers at the peak hour."""
    out = []
    bay, ocean = row.get("bay_ref_temp_f"), row.get("ocean_ref_temp_f")
    fl = row.get("bait_flush")
    if fl is not None and np.isfinite(fl) and fl >= 0.55 and bool(row.get("bait_flush_measured")):
        out.append(f"bait flushing out of the bays (bay {_f(bay)}°F vs ocean {_f(ocean)}°F)")
    elif fl is not None and np.isfinite(fl) and fl >= 0.55:
        out.append("water cooling — fall bait on the move (est.)")
    else:
        out.append(f"bait in season ({str(row.get('bait_top') or '').replace('_', ' ')})")
    sp = row.get("blitz_species")
    name = species_meta(sp)["short"].lower() if sp else "fish"
    out.append(f"{_f(row.get('water_temp_f'), '{:.0f}')}°F water, {name} migration on")
    on = row.get("onshore_mph")
    rel = "onshore" if (on or 0) > 2 else "offshore" if (on or 0) < -2 else "side-shore"
    surf = row.get("nearshore_wave_ft")
    out.append(f"{_f(row.get('wind_mph'))} mph {compass(row.get('wind_dir'))} ({rel})"
               + (f", {_f(surf, '{:.1f}')} ft surf" if spot.is_open_ocean else ""))
    stage = row.get("tide_stage") or "tide n/a"
    ph = row.get("current_phase")
    out.append(f"{stage}" + (f" · {ph}" if isinstance(ph, str) and ph in ("ebb", "flood") else ""))
    return out


def limits(row: pd.Series) -> list[str]:
    """Weakest factors (below 0.75) at the peak hour, weakest first."""
    items = [(row.get(f"bz_{c}"), LABELS[c]) for c in COMPONENTS]
    items = [(float(v), lab) for v, lab in items if v is not None and np.isfinite(v) and v < 0.75]
    return [f"{lab.lower()} {v * 100:.0f}%" for v, lab in sorted(items)]


def summary_text(w: BlitzWindow) -> str:
    return "; ".join(w.reasons[:3])
