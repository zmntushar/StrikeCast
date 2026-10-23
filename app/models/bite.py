"""Bite Index = presence x feeding x fishability, plus plain-English explanations.

Bite Index (0-100) ranks hours and spots for a species. It is a relative
suitability score built from the separate indices — NOT a calibrated catch
probability. Fishability captures shore-angler practicality and safety (wind,
surf on jetties, lightning, heavy rain, fog).
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import regulations_config
from app.core.geo import compass
from app.models.common import col, species_meta
from app.schemas.env import FactorScore, Spot, Window

LEGEND = [  # (min score, label, color)
    (80, "Hot", "#ef4444"),
    (65, "Very good", "#f97316"),
    (50, "Good", "#facc15"),
    (35, "Fair", "#22c55e"),
    (0, "Slow", "#64748b"),
]


def score_label(v: float | None) -> tuple[str, str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "No data", "#334155"
    for lo, lab, colr in LEGEND:
        if v >= lo:
            return lab, colr
    return "Slow", "#64748b"


def fishability(df: pd.DataFrame, spot: Spot, species: str) -> pd.DataFrame:
    fcfg = species_meta(species)["fishability"]
    max_w = fcfg["max_wind_mph"]
    max_wave = fcfg["max_wave_ft"] * (0.8 if spot.environment == "jetty" else 1.0)
    wind = col(df, "wind_mph")
    gust = col(df, "gust_mph")
    wave = col(df, "nearshore_wave_ft")
    if "inlet_wave_ft" in df and spot.environment == "inlet":
        wave = np.fmax(wave, col(df, "inlet_wave_ft"))   # ebb running into the swell steepens the inlet bar
    precip = col(df, "precip_in", 0)
    vis = col(df, "visibility_mi")
    thunder = df.get("thunder", pd.Series(False, index=df.index)).fillna(False).to_numpy(dtype=bool)

    q_wind = np.where(np.isnan(wind), 1.0, np.where(wind <= 0.6 * max_w, 1.0,
                      np.where(wind <= max_w, 1 - 0.75 * (wind - 0.6 * max_w) / (0.4 * max_w), 0.12)))
    q_gust = np.where(np.nan_to_num(gust) > max_w + 12, 0.6, 1.0)
    q_wave = np.where(np.isnan(wave), 1.0, np.where(wave <= 0.6 * max_wave, 1.0,
                      np.where(wave <= max_wave, 1 - 0.7 * (wave - 0.6 * max_wave) / (0.4 * max_wave), 0.1)))
    q_rain = np.where(precip > 0.25, 0.65, np.where(precip > 0.1, 0.85, 1.0))
    q_fog = np.where(np.nan_to_num(vis, nan=10) < 0.3, 0.85, 1.0)
    q_thunder = np.where(thunder, 0.25, 1.0)
    q = np.clip(q_wind * q_gust * q_wave * q_rain * q_fog * q_thunder, 0, 1)
    return pd.DataFrame({f"fishability_{species}": q}, index=df.index)


def combine(df: pd.DataFrame, species: str) -> pd.Series:
    p = df[f"presence_{species}"].to_numpy() / 100
    f = df[f"feeding_{species}"].to_numpy() / 100
    q = df[f"fishability_{species}"].to_numpy()
    # The 1.2 exponent keeps "Hot" (80+) for hours where nearly everything lines up.
    return pd.Series(np.clip(100 * ((p ** 0.6) * (f ** 0.4)) ** 1.2 * q, 0, 100), index=df.index)


# ---------------------------------------------------------------- regulations
def season_status(species: str, day: date) -> dict[str, Any]:
    regs = regulations_config().get("regulations", {})
    key = species_meta(species).get("regulation_key", species)
    r = regs.get(key)
    if not r:
        return {"open": None, "text": "no regulation info"}
    md = day.strftime("%m-%d")
    is_open = any(a <= md <= b for a, b in r.get("open", []))
    return {"open": is_open, "size": r.get("size"), "bag": r.get("bag"), "note": r.get("note"),
            "text": ("Open" if is_open else "Closed (catch & release only)") + f" · {r.get('size')} · {r.get('bag')}"}


# --------------------------------------------------------------- explanations
def _fmt(v: Any, fmt: str = "{:.0f}", none: str = "n/a") -> str:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return none
        return fmt.format(v)
    except (TypeError, ValueError):
        return none


def explain_hour(row: pd.Series, species: str, spot: Spot) -> list[FactorScore]:
    """Factor-by-factor breakdown for one hour, ordered by contribution."""
    cfg = species_meta(species)
    w = cfg["feeding_weights"]
    items: list[FactorScore] = []

    def add(name: str, key: str, weight: float, detail: str) -> None:
        val = row.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            items.append(FactorScore(factor=name, score=0.5, weight=0.0, detail=detail + " — data missing"))
        else:
            items.append(FactorScore(factor=name, score=float(val), weight=weight, detail=detail))

    t = cfg["temp_f"]
    add("Season", f"pf_{species}_season", 0.5, f"day {row.name.tz_convert(spot.timezone).day_of_year} of the {spot.water_body} migration calendar")
    add("Water temperature", f"pf_{species}_temp", 0.5,
        f"{_fmt(row.get('water_temp_f'), '{:.1f}')}°F (best {t[1]}–{t[2]}°F)")
    add("Structure", f"pf_{species}_structure", 0.25, f"best feature here: {row.get(f'pf_{species}_structure_tag', '')}")
    if cfg.get("diet"):
        top = str(row.get("bait_top", "")).replace("_", " ")
        add("Bait", f"ff_{species}_bait", w.get("bait", 0) + 0.15, f"strongest bait signal: {top} (index {_fmt(row.get('bait_index'))})")
    cur = row.get("current_kt")
    cur_txt = (f"{abs(cur):.1f} kt {row.get('current_phase')}" if isinstance(cur, (int, float)) and not np.isnan(cur)
               else "tide-rate proxy")
    add("Current", f"ff_{species}_current", w.get("current", 0),
        f"{cur_txt}, {_fmt((row.get('current_norm') or np.nan) * 100)}% of peak"
        + (" (nearby inlet station — proxy)" if row.get("current_is_proxy") else ""))
    add("Tide stage", f"ff_{species}_tide_stage", w.get("tide_stage", 0),
        f"{row.get('tide_stage') or 'n/a'}; {_fmt(row.get('hours_to_turn'), '{:.1f}')} h to {row.get('next_turn_type') or 'turn'}")
    add("Light", f"ff_{species}_light", w.get("light", 0), f"{row.get('light_phase')}, clouds {_fmt(row.get('cloud_pct'))}%")
    add("Wind", f"ff_{species}_wind", w.get("wind", 0),
        f"{_fmt(row.get('wind_mph'))} mph from {compass(row.get('wind_dir'))} "
        f"({'onshore' if (row.get('onshore_mph') or 0) > 2 else 'offshore' if (row.get('onshore_mph') or 0) < -2 else 'side-shore'} {abs(row.get('onshore_mph') or 0):.0f} mph)")
    kind = row.get("surf_kind")
    surf_txt = (f"{_fmt(row.get('surf_ft'), '{:.1f}')} ft wind chop (sheltered water)" if kind == "wind chop"
                else f"{_fmt(row.get('surf_ft'), '{:.1f}')} ft breaking surf "
                     f"({_fmt(row.get('surf_lo_ft'), '{:.1f}')}–{_fmt(row.get('surf_hi_ft'), '{:.1f}')}), "
                     f"{_fmt(row.get('offshore_tp_s'), '{:.0f}')} s from {compass(row.get('offshore_dir'))}")
    from app.models.feeding import WAVE_WEIGHT_BY_EXPOSURE
    add("Surf", f"ff_{species}_waves", w.get("waves", 0) * WAVE_WEIGHT_BY_EXPOSURE.get(spot.exposure_class, 1.0), surf_txt)
    if w.get("clarity", 0) > 0:
        ci = row.get("clarity_index")
        word = "clear" if (ci or 0) < 0.25 else "some color" if (ci or 0) < 0.55 else "dirty"
        add("Water clarity", f"ff_{species}_clarity", w.get("clarity", 0),
            f"{word} (proxy {_fmt(ci, '{:.2f}')}: recent surf, rain, storms)")
    add("Pressure trend", f"ff_{species}_pressure", w.get("pressure", 0),
        f"{_fmt(row.get('pressure_tend_3h'), '{:+.1f}')} hPa / 3 h")
    if w.get("temp_trend", 0) > 0:
        add("Temperature trend", f"ff_{species}_temp_trend", w.get("temp_trend", 0),
            f"{_fmt(row.get('water_temp_trend_3d'), '{:+.1f}')}°F over 3 days")
    storm = row.get("storm_phase")
    if storm and storm != "none":
        items.append(FactorScore(factor="Storm", score=float(np.clip(0.5 * row.get(f"ff_{species}_storm", 1), 0, 1)),
                                 weight=0.2, detail=f"{storm}-storm conditions"))
    fq = row.get(f"fishability_{species}")
    if fq is not None and not np.isnan(fq) and fq < 0.95:
        items.append(FactorScore(factor="Fishability", score=float(fq) * 0.5, weight=0.3,
                                 detail="wind / surf / weather make shore fishing harder or unsafe"))
    return sorted(items, key=lambda x: x.contribution, reverse=True)


CONTEXT_FACTORS = ("Season", "Water temperature", "Structure")


def narrative(row: pd.Series, species: str, spot: Spot) -> str:
    """Plain-English summary: seasonal context, then what helps/hurts this hour."""
    items = explain_hour(row, species, spot)
    name = species_meta(species)["name"].split(" (")[0]
    ctx = {i.factor: i for i in items if i.factor in CONTEXT_FACTORS}
    hourly = [i for i in items if i.factor not in CONTEXT_FACTORS]
    season, temp = ctx.get("Season"), ctx.get("Water temperature")
    parts = []
    if season and temp:
        both = min(season.score, temp.score)
        mood = ("prime" if both >= 0.8 else "favorable" if both >= 0.55 else "marginal" if both >= 0.25 else "poor")
        parts.append(f"Seasonal setup for {name.lower()} is {mood} (water {temp.detail}).")
    good = [i for i in hourly if i.contribution > 0.02][:3]
    bad = [i for i in sorted(hourly, key=lambda x: x.contribution) if i.contribution < -0.02][:2]
    if good:
        parts.append("This hour, working in your favor: " + "; ".join(f"{g.factor.lower()} — {g.detail}" for g in good) + ".")
    if bad:
        parts.append("Holding it back: " + "; ".join(f"{b.factor.lower()} — {b.detail}" for b in bad) + ".")
    st = ctx.get("Structure")
    if st and st.score < 0.5:
        parts.append(f"This shoreline is not ideal {name.lower()} habitat ({st.detail}).")
    return " ".join(parts) or "Conditions are middle-of-the-road across the board."


def tactic(row: pd.Series, species: str) -> str:
    t = species_meta(species).get("tactics", {})
    phase = row.get("light_phase")
    if (row.get("nearshore_wave_ft") or 0) > 4 and "rough" in t:
        return t["rough"]
    if species == "striped_bass" and str(row.get("bait_top")) == "adult_bunker" and (row.get("bait_index") or 0) > 60:
        return t.get("bunker", t.get("day", ""))
    if phase == "night":
        return t.get("night", "")
    if phase in ("dawn", "dusk"):
        return t.get("low_light", "")
    return t.get("day", "")


# -------------------------------------------------------------------- windows
def best_windows(bite: pd.Series, species: str, tz: str, *, min_score: float = 45, top: int = 6) -> list[Window]:
    s = bite.dropna()
    if s.empty:
        return []
    thr = max(min_score, float(np.nanpercentile(s, 80)))
    good = s >= thr
    windows: list[Window] = []
    run: list[pd.Timestamp] = []
    for ts, ok in good.items():
        if ok and (not run or (ts - run[-1]) <= pd.Timedelta(hours=1)):
            run.append(ts)
        else:
            if run:
                windows.append(_mk_window(s, run, species))
            run = [ts] if ok else []
    if run:
        windows.append(_mk_window(s, run, species))
    windows.sort(key=lambda w: w.peak_score, reverse=True)
    return windows[:top]


def _mk_window(s: pd.Series, run: list[pd.Timestamp], species: str) -> Window:
    seg = s.loc[run[0]:run[-1]]
    peak = seg.idxmax()
    return Window(species=species, start=run[0].to_pydatetime(),
                  end=(run[-1] + pd.Timedelta(hours=1)).to_pydatetime(), peak=peak.to_pydatetime(),
                  peak_score=round(float(seg.max()), 1), mean_score=round(float(seg.mean()), 1))
