"""Dynamic 90-minute fishing-window optimization.

For a spot, local date and species: build a 5-minute grid from local midnight to
the next local midnight + 90 min (so windows may cross midnight), score every
grid point (``app.models.window_scoring``), evaluate every 90-minute window that
starts on that date (``app.models.window_optimizer``), and return the best and
second-best windows with an explanation, the factor breakdown and a separate
forecast-confidence rating. For today, only windows that have not started yet
are considered (unless the whole day has already passed).

The score is RELATIVE environmental favorability (0-100), not a probability of
catching fish. Nothing here invents current speeds or measurements: missing
inputs are dropped from the weighted mean and listed in the confidence notes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.features import current_model as cm
from app.features.solar import solar_events
from app.models.common import species_meta
from app.models.feeding_profiles import COMPONENTS, FeedingProfile, load_profile
from app.models.window_optimizer import STEP, WINDOW, WindowPick, interval_score, rank_windows
from app.models.window_scoring import component_frame
from app.schemas.env import CurrentEvent, TideEvent

LABELS = {"solar": "Light (sun elevation)", "current": "Tidal current", "tide": "Tide phase & direction",
          "habitat": "Habitat & structure", "seasonal": "Season", "temperature": "Water temperature",
          "conditions": "Wind, surf, pressure, clarity, bait"}


@dataclass
class WindowInputs:
    """Everything the optimizer needs for one spot (no network access inside the engine)."""

    spot_id: str
    spot_name: str
    lat: float
    lon: float
    tz: str
    environment: str
    exposure_class: str = "open_ocean"
    structure: list[str] = field(default_factory=list)
    tide_events: list[TideEvent] = field(default_factory=list)
    tide_station: str | None = None
    tide_distance_km: float | None = None
    current_events: list[CurrentEvent] = field(default_factory=list)
    current_station: str | None = None
    current_is_proxy: bool = True
    hourly: pd.DataFrame | None = None          # hourly forecast frame (UTC) with pf_/ff_/fishability columns
    data_quality: float | None = None           # 0-100 overall data-quality score of the forecast
    custom_spot: bool = False
    water_temp_kind: str | None = None          # measured | measured + model trend | model | missing
    generated_at: datetime | None = None


@dataclass(frozen=True)
class WindowResult:
    species: str
    profile: FeedingProfile
    inputs: WindowInputs
    day: date
    frame: pd.DataFrame                  # 5-min grid (UTC index) with drivers, components and total
    starts: pd.DatetimeIndex             # candidate starts actually considered
    current_source: str
    best: WindowPick | None
    second: WindowPick | None
    elapsed: bool                        # True when today's remaining windows were exhausted
    sun: dict[str, Any]
    slack: list[pd.Timestamp]
    confidence: dict[str, Any]
    explanation: dict[str, Any]
    computed_at: datetime

    @property
    def tz(self) -> str:
        return self.inputs.tz

    def day_frame(self) -> pd.DataFrame:
        """Grid rows for the local day plus the 90-min tail (chart range)."""
        return self.frame

    def local(self, ts: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(ts).tz_convert(self.tz)


# ----------------------------------------------------------------- inputs
def inputs_from_forecast(fc) -> WindowInputs:
    """Adapt a ``SpotForecast`` (services.forecast) to optimizer inputs."""
    sp = fc.spot
    tide_src = next((s for s in fc.sources if s.feature.startswith("tide times")), None)
    wt_kind = None
    if "water_temp_kind" in fc.hourly:
        k = fc.hourly["water_temp_kind"].dropna()
        wt_kind = str(k.iloc[-1]) if not k.empty else None
        now = pd.Timestamp.now(tz="UTC").floor("h")
        if now in fc.hourly.index and pd.notna(fc.hourly.loc[now, "water_temp_kind"]):
            wt_kind = str(fc.hourly.loc[now, "water_temp_kind"])
    return WindowInputs(
        spot_id=sp.id, spot_name=sp.name, lat=sp.lat, lon=sp.lon, tz=sp.timezone, environment=sp.environment,
        exposure_class=sp.exposure_class, structure=list(sp.structure), tide_events=list(fc.tide_events),
        tide_station=sp.tide_station, tide_distance_km=tide_src.distance_km if tide_src else None,
        current_events=list(getattr(fc, "current_events", []) or []), current_station=sp.current_station,
        current_is_proxy=sp.current_is_proxy, hourly=fc.hourly, data_quality=fc.quality.get("score"),
        custom_spot=sp.custom, water_temp_kind=wt_kind, generated_at=fc.generated_at)


def day_grid(day: date, tz: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """(evaluation grid, candidate starts) in UTC. Handles 23/25-hour daylight-saving days."""
    d0 = pd.Timestamp(day).tz_localize(tz).tz_convert("UTC")
    d1 = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC")
    grid = pd.date_range(d0, d1 + WINDOW, freq=STEP)
    starts = pd.date_range(d0, d1 - STEP, freq=STEP)
    return grid, starts


# --------------------------------------------------------------- compute
def compute_window(inputs: WindowInputs, species: str, day: date, now: datetime | None = None,
                   calibration_sessions: int = 0) -> WindowResult:
    profile = load_profile(species)
    grid, starts = day_grid(day, inputs.tz)
    frame, source = component_frame(
        grid, lat=inputs.lat, lon=inputs.lon, environment=inputs.environment, exposure_class=inputs.exposure_class,
        species=species, profile=profile, tide_events=inputs.tide_events, current_events=inputs.current_events,
        current_is_proxy=inputs.current_is_proxy, hourly=inputs.hourly)
    score = frame["total"]

    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    upcoming = starts[starts >= now_ts.ceil(STEP)]
    elapsed = False
    considered = starts
    if now_ts > starts[0]:            # today (or a past date): prefer windows that haven't started
        if len(upcoming):
            considered = upcoming
        else:
            elapsed = True
    picks = rank_windows(score, considered)
    best = picks[0] if picks else None
    second = picks[1] if len(picks) > 1 else None

    sun = solar_events(inputs.lat, inputs.lon, day, inputs.tz)
    slack = cm.slack_times(inputs.current_events, grid[0], grid[-1])
    conf = confidence(inputs, source, frame, day, now_ts, calibration_sessions)
    expl = explain(frame, profile, best, source, inputs) if best else {
        "summary": "Not enough data to score any 90-minute window for this date.", "factors": [], "bullets": []}
    return WindowResult(species=species, profile=profile, inputs=inputs, day=day, frame=frame, starts=considered,
                        current_source=source, best=best, second=second, elapsed=elapsed, sun=sun, slack=slack,
                        confidence=conf, explanation=expl, computed_at=datetime.now(timezone.utc))


def window_for_forecast(fc, species: str, day: date, now: datetime | None = None,
                        calibration_sessions: int = 0) -> WindowResult:
    return compute_window(inputs_from_forecast(fc), species, day, now, calibration_sessions)


def compare_window(result: WindowResult, start: pd.Timestamp) -> dict[str, Any]:
    """Score a user-chosen 90-min window. Never modifies the recommended window."""
    g = result.frame.index
    s = pd.Timestamp(start)
    s = s.tz_localize(result.tz) if s.tzinfo is None else s
    s = s.tz_convert("UTC").floor(STEP)
    s = min(max(s, g[0]), g[-1] - WINDOW)
    sc = interval_score(result.frame["total"], s)
    pick = WindowPick(start=s, end=s + WINDOW, score=round(sc, 1) if np.isfinite(sc) else float("nan"), rank=0)
    return {"pick": pick, "factors": factor_table(result.frame, result.profile, pick, result.current_source, result.inputs)}


def window_changed(prev: WindowPick | None, new: WindowPick | None) -> bool:
    """True when a data update moved (or created/removed) the recommended window."""
    if prev is None or new is None:
        return (prev is None) != (new is None)
    return prev.start != new.start


# ----------------------------------------------------------- explanation
def _seg(frame: pd.DataFrame, pick: WindowPick) -> pd.DataFrame:
    return frame.loc[pick.start:pick.end]


def _day(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[:-int(WINDOW / STEP)] if len(frame) > WINDOW / STEP else frame


def light_phrase(e0: float, e1: float, emin: float, emax: float) -> str:
    rising = e1 >= e0
    if emax < -18:
        return "full darkness"
    if emin < -0.833 <= emax:
        return "dawn lighting (sunrise inside the window)" if rising else "dusk lighting (sunset inside the window)"
    if emax < -0.833:
        return "pre-dawn twilight" if rising else "evening twilight" if emin > -18 else "night into twilight"
    if emax < 12:
        return "low morning sun" if rising else "low evening sun"
    return "full daylight"


def _q(v: float, words: tuple[str, str, str]) -> str:
    if not np.isfinite(v):
        return words[2]
    return words[0] if v >= 0.8 else words[1] if v >= 0.55 else words[2]


def _fmt_t(ts: pd.Timestamp, tz: str) -> str:
    return pd.Timestamp(ts).tz_convert(tz).strftime("%I:%M %p").lstrip("0")


def tide_phrase(seg: pd.DataFrame) -> str:
    """Noun phrase without article, e.g. 'mid-incoming tide' or 'high-tide turn'."""
    dirs = seg["tide_dir"].dropna()
    if dirs.empty:
        return "unknown tide (no tide predictions)"
    if dirs.nunique() > 1:
        turn = seg["next_turn_type"].dropna()
        return f"{turn.iloc[0]}-tide turn" if not turn.empty else "tide turn"
    mid = seg.iloc[len(seg) // 2]
    st = str(mid.get("tide_stage") or dirs.iloc[0])
    if "slack" in st:
        return f"{st} (tide height turning)"
    words = st.split(" ")
    return (f"{words[0]}-{' '.join(words[1:])} tide" if words[0] in ("early", "mid", "late") else f"{st} tide")


def current_phrase(seg: pd.DataFrame, source: str) -> str:
    if source == cm.MISSING:
        return "unknown current (no data)"
    if source == cm.TIDE_PHASE:
        n = seg["current_norm"].dropna()
        return (f"estimated tidal flow ({n.min() * 100:.0f}–{n.max() * 100:.0f}% of peak from tide phase)"
                if not n.empty else "estimated tidal flow")
    kt = seg["current_kt"].dropna()
    if kt.empty:
        return "current outside prediction range"
    ph = seg["current_phase"].dropna()
    phases = sorted(set(ph)) if not ph.empty else []
    lo, hi = np.abs(kt).min(), np.abs(kt).max()
    word = "/".join(phases) if phases else ""
    return f"{lo:.1f}–{hi:.1f} kt {word} current".replace("  ", " ")


def explain(frame: pd.DataFrame, profile: FeedingProfile, pick: WindowPick, source: str,
            inputs: WindowInputs) -> dict[str, Any]:
    seg = _seg(frame, pick)
    factors = factor_table(frame, profile, pick, source, inputs)
    m = {f["component"]: f["window"] for f in factors}
    e = seg["sun_alt"].to_numpy()
    light = light_phrase(e[0], e[-1], float(np.min(e)), float(np.max(e)))
    sol = _q(m.get("solar", np.nan), ("Favorable", "Fair", "Unfavorable"))
    tid = _q(m.get("tide", np.nan), ("productive", "workable", "less-favored"))
    cur = _q(m.get("current", np.nan), ("well-suited", "suitable", "less-than-ideal"))
    summary = (f"{sol} {light} overlaps with a {tid} {tide_phrase(seg)} and {cur} "
               f"{current_phrase(seg, source)}.")
    ranked = sorted([f for f in factors if np.isfinite(f["contribution"])], key=lambda f: f["contribution"], reverse=True)
    helps = [f for f in ranked if f["contribution"] > 0.005][:3]
    hurts = [f for f in sorted(factors, key=lambda f: f["window"] if np.isfinite(f["window"]) else 9)
             if np.isfinite(f["window"]) and f["window"] < 0.5][:2]
    bullets = []
    if helps:
        bullets.append("Beats the rest of the day mainly on: " + "; ".join(
            f"{f['label'].lower()} ({f['detail']})" for f in helps) + ".")
    if hurts:
        bullets.append("Holding it back: " + "; ".join(f"{f['label'].lower()} ({f['detail']})" for f in hurts) + ".")
    ctx = [f for f in factors if f["component"] in ("seasonal", "temperature", "habitat") and np.isfinite(f["window"])]
    if ctx:
        bullets.append("Context (same all day): " + "; ".join(f"{f['label'].lower()} {f['window'] * 100:.0f}%"
                                                              for f in ctx) + ".")
    if source == cm.TIDE_PHASE:
        bullets.append("No current station here: tidal flow is approximated from the tide curve, so real slack "
                       "and peak flow may be offset — confidence is reduced.")
    return {"summary": summary, "factors": factors, "bullets": bullets, "light": light}


def factor_table(frame: pd.DataFrame, profile: FeedingProfile, pick: WindowPick, source: str,
                 inputs: WindowInputs) -> list[dict[str, Any]]:
    seg, day = _seg(frame, pick), _day(frame)
    tz = inputs.tz
    meta = species_meta(profile.species)
    out = []
    for c in COMPONENTS:
        w = profile.weights.get(c, 0.0)
        if w <= 0:
            continue
        win = float(np.nanmean(seg[c])) if seg[c].notna().any() else float("nan")
        dmean = float(np.nanmean(day[c])) if day[c].notna().any() else float("nan")
        contrib = w * (win - dmean) if np.isfinite(win) and np.isfinite(dmean) else float("nan")
        out.append({"component": c, "label": LABELS[c], "weight": round(w, 3), "window": win, "day": dmean,
                    "contribution": contrib, "detail": _detail(c, seg, source, inputs, meta, tz),
                    "kind": _kind(c, source, inputs), "limiting": profile.limiting.get(c)})
    if not seg.empty:
        fq = float(seg["fishability"].mean())
        out.append({"component": "fishability", "label": "Shore fishability & safety (multiplier)", "weight": None,
                    "window": fq, "day": float(day["fishability"].mean()), "contribution": 0.0,
                    "detail": ("no wind/surf/weather problems forecast" if fq >= 0.95 else
                               "wind, surf or weather make shore fishing harder or unsafe"),
                    "kind": "forecast (model)", "limiting": None})
    return out


def _detail(c: str, seg: pd.DataFrame, source: str, inputs: WindowInputs, meta: dict, tz: str) -> str:
    if seg.empty:
        return "no data"
    if c == "solar":
        e = seg["sun_alt"].to_numpy()
        cl = seg["cloud_pct"].dropna()
        return (f"sun {e[0]:+.0f}° → {e[-1]:+.0f}°, {light_phrase(e[0], e[-1], e.min(), e.max())}"
                + (f", clouds ~{cl.mean():.0f}%" if not cl.empty else ""))
    if c == "current":
        txt = current_phrase(seg, source)
        sl = [t for t in cm.slack_times(inputs.current_events, seg.index[0], seg.index[-1])]
        if sl:
            txt += f"; slack at {', '.join(_fmt_t(t, tz) for t in sl)}"
        if source == cm.PROXY_STATION:
            txt += f" (station {inputs.current_station}, used as proxy)"
        elif source == cm.STATION:
            txt += f" (station {inputs.current_station})"
        return txt
    if c == "tide":
        a, b = seg.iloc[0], seg.iloc[-1]
        if pd.isna(a.get("tide_ft")):
            return "no tide predictions"
        return (f"{a.get('tide_stage') or '—'} → {b.get('tide_stage') or '—'}; "
                f"{a['tide_ft']:.1f} → {b['tide_ft']:.1f} ft MLLW")
    if c == "habitat":
        return "spot structure: " + (", ".join(inputs.structure) or inputs.environment)
    if c == "seasonal":
        return f"migration calendar for {meta.get('short', '')}".strip()
    if c == "temperature":
        wt = seg["water_temp_f"].dropna()
        t = meta.get("temp_f", [0, 0, 0, 0])
        return (f"water {wt.mean():.1f}°F" if not wt.empty else "water temp unavailable") + f" (best {t[1]}–{t[2]}°F)"
    if c == "conditions":
        return "hourly forecast of wind, surf, pressure trend, clarity and bait"
    return ""


def _kind(c: str, source: str, inputs: WindowInputs) -> str:
    if c == "solar":
        return "computed"
    if c == "current":
        return {cm.STATION: "predicted", cm.PROXY_STATION: "predicted (proxy)", cm.TIDE_PHASE: "estimated",
                cm.MISSING: "missing"}[source]
    if c == "tide":
        return "predicted" if inputs.tide_events else "missing"
    if c == "temperature":
        return inputs.water_temp_kind or "model"
    if c == "conditions":
        return "forecast (model)"
    return "heuristic"


# ------------------------------------------------------------- confidence
def confidence(inputs: WindowInputs, source: str, frame: pd.DataFrame, day: date, now: pd.Timestamp,
               calibration_sessions: int) -> dict[str, Any]:
    """Forecast confidence (0-100), separate from the suitability score."""
    parts: dict[str, float] = {}
    missing: list[str] = []
    estimated: list[str] = []
    reasons: list[str] = []

    parts["current"] = {cm.STATION: 1.0, cm.PROXY_STATION: 0.7, cm.TIDE_PHASE: 0.4, cm.MISSING: 0.0}[source]
    if source == cm.PROXY_STATION:
        estimated.append(f"Current: nearby station {inputs.current_station} used as a proxy for this shoreline")
    elif source == cm.TIDE_PHASE:
        estimated.append("Current: approximate tidal-phase index (no current-station predictions)")
    elif source == cm.MISSING:
        missing.append("tidal current")

    if inputs.tide_events and frame["tide_ft"].notna().any():
        d = inputs.tide_distance_km
        parts["tide"] = 1.0 if d is None or d <= 10 else 0.85 if d <= 25 else 0.7
        if d is not None and d > 10:
            reasons.append(f"Tide station is {d:.0f} km away")
    else:
        parts["tide"] = 0.0
        missing.append("tide predictions")

    dq = inputs.data_quality
    parts["data"] = (dq / 100) if dq is not None else 0.5
    if dq is not None and dq < 70:
        reasons.append(f"Overall data quality {dq:.0f}/100")
    for c, name in (("conditions", "wind/surf/weather forecast"), ("temperature", "water temperature"),
                    ("seasonal", "seasonal model"), ("habitat", "habitat model")):
        if frame[c].isna().all():
            missing.append(name)
    if inputs.water_temp_kind == "model":
        estimated.append("Water temperature: model sea-surface temperature (no fresh measurement)")

    parts["location"] = 0.7 if inputs.custom_spot else 1.0
    if inputs.custom_spot:
        reasons.append("Custom location: stations and shoreline were matched automatically")

    n = calibration_sessions
    parts["calibration"] = 0.4 if n < 5 else 0.6 if n < 15 else 0.8 if n < 40 else 1.0
    reasons.append(f"Species model is a heuristic profile; {n} logged session{'s' if n != 1 else ''} with predictions")

    w = {"current": 0.30, "tide": 0.25, "data": 0.20, "location": 0.10, "calibration": 0.15}
    score = 100 * sum(w[k] * parts[k] for k in w)
    lead = (pd.Timestamp(day) - pd.Timestamp(now.tz_convert(inputs.tz).date())).days
    if lead > 5:
        score *= 0.85
        reasons.append(f"{lead} days ahead — weather forecast skill drops")
    elif lead > 3:
        score *= 0.92
        reasons.append(f"{lead} days ahead — weather forecast skill drops")
    score = round(float(np.clip(score, 0, 100)))
    label = "High" if score >= 75 else "Medium" if score >= 55 else "Low"
    return {"score": score, "label": label, "parts": parts, "missing": missing, "estimated": estimated,
            "reasons": reasons}
