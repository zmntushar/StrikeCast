"""Small HTML builders and formatting helpers for the dashboard."""
from __future__ import annotations

import html
import math
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.geo import compass
from app.models.bite import LEGEND, score_label
from app.models.common import species_meta


def esc(s: object) -> str:
    return html.escape(str(s))


def isnum(v) -> bool:
    return isinstance(v, (int, float, np.floating, np.integer)) and not (isinstance(v, float) and math.isnan(v)) and np.isfinite(v)


def fmt(v, spec: str = "{:.0f}", none: str = "—") -> str:
    return spec.format(v) if isnum(v) else none


def hour_label(ts: pd.Timestamp | datetime, with_day: bool = True) -> str:
    ts = pd.Timestamp(ts)
    h = ts.hour % 12 or 12
    ampm = "am" if ts.hour < 12 else "pm"
    return (f"{ts:%a} " if with_day else "") + f"{h}{ampm}"


def window_text(w, tz: str) -> str:
    s = pd.Timestamp(w.start).tz_convert(tz)
    e = pd.Timestamp(w.end).tz_convert(tz)
    same = s.date() == e.date() or (e - s) <= pd.Timedelta(hours=12)
    return f"{hour_label(s)}–{hour_label(e, with_day=not same)}"


def chip(species: str, value: float | None, sub: str = "", pop: str = "") -> str:
    """Score chip; ``pop`` is optional HTML shown on hover / tap / keyboard focus."""
    _, color = score_label(value)
    short = species_meta(species)["short"]
    body = (f'<span class="sc-chip" style="background:{color}">'
            f'<span class="t">{esc(short)}</span><span class="n">{fmt(value)}</span>'
            + (f'<span class="t">{esc(sub)}</span>' if sub else "") + "</span>")
    if not pop:
        return body
    return f'<span class="sc-tip" tabindex="0">{body}<span class="sc-pop">{pop}</span></span>'


def species_popover(fc, sp: str, d: pd.DataFrame, wins: list, reg: dict, rec: dict, why: str) -> str:
    """Detail card for one species at one spot: best windows, peak conditions, why, tackle, regs."""
    tz = fc.spot.timezone
    meta = species_meta(sp)
    col = f"bite_{sp}"
    s = d[col].dropna() if col in d else pd.Series(dtype=float)
    peak_ts = pd.Timestamp(wins[0].peak) if wins else (s.idxmax() if not s.empty else None)
    pr = fc.hourly.loc[peak_ts] if peak_ts is not None and peak_ts in fc.hourly.index else None
    lab, colr = score_label(s.max() if not s.empty else None)

    if wins:
        win_html = "".join(
            f'<div class="pw"><b>{window_text(w, tz)}</b> '
            f'<span style="color:{score_label(w.peak_score)[1]}">peak {w.peak_score:.0f}</span></div>'
            for w in wins[:3])
    elif peak_ts is not None:
        win_html = (f'<div class="pw">No standout window · best hour '
                    f'<b>{hour_label(pd.Timestamp(peak_ts).tz_convert(tz))}</b> ({s.max():.0f})</div>')
    else:
        win_html = '<div class="pw">No data</div>'

    cond = ""
    if pr is not None:
        cur = pr.get("current_kt")
        cur_t = f"{abs(cur):.1f} kt {pr.get('current_phase')}" if isnum(cur) else "current est."
        cond = (f'<div class="pg"><span>Tide</span>{esc(pr.get("tide_stage") or "—")} · {esc(cur_t)}</div>'
                f'<div class="pg"><span>Light</span>{esc(pr.get("light_phase"))} · clouds {fmt(pr.get("cloud_pct"))}%</div>'
                f'<div class="pg"><span>Wind</span>{fmt(pr.get("wind_mph"))} mph {compass(pr.get("wind_dir"))}</div>'
                f'<div class="pg"><span>Surf · water</span>{esc(surf_headline(pr))} · '
                f'{fmt(pr.get("water_temp_f"), "{:.1f}")}°F</div>')

    if reg.get("open") is True:
        reg_badge = '<span class="pb ok">season open</span>'
    elif reg.get("open") is False:
        reg_badge = '<span class="pb no">season closed · catch &amp; release</span>'
    else:
        reg_badge = ""

    tackle = ""
    if rec.get("primary"):
        notes = "".join(f"<li>{esc(n)}</li>" for n in rec.get("notes", []))
        bait = (f'<li><b>Match the bait:</b> {esc(rec["bait_match"])}</li>'
                if rec.get("bait_match") and meta.get("diet") else "")
        tackle = (f'<div class="ph">🎣 Tackle for these conditions</div><ul>'
                  f'<li><b>Lure:</b> {esc(rec["primary"])}</li>'
                  f'<li><b>Jig:</b> {esc(rec["jig"])}</li>'
                  f'<li><b>Color:</b> {esc(rec["color"])}</li>{bait}'
                  f'<li>{esc(rec["rig"])}</li>{notes}</ul>'
                  f'<div class="pn">Sized for ~{rec["depth_ft"]} ft of water in casting range, '
                  f'{rec["current_kt"]} kt current, {esc(rec["clarity"])} water.</div>')

    regs = (f'<div class="pn">Size {esc(reg.get("size", ""))} · bag {esc(reg.get("bag", ""))} — verify with NYSDEC</div>'
            if reg.get("size") else "")
    return (f'<span class="pt" style="border-color:{meta["color"]}">{esc(meta["name"])} · {esc(fc.spot.short)} '
            f'<span style="color:{colr}">{lab}</span></span>{reg_badge}'
            f'<div class="ph">⏱ Best windows</div>{win_html}'
            + (f'<div class="ph">At the peak hour</div>{cond}' if cond else "")
            + (f'<div class="ph">Why</div><div class="pn" style="color:#cbd5e1">{esc(why)}</div>' if why else "")
            + tackle + regs)


# ------------------------------------------------------------------ surf
TIPS = {
    "surf": "Estimated significant height of BREAKING waves at this beach (the bigger, frequent waves; occasional sets "
            "run ~1.5x). Computed from offshore waves with shoaling, refraction, sheltering by land and depth-limited "
            "breaking. The range covers wave-direction (±10°) and offshore-height uncertainty plus formula scatter.",
    "chop": "Locally generated wind waves (fetch-limited growth, SPM 1984) from the wind blowing over the bay or Sound "
            "toward this shore. Open-ocean swell does not reach this shoreline.",
    "offshore": "Significant wave height Hs (average of the highest third of waves) about 12 km offshore in 20-30 m of "
                "water, with peak period and the direction waves come FROM. This is NOT the surf at the beach.",
    "swell": "Longer-period waves from distant weather: height, peak period, direction. Swell and wind-wave ENERGIES "
             "combine into the offshore Hs; their heights do not add.",
    "windwave": "Short-period waves raised by recent local wind: height, peak period, direction.",
    "trend": "Expected change over the next ~6 hours: building, steady or subsiding.",
    "confidence": "How much to trust the surf number: data freshness, agreement between the buoy, the wave model and "
                  "the NWS forecast, forecast lead time, and how oblique or land-sheltered the incoming swell is.",
    "source": "Where the offshore conditions came from. Measured buoy data correct the model near the present; the NWS "
              "zone forecast is the fallback and a cross-check.",
    "updated": "Time of the latest buoy observation, model download and NWS forecast issuance used here.",
    "buoy": "Latest MEASURED offshore reading at the NOAA NDBC buoy (significant height, dominant period, direction).",
    "safety": "Surf SAFETY for shore anglers, separate from the Bite Index: considers breaker height, long-period surges "
              "on rocks and inlet ebb currents.",
}
CONF_COLOR = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}
TREND_ICON = {"building": "↗", "steady": "→", "subsiding": "↘"}


def hint(label: str, tip_key: str) -> str:
    return f'<span class="sc-h" tabindex="0" data-tip="{esc(TIPS.get(tip_key, tip_key))}">{esc(label)}</span>'


def wave_txt(h, t, d) -> str:
    if not isnum(h):
        return "—"
    return f"{h:.1f} ft" + (f" · {t:.0f} s" if isnum(t) else "") + (f" · {compass(d)}" if isnum(d) else "")


def surf_source_text(fc, src) -> str:
    a = fc.surf.get("anchor") or {}
    if src == "model":
        txt = "Open-Meteo Marine (MFWAM)"
        if a.get("applied"):
            txt += f" + NDBC {a.get('buoy')} correction"
        return txt
    if src == "nws":
        return f"NWS coastal waters forecast {fc.spot.marine_zone}"
    if src == "buoy":
        return f"NDBC {a.get('buoy') or 'buoy'} (measured)"
    if src and "wind" in str(src):
        return "Wind (NWS / Open-Meteo) + NOAA DEM fetch"
    return "—"


def _updated_text(fc) -> str:
    tz = fc.spot.timezone
    parts = []
    b = fc.latest_buoy()
    if b:
        parts.append(f"buoy {hour_label(pd.Timestamp(b['time']).tz_convert(tz), False)}")
    m = next((s for s in fc.sources if s.feature.startswith("waves / swell") and s.fetched_at), None)
    if m:
        parts.append(f"model {pd.Timestamp(m.fetched_at).tz_convert(tz):%I:%M %p}".replace(" 0", " "))
    if fc.marine.get("issued") is not None:
        parts.append(f"NWS {pd.Timestamp(fc.marine['issued']).tz_convert(tz):%I:%M %p}".replace(" 0", " "))
    return " · ".join(parts) or "—"


def surf_headline(row) -> str:
    """Short surf text for maps / tables."""
    s = row.get("surf_ft")
    if not isnum(s):
        return "surf estimate unavailable"
    kind = row.get("surf_kind")
    rng = f"{fmt(row.get('surf_lo_ft'), '{:.1f}')}–{fmt(row.get('surf_hi_ft'), '{:.1f}')} ft"
    return (f"~{s:.1f} ft wind chop ({rng})" if kind == "wind chop" else f"{s:.1f} ft surf ({rng})")


def surf_block(fc, row) -> str:
    """Surf section of a spot card: surf, offshore, swell, wind waves, trend, source, updated."""
    s = row.get("surf_ft")
    conf = row.get("surf_confidence")
    trend = row.get("surf_trend")
    sheltered = not fc.spot.is_open_ocean
    head_lab = hint("Wind chop" if sheltered else "Surf", "chop" if sheltered else "surf")
    if isnum(s):
        main = (f'<b class="v">{s:.1f} ft</b> <span class="rng">{fmt(row.get("surf_lo_ft"), "{:.1f}")}–'
                f'{fmt(row.get("surf_hi_ft"), "{:.1f}")} ft</span>')
        badges = (f'<span class="sc-conf" style="--c:{CONF_COLOR.get(conf, "#64748b")}" tabindex="0" '
                  f'data-tip="{esc(TIPS["confidence"])}">● {esc(conf or "—")}</span>')
        if trend:
            badges += (f'<span class="sc-trend" tabindex="0" data-tip="{esc(TIPS["trend"])}">'
                       f'{TREND_ICON.get(trend, "")} {esc(trend)}</span>')
    else:
        main, badges = '<b class="v na">Surf estimate unavailable</b>', ""
    rows = []
    if sheltered:
        env = "Sound" if fc.spot.exposure_class == "sound" else "bay"
        rows.append(f'<div class="sc-surf-note">Sheltered {env} — no ocean surf reaches this shoreline.</div>')
        if isnum(s):
            rows.append(f'<div><span>Wind over water</span>{fmt(row.get("wind_mph"))} mph {compass(row.get("wind_dir"))} · '
                        f'{fmt(row.get("chop_fetch_km"), "{:.1f}")} km fetch · {fmt(row.get("chop_period_s"), "{:.0f}")} s</div>')
    else:
        rows.append(f'<div><span>{hint("Offshore", "offshore")}</span>'
                    f'{wave_txt(row.get("offshore_hs_ft"), row.get("offshore_tp_s"), row.get("offshore_dir"))}</div>')
        sw2 = (f' + {wave_txt(row.get("bc_swell2_ft"), row.get("bc_swell2_tp_s"), row.get("bc_swell2_dir"))}'
               if isnum(row.get("bc_swell2_ft")) and row.get("bc_swell2_ft") >= 0.5 else "")
        rows.append(f'<div><span>{hint("Swell", "swell")}</span>'
                    f'{wave_txt(row.get("bc_swell_ft"), row.get("bc_swell_tp_s"), row.get("bc_swell_dir"))}{sw2}</div>')
        rows.append(f'<div><span>{hint("Wind waves", "windwave")}</span>'
                    f'{wave_txt(row.get("bc_windwave_ft"), row.get("bc_windwave_tp_s"), row.get("bc_windwave_dir"))}</div>')
        b = fc.latest_buoy()
        if b:
            rows.append(f'<div><span>{hint("Buoy " + str(b["station"]), "buoy")}</span>'
                        f'{wave_txt(b["hs_ft"], b["period_s"], b["dir"])} <i>measured</i></div>')
        if (row.get("surf_shadow_share") or 0) > 0.3:
            rows.append('<div class="sc-surf-note">Swell partly blocked by nearby land at this angle — '
                        'estimate leans on diffraction; real surf may be larger.</div>')
    nws = ""
    if isnum(row.get("nws_seas_hi_ft")):
        lo, hi = row.get("nws_seas_lo_ft"), row.get("nws_seas_hi_ft")
        word = "seas" if fc.spot.is_open_ocean else "waves"
        nws = (f'<div><span>NWS {esc(fc.spot.marine_zone or "")}</span>{word} {lo:.0f}–{hi:.0f} ft'
               + (f", occ. {row.get('nws_seas_occ_ft'):.0f}" if isnum(row.get("nws_seas_occ_ft")) else "") + "</div>")
    safety = row.get("surf_safety")
    saf = (f'<span class="sc-safe s-{esc(str(safety).lower())}" tabindex="0" data-tip="{esc(TIPS["safety"])}">'
           f'{esc(safety)}</span>') if safety else ""
    flags = row.get("surf_flags") or ""
    return (f'<div class="sc-surf"><div class="sc-surf-top"><span class="lab">🌊 {head_lab}</span> {main} {badges} {saf}</div>'
            f'<div class="sc-surf-grid">{"".join(rows)}{nws}'
            f'<div><span>{hint("Source", "source")}</span>{esc(surf_source_text(fc, row.get("surf_source")))}</div>'
            f'<div><span>{hint("Updated", "updated")}</span>{esc(_updated_text(fc))}</div></div>'
            + (f'<div class="sc-surf-flag">⚠ {esc(flags)}</div>' if flags else "")
            + (f'<div class="sc-surf-note">{esc(str(row.get("surf_quality")))}</div>' if isnum(s) and row.get("surf_quality") else "")
            + "</div>")


def kind_badge(kind: str) -> str:
    k = (kind or "missing").split(" ")[0].replace("+", "")
    k = k if k in ("measured", "predicted", "model", "interpolated", "computed", "heuristic", "missing") else "model"
    return f'<span class="sc-kind k-{k}">{esc(kind)}</span>'


def legend_html() -> str:
    items = "".join(f'<span><i style="background:{c}"></i>{lab} {lo}+</span>' if lo else f'<span><i style="background:{c}"></i>{lab} &lt;35</span>'
                    for lo, lab, c in LEGEND)
    return f'<div class="sc-legend"><b>Bite Index</b>{items}<span style="color:#64748b">· relative suitability, not a calibrated probability</span></div>'


def wind_text(row: pd.Series) -> str:
    on = row.get("onshore_mph")
    rel = ("onshore" if isnum(on) and on > 2 else "offshore" if isnum(on) and on < -2 else "side-shore")
    return f"{fmt(row.get('wind_mph'))} mph {compass(row.get('wind_dir'))} · {rel}"


def tide_text(row: pd.Series) -> str:
    st = row.get("tide_stage") or "—"
    cur = row.get("current_kt")
    c = f" · {abs(cur):.1f} kt {row.get('current_phase')}" if isnum(cur) else ""
    return f"{st}{c}"
