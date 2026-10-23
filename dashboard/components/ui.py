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
                f'<div class="pg"><span>Surf · water</span>~{fmt(pr.get("nearshore_wave_ft"), "{:.1f}")} ft · '
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
