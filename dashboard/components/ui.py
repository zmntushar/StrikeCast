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


def chip(species: str, value: float | None, sub: str = "") -> str:
    _, color = score_label(value)
    short = species_meta(species)["short"]
    return (f'<span class="sc-chip" style="background:{color}">'
            f'<span class="t">{esc(short)}</span><span class="n">{fmt(value)}</span>'
            + (f'<span class="t">{esc(sub)}</span>' if sub else "") + "</span>")


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
