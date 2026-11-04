"""Synchronized solar + tide charts with the recommended 90-minute window.

Two vertically stacked panels share one 24-hour time axis (plotly shared x axes,
so zooming/panning one moves the other). The recommended window is a single
translucent rectangle drawn in paper coordinates so it spans BOTH panels, with
its start and end times printed at the boundaries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.features import current_model as cm
from app.features.solar import ASTRONOMICAL_DEG, CIVIL_DEG, NAUTICAL_DEG, band_segments, light_band
from app.models.window_optimizer import WindowPick

FONT = dict(family="Inter, system-ui, sans-serif", color="#cbd5e1", size=12)
GRID = "rgba(148,163,184,0.12)"
REGION = {"night": "rgba(15,23,42,0.0)", "twilight": "rgba(167,139,250,0.22)", "daylight": "rgba(250,204,21,0.12)"}
FLOOD, EBB = "rgba(52,211,153,0.13)", "rgba(244,114,182,0.13)"
BEST, SECOND, COMPARE = "#22d3ee", "#a5b4fc", "#fb923c"


def time_label(ts: pd.Timestamp, tz: str) -> str:
    """Local wall-clock label, e.g. '6:15 AM'."""
    return pd.Timestamp(ts).tz_convert(tz).strftime("%I:%M %p").lstrip("0")


def _x(ts, tz: str):
    """Local wall-clock (naive) for plotting, matching the rest of the dashboard."""
    t = pd.Timestamp(ts) if not isinstance(ts, pd.DatetimeIndex) else ts
    return t.tz_convert(tz).tz_localize(None)


def _runs(index: pd.DatetimeIndex, labels: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    out, s = [], 0
    for i in range(1, len(index) + 1):
        if i == len(index) or labels[i] != labels[s]:
            out.append((index[s], index[min(i, len(index) - 1)], labels[s]))
            s = i
    return out


def _window_shape(fig: go.Figure, p: WindowPick, tz: str, color: str, fill: str, name: str, dash: str | None,
                  label: str, top: bool = True) -> None:
    """Rectangle across both panels. The recommended window labels its times above the charts; secondary
    windows label theirs below, so adjacent windows never print over each other."""
    x0, x1 = _x(p.start, tz), _x(p.end, tz)
    fig.add_shape(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1, fillcolor=fill,
                  line=dict(color=color, width=2 if dash is None else 1.5, dash=dash), layer="above", name=name)
    # Recommended window: times on the row ABOVE the panel titles, score above them, so nothing collides
    # with the titles. Secondary windows: times below the x-axis tick labels, score tag inside the box.
    y, ya, shift = (1.0, "bottom", 22) if top else (0.0, "top", -20)
    for x, anchor, t, nm in ((x0, "right", p.start, "start"), (x1, "left", p.end, "end")):
        fig.add_annotation(x=x, y=y, xref="x", yref="paper", text=f"<b>{time_label(t, tz)}</b>", showarrow=False,
                           xanchor=anchor, yanchor=ya, yshift=shift, font=dict(color=color, size=11),
                           name=f"{name}_{nm}", bgcolor="rgba(2,6,23,0.6)")
    score = f"{p.score:.0f}" if np.isfinite(p.score) else "n/a"
    xm = x0 + (x1 - x0) / 2
    if top:
        fig.add_annotation(x=xm, y=1.0, xref="x", yref="paper", text=f"<b>{label} · {score}/100</b>", showarrow=False,
                           yanchor="bottom", yshift=42, font=dict(color="#04121f", size=11), bgcolor=color,
                           borderpad=3, name=f"{name}_score")
    else:
        fig.add_annotation(x=xm, y=0.0, xref="x", yref="paper", text=f"{label} · {score}", showarrow=False,
                           yanchor="bottom", yshift=4, font=dict(color=color, size=10), bgcolor="rgba(2,6,23,0.85)",
                           name=f"{name}_score")


def _legend_box(fig: go.Figure, name: str, color: str, symbol: str) -> None:
    """Legend-only entry explaining a window box."""
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, hoverinfo="skip",
                             marker=dict(symbol=symbol, size=13, color=color, line=dict(color=color, width=2))),
                  row=1, col=1)


def window_figure(result, compare: WindowPick | None = None, show_second: bool = True, height: int = 640) -> go.Figure:
    tz = result.tz
    f = result.frame
    x = _x(f.index, tz)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.46, 0.54],
                        specs=[[{}], [{"secondary_y": True}]],
                        subplot_titles=("Solar cycle · sun elevation (°)",
                                        "Tidal cycle · predicted height (ft MLLW)"
                                        + (" & current" if result.current_source != cm.MISSING else "")))

    # ---------------- upper: solar
    for a, b, band in band_segments(f.index, f["sun_alt"].to_numpy()):
        fig.add_vrect(x0=_x(a, tz), x1=_x(b, tz), fillcolor=REGION[band], line_width=0, layer="below", row=1, col=1,
                      exclude_empty_subplots=False)
    bands = light_band(f["sun_alt"].to_numpy())
    fig.add_trace(go.Scatter(
        x=x, y=f["sun_alt"], name="Sun elevation", line=dict(color="#fbbf24", width=2.4),
        customdata=np.stack([bands, f["solar"].to_numpy() * 100, f["total"].to_numpy()], axis=-1),
        hovertemplate="%{y:.1f}° · %{customdata[0]}<br>light suitability %{customdata[1]:.0f}%"
                      "<br><b>overall suitability %{customdata[2]:.0f}</b><extra></extra>"), row=1, col=1)
    for lvl, lab in ((0, "horizon"), (CIVIL_DEG, "civil"), (NAUTICAL_DEG, "nautical"), (ASTRONOMICAL_DEG, "astronomical")):
        fig.add_hline(y=lvl, line=dict(color="rgba(148,163,184,0.35)", width=1, dash="dot"), row=1, col=1,
                      annotation_text=lab, annotation_position="top left", annotation_font=dict(size=8, color="#64748b"))
    marks = [("sunrise", "Sunrise", "triangle-up"), ("solar_noon", "Solar noon", "star"), ("sunset", "Sunset", "triangle-down")]
    mx, my, mt, ms = [], [], [], []
    for key, lab, sym in marks:
        t = result.sun.get(key)
        if t is None:
            continue
        ts = pd.Timestamp(t)
        i = f.index.get_indexer([ts], method="nearest")[0]
        mx.append(_x(ts, tz))
        my.append(float(f["sun_alt"].iloc[i]))
        mt.append(f"{lab} {time_label(ts, tz)}")
        ms.append(sym)
    if mx:
        fig.add_trace(go.Scatter(x=mx, y=my, mode="markers+text", text=mt, textposition="top center",
                                 marker=dict(symbol=ms, size=11, color="#fde68a", line=dict(color="#78350f", width=1)),
                                 textfont=dict(size=10, color="#fde68a"), name="Sun events", hoverinfo="text"),
                      row=1, col=1)

    # ---------------- lower: tide (+ current on its own axis)
    if f["tide_dir"].notna().any():
        for a, b, d in _runs(f.index, f["tide_dir"].fillna("none").to_numpy()):
            if d in ("incoming", "outgoing"):
                fig.add_vrect(x0=_x(a, tz), x1=_x(b, tz), fillcolor=FLOOD if d == "incoming" else EBB, line_width=0,
                              layer="below", row=2, col=1, exclude_empty_subplots=False)
    fig.add_trace(go.Scatter(x=x, y=f["tide_ft"], name="Tide height (predicted)", line=dict(color="#38bdf8", width=2.4),
                             customdata=np.stack([f["tide_stage"].fillna("—").to_numpy()], axis=-1),
                             hovertemplate="%{y:.2f} ft · %{customdata[0]}<extra></extra>"), row=2, col=1)
    hi = [e for e in result.inputs.tide_events if f.index[0] <= pd.Timestamp(e.time) <= f.index[-1]]
    for typ, sym, col, lab in (("H", "triangle-up", "#e0f2fe", "High"), ("L", "triangle-down", "#7dd3fc", "Low")):
        ev = [e for e in hi if e.type == typ]
        if ev:
            fig.add_trace(go.Scatter(
                x=[_x(pd.Timestamp(e.time), tz) for e in ev], y=[e.height_ft for e in ev], mode="markers+text",
                text=[f"{lab[0]} {time_label(pd.Timestamp(e.time), tz)}" for e in ev],
                textposition="top center" if typ == "H" else "bottom center", textfont=dict(size=10, color=col),
                marker=dict(symbol=sym, size=11, color=col), name=f"{lab} tide",
                hovertemplate=f"{lab} tide %{{y:.2f}} ft<extra></extra>"), row=2, col=1)
    src = result.current_source
    if src in (cm.STATION, cm.PROXY_STATION):
        name = "Predicted current (kt, + flood / − ebb)" + (" · nearby-station proxy" if src == cm.PROXY_STATION else "")
        fig.add_trace(go.Scatter(x=x, y=f["current_kt"], name=name, line=dict(color="#34d399", width=1.8, dash="dash"),
                                 hovertemplate="%{y:+.2f} kt<extra>current</extra>"), row=2, col=1, secondary_y=True)
        if result.slack:
            fig.add_trace(go.Scatter(x=[_x(t, tz) for t in result.slack], y=[0] * len(result.slack), mode="markers",
                                     marker=dict(symbol="circle-open", size=12, color="#f8fafc", line=dict(width=2)),
                                     name="Slack current (predicted)",
                                     hovertext=[f"Slack {time_label(t, tz)}" for t in result.slack], hoverinfo="text"),
                          row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="current (kt)", showgrid=False, zeroline=True, zerolinecolor="rgba(52,211,153,0.3)",
                         row=2, col=1, secondary_y=True)
    elif src == cm.TIDE_PHASE:
        fig.add_trace(go.Scatter(x=x, y=f["current_norm"], name="Approx. tidal-flow index (0–1, not a velocity)",
                                 line=dict(color="#34d399", width=1.4, dash="dot"),
                                 hovertemplate="%{y:.2f} (estimated)<extra>flow index</extra>"),
                      row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="flow index (est.)", range=[0, 1.05], showgrid=False, row=2, col=1, secondary_y=True)

    # ---------------- windows spanning both panels
    if result.best is not None:
        _legend_box(fig, "Recommended 90-min window", BEST, "square")
    if result.second is not None and show_second:
        _window_shape(fig, result.second, tz, SECOND, "rgba(165,180,252,0.07)", "second_window", "dot", "2nd",
                      top=False)
        _legend_box(fig, "Second-best window (backup, no overlap)", SECOND, "square-open")
    if compare is not None:
        _window_shape(fig, compare, tz, COMPARE, "rgba(251,146,60,0.10)", "compare_window", "dash", "Compare", top=False)
        _legend_box(fig, "Your comparison window", COMPARE, "square-open")
    if result.best is not None:
        _window_shape(fig, result.best, tz, BEST, "rgba(34,211,238,0.16)", "best_window", None, "Best")

    now = pd.Timestamp.now(tz="UTC")
    if f.index[0] <= now <= f.index[-1]:
        fig.add_vline(x=_x(now, tz), line=dict(color="#f8fafc", width=1, dash="dot"))

    e = f["sun_alt"]
    fig.update_yaxes(range=[float(e.min()) - 6, float(e.max()) + 12], title_text="°", row=1, col=1)
    fig.update_yaxes(title_text="ft", row=2, col=1, secondary_y=False)
    fig.update_xaxes(range=[x[0], x[-1]], tickformat="%-I %p", nticks=8, tickangle=0, gridcolor=GRID,
                     showspikes=True, spikemode="across", spikecolor="#22d3ee", spikethickness=1)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    # Panel titles: left-aligned, away from the window labels drawn over the plot.
    fig.update_annotations(selector=dict(xref="paper"), font=dict(size=12, color="#94a3b8"), x=0, xanchor="left")
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(2,6,23,0.35)", font=FONT,
                      margin=dict(l=8, r=8, t=78, b=70), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.17, yanchor="top", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                      hoverlabel=dict(bgcolor="#0f172a", font_size=12), dragmode="pan")
    return fig
