"""Plotly charts for the dashboard (dark, transparent backgrounds)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.features.seasonal import seasonal_factor
from app.models.common import species_meta

FONT = dict(family="Inter, system-ui, sans-serif", color="#cbd5e1", size=12)
GRID = "rgba(148,163,184,0.12)"
NIGHT = "rgba(99,102,241,0.13)"
BITE_SCALE = [[0.0, "#334155"], [0.34, "#64748b"], [0.35, "#22c55e"], [0.4999, "#22c55e"], [0.5, "#facc15"],
              [0.6499, "#facc15"], [0.65, "#f97316"], [0.7999, "#f97316"], [0.8, "#ef4444"], [1.0, "#ef4444"]]


def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(2,6,23,0.35)", font=FONT,
                      margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                      hoverlabel=dict(bgcolor="#0f172a", font_size=12))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showspikes=True, spikecolor="#22d3ee", spikethickness=1)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _nights(fig: go.Figure, daily: list[dict], tz: str, x0, x1) -> None:
    for i, d in enumerate(daily):
        ss = d.get("sunset")
        nxt = daily[i + 1].get("sunrise") if i + 1 < len(daily) else None
        if ss is None:
            continue
        a = pd.Timestamp(ss).tz_convert(tz).tz_localize(None)
        b = pd.Timestamp(nxt).tz_convert(tz).tz_localize(None) if nxt else a + pd.Timedelta(hours=11)
        a, b = max(a, x0), min(b, x1)
        if a < b:
            fig.add_vrect(x0=a, x1=b, fillcolor=NIGHT, line_width=0, layer="below", row="all", col=1)
    if daily and daily[0].get("sunrise") is not None:
        first_rise = pd.Timestamp(daily[0]["sunrise"]).tz_convert(tz).tz_localize(None)
        if x0 < first_rise:
            fig.add_vrect(x0=x0, x1=first_rise, fillcolor=NIGHT, line_width=0, layer="below", row="all", col=1)


BLITZ_COLOR = {"alert": ("239,68,68", "#f87171", "🔥 Blitz alert"), "watch": ("245,158,11", "#fbbf24", "👀 Blitz watch")}


def add_blitz(fig: go.Figure, fc, d: pd.DataFrame, row: int | None = 1) -> None:
    """Blitz windows as shaded bands with a label, plus star markers on the hours that qualify."""
    tz = fc.spot.timezone
    if d.empty or "blitz_level" not in d:
        return
    x0, x1 = d.index.min(), d.index.max() + pd.Timedelta(hours=1)
    for w in fc.blitz or []:
        if w.end <= x0 or w.start >= x1:
            continue
        rgb, line, label = BLITZ_COLOR[w.level]
        a = pd.Timestamp(w.start).tz_convert(tz).tz_localize(None)
        b = pd.Timestamp(w.end).tz_convert(tz).tz_localize(None)
        kw = dict(row=row, col=1) if row else {}
        fig.add_vrect(x0=a, x1=b, fillcolor=f"rgba({rgb},{0.20 if w.level == 'alert' else 0.12})",
                      line=dict(color=line, width=1.5 if w.level == "alert" else 1, dash=None if w.level == "alert" else "dot"),
                      layer="below", **kw)
        if row == 1:
            fig.add_annotation(x=a + (b - a) / 2, y=99, xref="x", yref="y", text=label.split(" ")[0], showarrow=False,
                               font=dict(size=14), yanchor="top", hovertext=f"{label} · {w.score:.0f}/100", row=1, col=1)
    lv = d["blitz_level"].fillna("").astype(str)
    x = d.index.tz_convert(tz).tz_localize(None)
    for level in ("alert", "watch"):
        m = (lv == level).to_numpy()
        if not m.any():
            continue
        rgb, line, label = BLITZ_COLOR[level]
        fig.add_trace(go.Scatter(x=x[m], y=d["blitz_score"].to_numpy()[m], mode="markers", name=label,
                                 marker=dict(symbol="star", size=12 if level == "alert" else 9, color=line,
                                             line=dict(color="#0b1220", width=1)),
                                 customdata=d["blitz_species"].fillna("").to_numpy()[m],
                                 hovertemplate=f"{label}: %{{y:.0f}}/100 · %{{customdata}}<extra></extra>"),
                      row=row or 1, col=1)


def forecast_timeline(fc, species: list[str], d: pd.DataFrame, bands: list | None = None) -> go.Figure:
    """``bands`` = [(start, end[, rank]), ...] highlighted on the Bite Index panel (best = cyan, 2nd = violet).
    Blitz alerts / watches are drawn as red / amber bands with stars at the qualifying hours."""
    tz = fc.spot.timezone
    x = d.index.tz_convert(tz).tz_localize(None)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
                        row_heights=[0.34, 0.22, 0.22, 0.22],
                        specs=[[{}], [{"secondary_y": True}], [{}], [{"secondary_y": True}]],
                        subplot_titles=("Bite Index (0–100)", "Tide (ft MLLW) & current (kt, + flood / − ebb)",
                                        "Wind (mph) · arrows show where it blows", "Nearshore surf (ft, est.) & water temp (°F)"))
    for sp in species:
        meta = species_meta(sp)
        fig.add_trace(go.Scatter(x=x, y=d[f"bite_{sp}"], name=meta["short"], line=dict(color=meta["color"], width=2.4),
                                 hovertemplate="%{y:.0f}"), row=1, col=1)
    spans = bands if bands is not None else [(w.start, w.end) for w in (fc.windows.get(species[0]) or [])[:6]]
    for span in spans:
        w0, w1 = span[0], span[1]
        rgb = "192,132,252" if len(span) > 2 and span[2] == 2 else "34,211,238"
        a = pd.Timestamp(w0).tz_convert(tz).tz_localize(None)
        b = pd.Timestamp(w1).tz_convert(tz).tz_localize(None)
        if b >= x.min() and a <= x.max():
            fig.add_vrect(x0=a, x1=b, fillcolor=f"rgba({rgb},0.12)", line=dict(color=f"rgba({rgb},.5)", width=1),
                          row=1, col=1)
    fig.add_hline(y=80, line=dict(color="rgba(239,68,68,.35)", dash="dot"), row=1, col=1)
    fig.add_hline(y=65, line=dict(color="rgba(249,115,22,.30)", dash="dot"), row=1, col=1)
    add_blitz(fig, fc, d, row=1)

    fig.add_trace(go.Scatter(x=x, y=d["tide_ft"], name="Tide", fill="tozeroy", line=dict(color="#38bdf8", width=2),
                             fillcolor="rgba(56,189,248,0.18)", hovertemplate="%{y:.1f} ft"), row=2, col=1)
    if d["current_kt"].notna().any():
        cur = d["current_kt"]
        fig.add_trace(go.Bar(x=x, y=cur, name="Current", marker_color=np.where(cur >= 0, "#34d399", "#f472b6"),
                             opacity=0.55, hovertemplate="%{y:+.1f} kt"), row=2, col=1, secondary_y=True)
    else:
        fig.add_trace(go.Scatter(x=x, y=d["current_norm"], name="Current (proxy, share of peak)",
                                 line=dict(color="#34d399", dash="dot")), row=2, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=x, y=d["gust_mph"], name="Gusts", line=dict(color="#a78bfa", width=1, dash="dot"),
                             hovertemplate="%{y:.0f} mph"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["wind_mph"], name="Wind", line=dict(color="#e2e8f0", width=2),
                             hovertemplate="%{y:.0f} mph"), row=3, col=1)
    step = max(1, len(d) // 40)
    sub = d.iloc[::step]
    fig.add_trace(go.Scatter(x=x[::step], y=sub["wind_mph"], mode="markers", name="Direction", showlegend=False,
                             marker=dict(symbol="arrow", size=13, angle=(sub["wind_dir"].fillna(0) + 180).to_numpy(),
                                         color=np.where(sub["onshore_mph"] > 2, "#fbbf24", "#94a3b8")),
                             hoverinfo="skip"), row=3, col=1)

    surf_name = "Surf (breaking est.)" if fc.spot.is_open_ocean else "Wind chop (est.)"
    fig.add_trace(go.Bar(x=x, y=d["nearshore_wave_ft"], name=surf_name, marker_color="#60a5fa", opacity=0.6,
                         hovertemplate="%{y:.1f} ft"), row=4, col=1)
    if fc.spot.is_open_ocean and "offshore_hs_ft" in d and d["offshore_hs_ft"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=d["offshore_hs_ft"], name="Offshore Hs", line=dict(color="#93c5fd", width=1.4, dash="dot"),
                                 hovertemplate="%{y:.1f} ft offshore"), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["water_temp_f"], name="Water temp", line=dict(color="#fb923c", width=2),
                             hovertemplate="%{y:.1f}°F"), row=4, col=1, secondary_y=True)
    _nights(fig, fc.daily, tz, x.min(), x.max())
    fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1, secondary_y=True)
    fig.update_yaxes(showgrid=False, row=4, col=1, secondary_y=True)
    # panel titles only (paper coordinates) — the blitz labels sit on the time axis
    fig.update_annotations(selector=dict(xref="paper"), font=dict(size=12, color="#94a3b8"), x=0, xanchor="left")
    fig.update_layout(bargap=0)
    fig.update_yaxes(tickformat=".1f", row=4, col=1, secondary_y=True)
    _style(fig, 880)
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10), legend=dict(y=1.075))
    return fig


def species_heatmap(fc, species: list[str], d: pd.DataFrame) -> go.Figure:
    x = d.index.tz_convert(fc.spot.timezone).tz_localize(None)
    z = [d[f"bite_{s}"].to_numpy() for s in species]
    y = [species_meta(s)["short"] for s in species]
    fig = go.Figure(go.Heatmap(x=x, y=y, z=z, zmin=0, zmax=100, colorscale=BITE_SCALE, xgap=1, ygap=3,
                               colorbar=dict(title="Bite", thickness=10, len=0.9),
                               hovertemplate="%{y} · %{x|%a %I %p}: <b>%{z:.0f}</b><extra></extra>"))
    if "blitz_level" in d:
        # Blitz row: only hours that reach watch (amber) or alert (red); everything else stays blank.
        lv = d["blitz_level"].fillna("").astype(str).map({"alert": 2.0, "watch": 1.0, "": np.nan}).to_numpy()
        txt = d["blitz_score"].round().astype("Int64").astype(str).to_numpy()
        fig.add_trace(go.Heatmap(x=x, y=["🔥 Blitz"], z=[lv], zmin=0, zmax=2, showscale=False, xgap=1, ygap=3,
                                 colorscale=[[0, "#334155"], [0.5, "#f59e0b"], [1.0, "#ef4444"]], customdata=[txt],
                                 hovertemplate="Blitz · %{x|%a %I %p}: <b>%{customdata}/100</b><extra></extra>"))
        y = ["🔥 Blitz"] + y
    _style(fig, 90 + 42 * len(y))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(y)))
    return fig


def factor_chart(items) -> go.Figure:
    items = [i for i in items if i.weight > 0]
    items = sorted(items, key=lambda i: i.contribution)
    vals = [i.contribution for i in items]
    labels = [f"<b>{i.factor}</b>  <span style='color:#94a3b8'>{(i.detail[:44] + '…') if len(i.detail) > 45 else i.detail}</span>"
              for i in items]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=["#22c55e" if v > 0.01 else "#ef4444" if v < -0.01 else "#64748b" for v in vals],
        customdata=[i.detail for i in items], hovertemplate="%{customdata}<extra></extra>"))
    fig.update_xaxes(title="← hurts · helps →", zeroline=True, zerolinecolor="#475569", showticklabels=False)
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    _style(fig, 60 + 34 * len(items))
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def season_chart(spot, species: list[str], today_doy: int) -> go.Figure:
    doy = np.arange(1, 366)
    x = pd.Timestamp("2026-01-01") + pd.to_timedelta(doy - 1, unit="D")
    fig = go.Figure()
    for sp in species:
        meta = species_meta(sp)
        fig.add_trace(go.Scatter(x=x, y=seasonal_factor(meta["season"], doy, spot.lat, spot.water_body),
                                 name=meta["short"], line=dict(color=meta["color"], width=2.4, shape="spline")))
    fig.add_vline(x=pd.Timestamp("2026-01-01") + pd.Timedelta(days=today_doy - 1), line=dict(color="#f8fafc", dash="dot"))
    fig.update_yaxes(title="season factor", range=[0, 1.05])
    fig.update_xaxes(tickformat="%b")
    return _style(fig, 320)


def compare_chart(a, b, label_a: str, label_b: str) -> go.Figure:
    da, db = a.display, b.display
    ha = da.index.tz_convert(a.spot.timezone).hour + da.index.tz_convert(a.spot.timezone).minute / 60
    hb = db.index.tz_convert(b.spot.timezone).hour + db.index.tz_convert(b.spot.timezone).minute / 60
    panels = [("wind_mph", "Wind (mph)"), ("surf_ft", "Surf / wind chop (ft, est.)"),
              ("buoy_wave_ft", "Offshore buoy waves (ft, measured)"), ("tide_ft", "Tide (ft MLLW)"),
              ("water_temp_f", "Water temp (°F)"), ("bite_striped_bass", "Striped bass Bite Index")]
    fig = make_subplots(rows=3, cols=2, subplot_titles=[p[1] for p in panels], vertical_spacing=0.11, horizontal_spacing=0.07)
    for i, (col, _) in enumerate(panels):
        r, c = i // 2 + 1, i % 2 + 1
        for df, h, lab, color in ((da, ha, label_a, "#22d3ee"), (db, hb, label_b, "#c084fc")):
            if col in df and df[col].notna().any():
                fig.add_trace(go.Scatter(x=h, y=df[col], name=lab, line=dict(color=color, width=2.2),
                                         showlegend=(i == 0), legendgroup=lab), row=r, col=c)
    fig.update_xaxes(title=None, tickvals=[0, 6, 12, 18, 23], ticktext=["12am", "6am", "noon", "6pm", "11pm"])
    fig.update_annotations(font=dict(size=12, color="#94a3b8"))
    return _style(fig, 720)


def surf_chart(fc, d: pd.DataFrame) -> go.Figure:
    """Surf estimate with its uncertainty band, offshore Hs, NWS zone range and buoy measurements."""
    tz = fc.spot.timezone
    hist = fc.hourly.loc[d.index.min() - pd.Timedelta(hours=24):d.index.max()] if not d.empty else d
    x = hist.index.tz_convert(tz).tz_localize(None)
    fig = go.Figure()
    if "nws_seas_hi_ft" in hist and hist["nws_seas_hi_ft"].notna().any():
        nws_line = dict(width=1.2, shape="hv", color="rgba(250,204,21,0.75)", dash="dot")
        fig.add_trace(go.Scatter(x=x, y=hist["nws_seas_hi_ft"], line=nws_line, showlegend=False,
                                 hovertemplate="NWS up to %{y:.0f} ft<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=hist["nws_seas_lo_ft"], fill="tonexty", fillcolor="rgba(250,204,21,0.08)",
                                 line=nws_line, name=f"NWS {'seas' if fc.spot.is_open_ocean else 'waves'} range ({fc.spot.marine_zone})",
                                 hovertemplate="NWS from %{y:.0f} ft<extra></extra>"))
    if "surf_hi_ft" in hist and hist["surf_hi_ft"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=hist["surf_hi_ft"], line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=hist["surf_lo_ft"], fill="tonexty", fillcolor="rgba(96,165,250,0.22)",
                                 line=dict(width=0), name="Estimate range", hoverinfo="skip"))
    lab = "Surf (breaking)" if fc.spot.is_open_ocean else "Wind chop"
    fig.add_trace(go.Scatter(x=x, y=hist.get("surf_ft"), name=lab, line=dict(color="#60a5fa", width=2.6),
                             hovertemplate="%{y:.1f} ft<extra>" + lab + "</extra>"))
    if fc.spot.is_open_ocean and "offshore_hs_ft" in hist:
        fig.add_trace(go.Scatter(x=x, y=hist["offshore_hs_ft"], name="Offshore Hs (model, buoy-corrected)",
                                 line=dict(color="#c4b5fd", width=1.6, dash="dot"), hovertemplate="%{y:.1f} ft<extra>offshore</extra>"))
    if "buoy_wave_ft" in hist and hist["buoy_wave_ft"].notna().any():
        src = next((s for s in fc.sources if s.feature == "offshore waves (measured)"), None)
        blab = (f"Buoy {src.station_id} Hs (measured, {src.distance_km:.0f} km away)"
                if src and src.distance_km is not None else "Buoy Hs (measured)")
        fig.add_trace(go.Scatter(x=x, y=hist["buoy_wave_ft"], mode="markers", name=blab,
                                 marker=dict(color="#86efac", size=5), hovertemplate="%{y:.1f} ft<extra>buoy</extra>"))
    now = pd.Timestamp.now(tz=tz).tz_localize(None)
    if len(x) and x.min() <= now <= x.max():
        fig.add_vline(x=now, line=dict(color="#f8fafc", dash="dot", width=1))
    for i, dd in enumerate(fc.daily):
        if dd.get("sunset") is None or not len(x):
            continue
        a = pd.Timestamp(dd["sunset"]).tz_convert(tz).tz_localize(None)
        nxt = fc.daily[i + 1].get("sunrise") if i + 1 < len(fc.daily) else None
        b = pd.Timestamp(nxt).tz_convert(tz).tz_localize(None) if nxt else a + pd.Timedelta(hours=11)
        a, b = max(a, x.min()), min(b, x.max())
        if a < b:
            fig.add_vrect(x0=a, x1=b, fillcolor=NIGHT, line_width=0, layer="below")
    fig.update_yaxes(title="ft", rangemode="tozero")
    return _style(fig, 330)
