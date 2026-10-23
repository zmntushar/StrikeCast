"""Folium map: spots colored by Bite Index, bathymetry structure and cast targets."""
from __future__ import annotations

import folium
from branca.element import Element, MacroElement, Template

from app.models.bite import LEGEND, score_label
from app.models.common import species_meta
from dashboard.components.ui import esc, fmt, tide_text, wind_text, window_text

STRUCT_COLORS = {"drop-off": "#f97316", "deep channel": "#3b82f6", "trough": "#06b6d4", "bar / shoal": "#eab308"}

NOAA_ENC = ("https://gis.charttools.noaa.gov/arcgis/rest/services/MCS/NOAAChartDisplay/MapServer/"
            "exts/MaritimeChartService/WMSServer")


def _base_layers(m: folium.Map) -> None:
    folium.TileLayer("https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
                     attr="USGS The National Map (public domain)", name="Satellite · USGS", max_zoom=19,
                     max_native_zoom=16, show=True).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri, Maxar, Earthstar Geographics", name="Satellite · Esri (sharper)", max_zoom=19, show=False).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri, GEBCO, NOAA, NGS", name="Ocean depth shading", max_zoom=19, max_native_zoom=13, show=False).add_to(m)
    folium.TileLayer("cartodbdark_matter", name="Dark", attr="© OpenStreetMap contributors © CARTO", show=False).add_to(m)
    folium.TileLayer("openstreetmap", name="Streets · OSM", show=False).add_to(m)
    folium.WmsTileLayer(NOAA_ENC, layers="0,1,2,3,4,5,6,7", fmt="image/png", transparent=True, version="1.3.0",
                        name="NOAA nautical chart (depths, buoys)", attr="NOAA Office of Coast Survey",
                        overlay=True, show=False, opacity=0.85).add_to(m)
    folium.TileLayer("https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png", attr="© OpenSeaMap contributors",
                     name="Sea marks · OpenSeaMap", overlay=True, show=False).add_to(m)


def _legend(species_label: str) -> MacroElement:
    rows = "".join(f'<div><span style="background:{c}"></span>{lab}{" " + str(lo) + "+" if lo else " <35"}</div>'
                   for lo, lab, c in LEGEND)
    srows = "".join(f'<div><span style="background:{c};border-radius:2px"></span>{k}</div>' for k, c in STRUCT_COLORS.items())
    html = f"""
{{% macro html(this, kwargs) %}}
<div style="position:absolute; bottom:24px; left:10px; z-index:9999; background:rgba(7,13,26,.86); color:#e2e8f0;
     border:1px solid rgba(148,163,184,.25); border-radius:10px; padding:7px 9px; font:11px Inter,system-ui,sans-serif;
     box-shadow:0 6px 18px rgba(0,0,0,.35); max-width:190px; line-height:1.35">
  <div style="font-weight:700; margin-bottom:4px;">Bite Index · {esc(species_label)}</div>
  <style>.lg div{{display:inline-block;margin-right:8px}} .lg span{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}}</style>
  <div class="lg">{rows}</div>
  <div style="font-weight:700; margin:6px 0 4px;">Structure (selected spot)</div>
  <div class="lg">{srows}<div><span style="background:#22d3ee"></span>Cast target #</div></div>
  <div style="color:#94a3b8; margin-top:5px">0–100 relative suitability, not a probability. ★ = favorite.</div>
</div>
{{% endmacro %}}"""
    el = MacroElement()
    el._template = Template(html)
    return el


def _popup(fc, species: list[str], focus: str) -> str:
    row = fc.now_row()
    chips = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:8px"><span>{esc(species_meta(s)["short"])}</span>'
        f'<b style="color:{score_label(fc.best_today(s))[1]}">{fmt(fc.bite_now(s))} now · {fmt(fc.best_today(s))} today</b></div>'
        for s in species)
    w = fc.windows.get(focus) or []
    win = f"<div style='margin-top:6px'>Next prime window ({esc(species_meta(focus)['short'])}): <b>{window_text(w[0], fc.spot.timezone)}</b> · peak {w[0].peak_score:.0f}</div>" if w else ""
    return f"""<div style="font:12px Inter,system-ui,sans-serif;min-width:230px">
      <div style="font-weight:700;font-size:14px">{'★ ' if fc.spot.favorite else ''}{esc(fc.spot.name)}</div>
      <div style="color:#64748b;margin-bottom:6px">{esc(fc.spot.environment.replace('_',' '))} · {esc(fc.spot.water_body)} · data {fc.quality['grade']}</div>
      {chips}
      <hr style="margin:6px 0;border:none;border-top:1px solid #e2e8f0">
      <div>💧 {fmt(row.get('water_temp_f'), '{:.1f}')}°F · 💨 {esc(wind_text(row))}</div>
      <div>🌊 ~{fmt(row.get('nearshore_wave_ft'), '{:.1f}')} ft surf · {esc(tide_text(row))}</div>
      {win}
    </div>"""


def build_map(forecasts: dict, species: list[str], focus: str, selected_id: str | None, center: tuple[float, float],
              zoom: int = 10, custom_center: tuple[float, float] | None = None, radius_mi: float | None = None) -> folium.Map:
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=True)
    _base_layers(m)
    spot_group = folium.FeatureGroup(name="Spots (Bite Index)", show=True)
    struct_group = folium.FeatureGroup(name="Structure & cast targets", show=True)

    if custom_center and radius_mi:
        folium.Circle(custom_center, radius=radius_mi * 1609.34, color="#22d3ee", weight=1, fill=False,
                      dash_array="6 6", tooltip=f"Search radius {radius_mi:.0f} mi").add_to(m)
        folium.Marker(custom_center, tooltip="Search center",
                      icon=folium.Icon(color="lightblue", icon="crosshairs", prefix="fa")).add_to(m)

    for sid, fc in forecasts.items():
        if focus == "best":
            val = max((fc.best_today(s) for s in species), default=float("nan"))
        else:
            val = fc.best_today(focus)
        label, color = score_label(val)
        is_sel = sid == selected_id
        size = 38 if fc.spot.favorite else 32
        ring = "#ffffff" if is_sel else "rgba(255,255,255,.55)"
        star = "★" if fc.spot.favorite else ""
        icon_html = f"""
        <div style="position:relative;width:{size}px;height:{size}px">
          <div style="width:{size}px;height:{size}px;box-sizing:border-box;border-radius:50%;background:{color};border:3px solid {ring};
               box-shadow:0 0 0 4px {color}55, 0 4px 14px rgba(0,0,0,.45);display:grid;place-items:center;
               font:700 13px 'Space Grotesk',Inter,sans-serif;color:#0b1220">{fmt(val)}</div>
          <span style="position:absolute;top:-11px;right:-9px;font-size:15px;color:#fde047;text-shadow:0 1px 2px #000">{star}</span>
          <div style="position:absolute;top:{size + 3}px;left:50%;transform:translateX(-50%);white-space:nowrap;
               font:600 11px Inter,sans-serif;color:#f8fafc;text-shadow:0 1px 3px #000,0 0 6px #000">{esc(fc.spot.short)}</div>
        </div>"""
        folium.Marker(
            [fc.spot.lat, fc.spot.lon],
            icon=folium.DivIcon(html=icon_html, icon_size=(size, size), icon_anchor=(size // 2, size // 2)),
            tooltip=f"{fc.spot.name} — {label} ({fmt(val)})",
            popup=folium.Popup(_popup(fc, species, species[0] if focus == "best" else focus), max_width=320),
        ).add_to(spot_group)

    sel = forecasts.get(selected_id) if selected_id else None
    if sel and sel.bathy and sel.bathy.ok:
        for c in sel.bathy.cells:
            folium.CircleMarker([c["lat"], c["lon"]], radius=3.2, weight=0, fill=True, fill_opacity=0.75,
                                fill_color=STRUCT_COLORS.get(c["kind"], "#94a3b8"),
                                tooltip=f"{c['kind']} · {c['depth_m']} m deep · slope {c['slope_pct']}%").add_to(struct_group)
        for i, t in enumerate(sel.bathy.targets, 1):
            folium.Marker([t["lat"], t["lon"]], tooltip=f"Cast target #{i}: {t['kind']}",
                          popup=folium.Popup(f"<b>Cast target #{i}</b><br>{esc(t['kind'])}<br>{t['depth_m']} m deep "
                                             f"(NAVD88) · slope {t['slope_pct']}%<br>~{t['cast_m']} m from shore", max_width=220),
                          icon=folium.DivIcon(html=f'<div style="width:24px;height:24px;border-radius:50%;background:#22d3ee;'
                                                   f'border:2px solid #fff;display:grid;place-items:center;font:700 12px Inter;'
                                                   f'color:#082f49;box-shadow:0 2px 8px rgba(0,0,0,.5)">{i}</div>',
                                              icon_size=(24, 24), icon_anchor=(12, 12))).add_to(struct_group)
    struct_group.add_to(m)
    spot_group.add_to(m)
    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    m.get_root().add_child(_legend("best of selected species" if focus == "best" else species_meta(focus)["name"]))
    m.get_root().header.add_child(Element("<style>.leaflet-popup-content-wrapper{border-radius:12px}</style>"))
    return m
