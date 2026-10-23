"""StrikeCast dashboard — run with:  streamlit run dashboard/streamlit_app.py"""
from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

st.set_page_config(page_title="StrikeCast · Long Island Fishing Forecast", page_icon="🎣", layout="wide",
                   initial_sidebar_state="expanded")

from app.core.config import regulations_config  # noqa: E402
from app.core.geo import compass, haversine_km, km_to_mi  # noqa: E402
from app.core.http import HttpClient  # noqa: E402
from app.ml.train import model_exists, train_species  # noqa: E402
from app.models.bite import explain_hour, narrative, score_label, season_status, tactic  # noqa: E402
from app.models.common import species_list, species_meta  # noqa: E402
from app.providers.geocode import geocode_name, geocode_zip  # noqa: E402
from app.services.compare import compare_dates  # noqa: E402
from app.services.forecast import (collect, feature_snapshot, forecast_frame, get_spot,  # noqa: E402
                                   historical_frame, load_spots, run_sync)
from app.services.observations import (add_observation, delete_observation, export_csv,  # noqa: E402
                                       import_csv, list_observations)
from app.services.stations import build_custom_spot  # noqa: E402
from dashboard.components import charts  # noqa: E402
from dashboard.components.maps import build_map  # noqa: E402
from dashboard.components.styles import CSS  # noqa: E402
from app.models.tackle import recommend  # noqa: E402
from dashboard.components.ui import (chip, esc, fmt, hour_label, isnum, kind_badge, legend_html,  # noqa: E402
                                     species_popover, tide_text, wind_text, window_text)

st.markdown(CSS, unsafe_allow_html=True)

SPECIES = species_list()
ENVS = ["beach", "inlet", "jetty", "river_mouth", "bay", "pier"]
ALL_SPOTS = load_spots()
FAVS = [s for s in ALL_SPOTS if s.favorite]
TZ = ALL_SPOTS[0].timezone
CACHE_MIN = 20
LOGO = ('<svg width="30" height="30" viewBox="0 0 64 64" fill="none"><path d="M6 34c8-12 22-18 34-14 6 2 10 6 14 10-4 4-8 8-14 10'
        '-12 4-26-2-34-6z" fill="#f8fafc"/><path d="M54 30l8-8v24l-8-8" fill="#e0f2fe"/><circle cx="18" cy="30" r="3" fill="#0e7490"/>'
        '<path d="M26 26c2 4 2 8 0 12M33 25c2 5 2 10 0 15" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round"/></svg>')
ss = st.session_state


# =============================================================== data access
def fetch_forecasts(spots, force: bool, slot) -> dict:
    key = tuple(sorted(s.id for s in spots))
    hit = ss.get("fc")
    if hit and not force and hit["key"] == key and time.time() - hit["t"] < CACHE_MIN * 60:
        return hit["data"]
    with slot.container():
        st.markdown('<div class="sc-card"><b>Collecting live data</b> <span class="sc-sub">· NOAA tides, currents, buoys, '
                    'NWS, Open-Meteo, bathymetry (cached locally after the first run)</span></div>', unsafe_allow_html=True)
        bar = st.progress(0.0, text="Starting…")

        def cb(p: float, label: str) -> None:
            bar.progress(min(max(p, 0.0), 1.0), text=f"{label} · {p * 100:.0f}%")

        data = run_sync(collect(spots, forecast_frame(7), cb))
        bar.progress(1.0, text="Done")
    slot.empty()
    ss["fc"] = {"key": key, "t": time.time(), "data": data, "at": datetime.now(timezone.utc)}
    return data


def custom_spot(lat: float, lon: float, env: str, label: str):
    k = ("custom", round(lat, 4), round(lon, 4), env)
    if k not in ss:
        async def _mk():
            async with HttpClient() as http:
                return await build_custom_spot(http, lat, lon, env, label)
        with st.spinner("Finding the nearest NOAA stations and shoreline orientation…"):
            ss[k] = run_sync(_mk())
    return ss[k]


async def _geo(q: str, kind: str):
    async with HttpClient() as http:
        return [await geocode_zip(http, q)] if kind == "zip" else await geocode_name(http, q)


# =================================================================== sidebar
with st.sidebar:
    st.markdown("### 📍 Location")
    mode = st.radio("Find spots by", ["My spots", "Named spot", "ZIP code", "Coordinates", "Search a place"],
                    label_visibility="collapsed")
    custom = None
    custom_notes: list[str] = []
    search_center = None  # (lat, lon, label) for ZIP searches
    focus_id = FAVS[0].id
    if mode == "My spots":
        names = {s.name: s.id for s in FAVS}
        focus_id = names[st.selectbox("Focus spot", list(names), index=4 if len(names) > 4 else 0)]
    elif mode == "Named spot":
        names = {s.name: s.id for s in ALL_SPOTS}
        focus_id = names[st.selectbox("Spot", list(names))]
    else:
        env = (st.selectbox("Fishing environment here", ENVS, format_func=lambda e: e.replace("_", " ").title())
               if mode != "ZIP code" else None)
        loc = None
        if mode == "ZIP code":
            z = st.text_input("ZIP code", "11702")
            if z and len(z.strip()) == 5:
                try:
                    loc = run_sync(_geo(z.strip(), "zip"))[0]
                except Exception as exc:  # noqa: BLE001
                    st.error(f"ZIP lookup failed: {exc}")
        elif mode == "Coordinates":
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", value=40.6210, format="%.4f")
            lon = c2.number_input("Longitude", value=-73.3050, format="%.4f")
            loc = {"lat": lat, "lon": lon, "label": f"{lat:.4f}, {lon:.4f}"}
        else:
            q = st.text_input("Place name", "Cupsogue Beach, NY")
            if q:
                try:
                    res = run_sync(_geo(q, "name"))
                    if res:
                        pick = st.selectbox("Matches", range(len(res)), format_func=lambda i: res[i]["label"][:60])
                        loc = res[pick]
                    else:
                        st.warning("No matches.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Search failed: {exc}")
        if loc and mode == "ZIP code":
            # A ZIP is a search center: rank the configured spots around it.
            search_center = (loc["lat"], loc["lon"], loc["label"])
            focus_id = min(ALL_SPOTS, key=lambda s: haversine_km(loc["lat"], loc["lon"], s.lat, s.lon)).id
            st.caption(f"📌 {loc['label']} — showing spots within the radius below")
        elif loc:
            custom, custom_notes = custom_spot(loc["lat"], loc["lon"], env, loc["label"].split(",")[0][:28])
            focus_id = custom.id
            st.caption(f"📌 {loc['label']}")
    radius = st.slider("Search radius (miles)", 5, 50, 50)
    envs = st.multiselect("Fishing environments", ENVS, default=ENVS, format_func=lambda e: e.replace("_", " ").title())

    st.markdown("### 🐟 Species")
    species = st.pills("Target species", SPECIES, selection_mode="multi", default=SPECIES,
                       format_func=lambda s: species_meta(s)["short"], label_visibility="collapsed") or SPECIES[:1]
    st.markdown("### 🗓️ Forecast window")
    horizon = st.segmented_control("Horizon", ["24 hours", "3 days", "7 days"], default="3 days") or "3 days"
    today_local = pd.Timestamp.now(tz=TZ).date()
    start_day = st.date_input("Starting", value=today_local, min_value=today_local, max_value=today_local + timedelta(days=6))
    tod = st.slider("Time of day for best windows", 0, 24, (0, 24), format="%d:00")
    st.toggle("Shore-based fishing", value=True, disabled=True, help="Boat-based mode is on the roadmap.")
    refresh = st.button("🔄 Refresh live data", use_container_width=True, type="primary")
    st.caption("Data is cached locally per provider update cycle; refresh pulls anything that has expired.")

# ================================================================== data
spots_to_fetch = ALL_SPOTS + ([custom] if custom else [])
focus_spot = custom or get_spot(focus_id)
center = (search_center[0], search_center[1]) if search_center else (focus_spot.lat, focus_spot.lon)

progress_slot = st.empty()
header_slot = st.empty()
try:
    forecasts = fetch_forecasts(spots_to_fetch, refresh, progress_slot)
except Exception as exc:  # noqa: BLE001
    st.error(f"Data collection failed: {exc}")
    st.stop()


def visible(fc) -> bool:
    if fc.spot.favorite or fc.spot.custom or fc.spot.id == focus_id:
        return True
    in_radius = km_to_mi(haversine_km(center[0], center[1], fc.spot.lat, fc.spot.lon)) <= radius
    return in_radius and fc.spot.environment in envs


VIS = {k: v for k, v in forecasts.items() if visible(v)}
hours = {"24 hours": 24, "3 days": 72, "7 days": 168}[horizon]
start_ts = max(pd.Timestamp(start_day).tz_localize(TZ).tz_convert("UTC"), pd.Timestamp.now(tz="UTC").floor("h"))
end_ts = start_ts + pd.Timedelta(hours=hours)


def view(fc) -> pd.DataFrame:
    return fc.hourly.loc[start_ts:end_ts]


def in_tod(ts) -> bool:
    h = pd.Timestamp(ts).tz_convert(TZ).hour
    return tod[0] <= h < tod[1] if tod != (0, 24) else True


def windows(fc, sp):
    return [w for w in fc.windows.get(sp, []) if start_ts <= pd.Timestamp(w.peak) <= end_ts and in_tod(w.peak)]


fetched_at = ss["fc"]["at"]
qual = [fc.quality["score"] for fc in VIS.values()]
alerts: dict[str, list] = {}
for _fc in VIS.values():
    for a in _fc.alerts:
        alerts.setdefault(a.event, [])
        if a.headline not in [x.headline for x in alerts[a.event]]:
            alerts[a.event].append(a)
header_slot.markdown(f"""
<div class="sc-header">
  <div class="sc-brand"><div class="sc-logo">{LOGO}</div>
    <div><div class="sc-name">Strike<span>Cast</span></div>
    <div class="sc-tag">Long Island shore-fishing intelligence · striped bass · bluefish · fluke · blackfish · porgy · 100% free & open data</div></div>
  </div>
  <div class="sc-meta">
    <span class="sc-pill">🕒 <b>{pd.Timestamp.now(tz=TZ):%a %b %d, %I:%M %p}</b></span>
    <span class="sc-pill">📡 data pulled <b>{pd.Timestamp(fetched_at).tz_convert(TZ):%I:%M %p}</b></span>
    <span class="sc-pill">✅ data quality <b>{int(np.mean(qual)) if qual else 0}/100</b></span>
    <span class="sc-pill">📍 <b>{len(VIS)}</b> spots in view</span>
  </div>
</div>""", unsafe_allow_html=True)

for ev, items in list(alerts.items())[:3]:
    last = max(items, key=lambda a: a.ends or datetime.min.replace(tzinfo=timezone.utc))
    until = f" until {hour_label(pd.Timestamp(last.ends).tz_convert(TZ))}" if last.ends else ""
    zones = f" · {len(items)} zones" if len(items) > 1 else ""
    st.markdown(f'<div class="sc-alert">⚠️ <b>{esc(ev)}</b>{esc(until)}{esc(zones)} — NWS Upton. '
                f'<span style="opacity:.8">{esc(last.headline)}</span></div>', unsafe_allow_html=True)

tabs = st.tabs(["🎯 Today", "🗺️ Map", "📈 7-Day Forecast", "🌊 Conditions & Data", "🔁 Compare Dates",
                "📓 Fishing Log", "📘 Species & Regs"])


# ================================================================ helpers
def best_bet(fcs: dict, sps: list[str], hours_ahead: int = 24):
    best = None
    t1 = start_ts + pd.Timedelta(hours=hours_ahead)
    for fc in fcs.values():
        d = fc.hourly.loc[start_ts:t1]
        if tod != (0, 24):
            d = d[[in_tod(t) for t in d.index]]
        for sp in sps:
            if d.empty or f"bite_{sp}" not in d:
                continue
            s = d[f"bite_{sp}"]
            if s.notna().any() and (best is None or s.max() > best[3]):
                best = (fc, sp, s.idxmax(), float(s.max()))
    return best


def spot_card(fc) -> str:
    row = fc.hourly.loc[start_ts:].iloc[0] if not fc.hourly.loc[start_ts:].empty else fc.now_row()
    d = view(fc)
    depth = fc.bathy.max_depth_cast_m if fc.bathy and fc.bathy.ok else None
    chip_html = []
    for sp in species:
        ws_sp = windows(fc, sp)
        s_sp = d[f"bite_{sp}"].dropna() if not d.empty else pd.Series(dtype=float)
        peak_ts = pd.Timestamp(ws_sp[0].peak) if ws_sp else (s_sp.idxmax() if not s_sp.empty else None)
        prow = fc.hourly.loc[peak_ts] if peak_ts is not None else row
        reg = season_status(sp, (peak_ts if peak_ts is not None else start_ts).tz_convert(fc.spot.timezone).date())
        rec = recommend(sp, prow, fc.spot.environment, depth)
        why = narrative(prow, sp, fc.spot) if peak_ts is not None else ""
        pop = species_popover(fc, sp, d, ws_sp, reg, rec, why)
        peak_txt = f"· peak {fmt(s_sp.max() if not s_sp.empty else None)}"
        chip_html.append(chip(sp, float(row.get(f"bite_{sp}", np.nan)), peak_txt, pop=pop))
    chips = "".join(chip_html)
    top_sp = max(species, key=lambda s: d[f"bite_{s}"].max() if not d.empty else -1)
    ws = windows(fc, top_sp)
    win = (f'<div class="sc-window">⏱ Best {esc(species_meta(top_sp)["short"])} window: <b>{window_text(ws[0], fc.spot.timezone)}</b>'
           f' · peak {ws[0].peak_score:.0f}</div>') if ws else '<div class="sc-window">No standout window in this range.</div>'
    g = fc.quality["grade"]
    gcol = {"A": "#22c55e", "B": "#84cc16", "C": "#f59e0b", "D": "#ef4444"}[g]
    return f"""<div class="sc-card sc-spot">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:.5rem">
        <div><h4>{'★ ' if fc.spot.favorite else ''}{esc(fc.spot.name)}</h4>
        <div class="sc-sub">{esc(fc.spot.region)} · {esc(fc.spot.environment.replace('_', ' '))}</div></div>
        <span class="sc-badge" style="background:{gcol}22;color:{gcol};border:1px solid {gcol}66">data {g}</span>
      </div>
      <div class="sc-chips">{chips}</div>
      <div class="sc-sub">Bite Index now · peak in selected window · hover a fish for details &amp; tackle</div>
      <div class="sc-cond">
        <div><span>Water</span> {fmt(row.get('water_temp_f'), '{:.1f}')}°F</div>
        <div><span>Surf</span> ~{fmt(row.get('nearshore_wave_ft'), '{:.1f}')} ft</div>
        <div><span>Wind</span> {esc(wind_text(row))}</div>
        <div><span>Tide</span> {esc(tide_text(row))}</div>
      </div>{win}
    </div>"""


# ==================================================================== TODAY
with tabs[0]:
    bb = best_bet(VIS, species, min(hours, 24))
    if bb:
        fc, sp, ts, val = bb
        lab, color = score_label(val)
        row = fc.hourly.loc[ts]
        ws = [w for w in fc.windows.get(sp, []) if pd.Timestamp(w.start) <= ts < pd.Timestamp(w.end)]
        wtxt = window_text(ws[0], fc.spot.timezone) if ws else hour_label(ts.tz_convert(fc.spot.timezone))
        reg = season_status(sp, ts.tz_convert(fc.spot.timezone).date())
        reg_html = ('<span class="sc-badge" style="background:#14532d;color:#bbf7d0">Season open</span>' if reg.get("open")
                    else '<span class="sc-badge" style="background:#7f1d1d;color:#fecaca">Season closed — C&R only</span>')
        c1, c2 = st.columns([1.35, 1])
        with c1:
            st.markdown(f"""<div class="sc-card sc-hero">
              <div class="sc-sub" style="text-transform:uppercase;letter-spacing:.08em">Best bet · next {min(hours, 24)} h</div>
              <div style="display:flex;align-items:center;gap:1rem;margin:.4rem 0 .6rem">
                <div class="big" style="color:{color}">{val:.0f}</div>
                <div><div style="font-size:1.25rem;font-weight:700;color:#f8fafc">{esc(species_meta(sp)['name'])}</div>
                <div style="color:#cbd5e1">at <b>{esc(fc.spot.name)}</b> · {esc(wtxt)} · <b style="color:{color}">{lab}</b> {reg_html}</div></div>
              </div>
              <div class="sc-explain">{esc(narrative(row, sp, fc.spot))}</div>
              <div class="sc-window">🎯 Try: {esc(tactic(row, sp))}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.plotly_chart(charts.factor_chart(explain_hour(row, sp, fc.spot)), use_container_width=True,
                            config={"displayModeBar": False})
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.markdown("#### ★ Your spots")
    fav_fcs = [VIS[s.id] for s in FAVS if s.id in VIS]
    for i in range(0, len(fav_fcs), 3):
        cols = st.columns(3)
        for col, fc in zip(cols, fav_fcs[i:i + 3]):
            col.markdown(spot_card(fc), unsafe_allow_html=True)
    if custom and custom.id in VIS:
        st.markdown("#### 📌 Your searched location")
        st.markdown(spot_card(VIS[custom.id]), unsafe_allow_html=True)
        for n in custom_notes:
            st.caption(n)

    if search_center or custom:
        label = search_center[2] if search_center else custom.name
        st.markdown(f"#### 📍 Best spots near {esc(label)} · within {radius} mi")
        near = []
        for fc in VIS.values():
            dist = km_to_mi(haversine_km(center[0], center[1], fc.spot.lat, fc.spot.lon))
            if dist > radius:
                continue
            d = view(fc)
            best_sp = max(species, key=lambda s: d[f"bite_{s}"].max() if not d.empty else -1)
            ws = windows(fc, best_sp)
            near.append({"Spot": ("★ " if fc.spot.favorite else "") + fc.spot.name, "Distance (mi)": round(dist, 1),
                         "Type": fc.spot.environment.replace("_", " "), "Best species": species_meta(best_sp)["short"],
                         "Peak": d[f"bite_{best_sp}"].max() if not d.empty else np.nan,
                         "Best window": window_text(ws[0], fc.spot.timezone) if ws else "—"})
        if near:
            st.dataframe(pd.DataFrame(near).sort_values("Peak", ascending=False), hide_index=True, use_container_width=True,
                         column_config={"Peak": st.column_config.ProgressColumn("Peak", min_value=0, max_value=100, format="%.0f")})
        else:
            st.info("No configured spots inside that radius — widen it, or use Coordinates to forecast an exact spot.")
    st.markdown(f"#### 🏆 Top windows · {horizon} from {start_day:%a %b %d}")
    rows = []
    for fc in VIS.values():
        for sp in species:
            for w in windows(fc, sp)[:2]:
                rows.append({"Species": species_meta(sp)["short"], "Spot": ("★ " if fc.spot.favorite else "") + fc.spot.name,
                             "Window": window_text(w, fc.spot.timezone), "Peak": w.peak_score, "Avg": w.mean_score,
                             "Tide at peak": fc.hourly.loc[pd.Timestamp(w.peak)].get("tide_stage") or "—",
                             "Light": fc.hourly.loc[pd.Timestamp(w.peak)].get("light_phase")})
    if rows:
        tdf = pd.DataFrame(rows).sort_values("Peak", ascending=False).head(25)
        st.dataframe(tdf, hide_index=True, use_container_width=True,
                     column_config={"Peak": st.column_config.ProgressColumn("Peak", min_value=0, max_value=100, format="%.0f"),
                                    "Avg": st.column_config.NumberColumn(format="%.0f")})
    else:
        st.info("No standout windows for the current filters — try widening the time-of-day range or horizon.")

# ====================================================================== MAP
with tabs[1]:
    c_map, c_side = st.columns([2.6, 1])
    with c_side:
        focus_opts = ["best"] + species
        focus_species = st.selectbox("Color spots by", focus_opts, format_func=lambda s: "Best of selected species" if s == "best" else species_meta(s)["name"])
        names = {fc.spot.name: sid for sid, fc in VIS.items()}
        default_sel = ss.get("map_sel", focus_id if focus_id in VIS else next(iter(VIS)))
        sel_name = st.selectbox("Show structure for", list(names), index=list(names.values()).index(default_sel) if default_sel in names.values() else 0)
        sel_id = names[sel_name]
        zoom_in = st.toggle("Zoom to this spot's structure", value=False,
                            help="Shows drop-offs, troughs, channels and numbered cast targets from NOAA bathymetry.")
    with c_map:
        m = build_map(VIS, species, focus_species, sel_id, (VIS[sel_id].spot.lat, VIS[sel_id].spot.lon),
                      zoom=16 if zoom_in else 10,
                      custom_center=center if (custom or search_center) else None,
                      radius_mi=radius if (custom or search_center) else None)
        out = st_folium(m, height=660, use_container_width=True, returned_objects=["last_object_clicked_tooltip"], key="map")
        tip = (out or {}).get("last_object_clicked_tooltip")
        if tip:
            for sid, fc in VIS.items():
                if tip.startswith(fc.spot.name) and ss.get("map_sel") != sid:
                    ss["map_sel"] = sid
                    st.rerun()
    with c_side:
        fc = VIS[sel_id]
        b = fc.bathy
        st.markdown(f"""<div class="sc-card"><h4>{'★ ' if fc.spot.favorite else ''}{esc(fc.spot.name)}</h4>
          <div class="sc-sub">{esc(fc.spot.access)}</div>
          <div class="sc-explain" style="margin-top:.5rem;font-size:.85rem">{esc(fc.spot.notes)}</div></div>""",
                    unsafe_allow_html=True)
        if b and b.ok:
            targets = "".join(f'<div class="sc-target"><div class="num">{i}</div><div><b>{esc(t["kind"].title())}</b> · '
                              f'{t["depth_m"]} m deep · slope {t["slope_pct"]}%<br><span class="sc-sub">~{t["cast_m"]} m cast from the waterline</span></div></div>'
                              for i, t in enumerate(b.targets, 1))
            st.markdown(f"""<div class="sc-card" style="margin-top:.7rem"><h4>Where to cast</h4>
              <div class="sc-sub">From NOAA {b.resolution_m} m bathymetry (NAVD88). Sandbars shift after storms — read the water.</div>
              {targets}
              <div class="sc-cond" style="margin-top:.5rem">
                <div><span>Max depth in reach</span> {fmt(b.max_depth_cast_m, '{:.1f}')} m</div>
                <div><span>Steepest slope</span> {fmt(b.max_slope_pct, '{:.0f}')}%</div>
                <div><span>3 m depth</span> {fmt(b.dist_to_3m_m)} m out</div>
                <div><span>Nearest inlet</span> {esc((b.nearest_inlet or {}).get('name', '—'))} ({fmt(km_to_mi((b.nearest_inlet or {}).get('distance_km', np.nan)), '{:.1f}')} mi)</div>
              </div></div>""", unsafe_allow_html=True)
        else:
            st.info("Bathymetry not available for this spot.")
        st.caption("Tip: switch layers (top-right of the map) to NOAA nautical chart or OpenSeaMap sea marks. Click a spot for its forecast.")

# ================================================================= FORECAST
with tabs[2]:
    names = {fc.spot.name: sid for sid, fc in VIS.items()}
    default = ss.get("map_sel", focus_id)
    fsel = names[st.selectbox("Spot", list(names), index=list(names.values()).index(default) if default in names.values() else 0, key="fsel")]
    fc = VIS[fsel]
    d = view(fc)
    if d.empty:
        st.warning("No forecast hours in the selected range.")
    else:
        row = d.iloc[0]
        k = st.columns(6)
        best_sp = max(species, key=lambda s: row.get(f"bite_{s}", 0))
        kpis = [("Baitfish index", row.get("bait_index"), f"top: {str(row.get('bait_top', '')).replace('_', ' ')}"),
                ("Bass habitat", row.get("presence_striped_bass"), "striped bass HSI"),
                ("Bass feeding", row.get("feeding_striped_bass"), "activity index"),
                (f"Bite · {species_meta(best_sp)['short']}", row.get(f"bite_{best_sp}"), score_label(row.get(f"bite_{best_sp}"))[0]),
                ("Data quality", fc.quality["score"], f"grade {fc.quality['grade']}"),
                ("Water temp", row.get("water_temp_f"), str(row.get("water_temp_kind")))]
        for col, (lab, v, note) in zip(k, kpis):
            col.markdown(f'<div class="sc-card sc-kpi"><div class="lab">{esc(lab)}</div><div class="val">{fmt(v, "{:.1f}" if lab == "Water temp" else "{:.0f}")}{"°F" if lab == "Water temp" else ""}</div>'
                         f'<div class="note">{esc(note)}</div></div>', unsafe_allow_html=True)
        st.caption("Indices are environmental suitability scores (0–100). Shaded columns = night; cyan bands = best windows for "
                   f"{species_meta(species[0])['short']}.")
        st.plotly_chart(charts.forecast_timeline(fc, species, d), use_container_width=True, config={"displayModeBar": False})
        st.markdown("##### Species × hour")
        st.plotly_chart(charts.species_heatmap(fc, species, d), use_container_width=True, config={"displayModeBar": False})

        st.markdown("##### 🔎 Why this score? Explain any hour")
        c1, c2 = st.columns([1, 1.3])
        with c1:
            opts = list(d.index)
            pick = st.select_slider("Hour", options=opts, value=opts[0],
                                    format_func=lambda t: hour_label(pd.Timestamp(t).tz_convert(fc.spot.timezone)))
            esp = st.selectbox("Species", species, format_func=lambda s: species_meta(s)["name"], key="esp")
            r = fc.hourly.loc[pick]
            v = r.get(f"bite_{esp}")
            lab, colr = score_label(v)
            st.markdown(f"""<div class="sc-card"><div class="sc-sub">{hour_label(pd.Timestamp(pick).tz_convert(fc.spot.timezone))}</div>
              <div style="display:flex;gap:.8rem;align-items:baseline"><div class="sc-kpi"><div class="val" style="color:{colr}">{fmt(v)}</div></div>
              <div><b style="color:{colr}">{lab}</b> · habitat {fmt(r.get(f'presence_{esp}'))} · feeding {fmt(r.get(f'feeding_{esp}'))} · fishability {fmt((r.get(f'fishability_{esp}') or 0) * 100)}%</div></div>
              <div class="sc-explain" style="margin-top:.4rem">{esc(narrative(r, esp, fc.spot))}</div>
              <div class="sc-window">🎯 {esc(tactic(r, esp))}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.plotly_chart(charts.factor_chart(explain_hour(r, esp, fc.spot)), use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown("##### 🗓️ Day by day")
        days = []
        for dd in fc.daily:
            lday = pd.Timestamp(dd["date"]).tz_localize(fc.spot.timezone)
            seg = fc.hourly.loc[lday.tz_convert("UTC"):(lday + pd.Timedelta(days=1)).tz_convert("UTC") - pd.Timedelta(hours=1)]
            if seg.empty:
                continue
            tides_txt = " · ".join(f"{'H' if e.type == 'H' else 'L'} {hour_label(pd.Timestamp(e.time).tz_convert(fc.spot.timezone), False)}"
                                   for e in fc.tide_events if pd.Timestamp(e.time).tz_convert(fc.spot.timezone).date() == dd["date"])
            rec = {"Day": f"{dd['date']:%a %b %d}",
                   "Sunrise": hour_label(pd.Timestamp(dd["sunrise"]).tz_convert(fc.spot.timezone), False) if dd.get("sunrise") else "—",
                   "Sunset": hour_label(pd.Timestamp(dd["sunset"]).tz_convert(fc.spot.timezone), False) if dd.get("sunset") else "—",
                   "Moon": f"{dd['phase_name']} ({dd['illumination_pct']:.0f}%)", "Tides": tides_txt,
                   "Wind max": f"{fmt(seg['wind_mph'].max())} mph", "Water": f"{fmt(seg['water_temp_f'].mean(), '{:.1f}')}°F"}
            for sp in species:
                rec[species_meta(sp)["short"]] = seg[f"bite_{sp}"].max()
            days.append(rec)
        st.dataframe(pd.DataFrame(days), hide_index=True, use_container_width=True,
                     column_config={species_meta(sp)["short"]: st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f") for sp in species})

# =============================================================== CONDITIONS
with tabs[3]:
    names = {fc.spot.name: sid for sid, fc in VIS.items()}
    csel = names[st.selectbox("Spot", list(names), key="csel",
                              index=list(names.values()).index(focus_id) if focus_id in names.values() else 0)]
    fc = VIS[csel]
    now = fc.now_row()
    hist = fc.hourly.loc[:pd.Timestamp.now(tz="UTC")]
    buoy_last = hist["buoy_wave_ft"].dropna() if "buoy_wave_ft" in hist else pd.Series(dtype=float)
    nxt = [e for e in fc.tide_events if pd.Timestamp(e.time) > pd.Timestamp.now(tz="UTC")][:4]
    tz = fc.spot.timezone
    ph = fc.daily[0] if fc.daily else {}
    cards = [
        ("💧 Water temperature", f"{fmt(now.get('water_temp_f'), '{:.1f}')}°F", f"{now.get('water_temp_kind')} · 3-day trend {fmt(now.get('water_temp_trend_3d'), '{:+.1f}')}°F"),
        ("💨 Wind", f"{fmt(now.get('wind_mph'))} mph {compass(now.get('wind_dir'))}", f"gusts {fmt(now.get('gust_mph'))} mph · {wind_text(now).split('· ')[-1]}"),
        ("🌊 Surf (est.)", f"~{fmt(now.get('nearshore_wave_ft'), '{:.1f}')} ft", f"spot model {fmt(now.get('wave_ft'), '{:.1f}')} ft @ {fmt(now.get('wave_period_s'))} s"
                                                                                + (f" · offshore buoy {buoy_last.iloc[-1]:.1f} ft (measured)" if len(buoy_last) else "")),
        ("🌀 Pressure", f"{fmt(now.get('pressure_hpa'))} hPa", f"{fmt(now.get('pressure_tend_3h'), '{:+.1f}')} hPa / 3 h · {fmt(now.get('pressure_hpa') * 0.02953 if isnum(now.get('pressure_hpa')) else None, '{:.2f}')} inHg"),
        ("↕️ Tide", esc(now.get("tide_stage") or "—"), " · ".join(f"{'High' if e.type == 'H' else 'Low'} {hour_label(pd.Timestamp(e.time).tz_convert(tz))} ({e.height_ft:.1f} ft)" for e in nxt[:2])),
        ("➰ Current", f"{fmt(abs(now.get('current_kt')) if isnum(now.get('current_kt')) else None, '{:.1f}')} kt {now.get('current_phase') or ''}",
         f"{fmt((now.get('current_norm') or np.nan) * 100)}% of peak" + (" · proxy" if now.get("current_is_proxy") else "")),
        ("🌙 Moon", f"{ph.get('phase_name', '—')}", f"{ph.get('illumination_pct', 0):.0f}% lit · {ph.get('tide_cycle', '')} tides"),
        ("☀️ Sun", f"{hour_label(pd.Timestamp(ph['sunrise']).tz_convert(tz), False) if ph.get('sunrise') else '—'} – {hour_label(pd.Timestamp(ph['sunset']).tz_convert(tz), False) if ph.get('sunset') else '—'}",
         f"civil dawn {hour_label(pd.Timestamp(ph['civil_dawn']).tz_convert(tz), False) if ph.get('civil_dawn') else '—'} · dusk {hour_label(pd.Timestamp(ph['civil_dusk']).tz_convert(tz), False) if ph.get('civil_dusk') else '—'}"),
        ("🌩️ Storm status", str(now.get("storm_phase", "none")).title(), f"surge {fmt(now.get('water_level_anom_ft'), '{:+.1f}')} ft (regional)" if isnum(now.get("water_level_anom_ft")) else "no surge data"),
        ("⛅ Sky", esc(now.get("short_forecast") or f"{fmt(now.get('cloud_pct'))}% cloud"), f"air {fmt(now.get('air_temp_f'))}°F · rain chance {fmt(now.get('pop_pct'))}%"),
    ]
    for i in range(0, len(cards), 5):
        cols = st.columns(5)
        for col, (t, v, n) in zip(cols, cards[i:i + 5]):
            col.markdown(f'<div class="sc-card sc-kpi" style="margin-bottom:.7rem"><div class="lab">{t}</div><div class="val" style="font-size:1.25rem">{v}</div><div class="note">{esc(n)}</div></div>',
                         unsafe_allow_html=True)

    st.markdown("##### 📚 Data sources for this spot")
    srows = []
    for s in fc.sources:
        age = f"{s.age_hours:.1f} h" if s.age_hours is not None else ("cached" if s.stale else "—")
        srows.append(f"<tr><td>{esc(s.feature)}</td><td>{kind_badge(s.kind.value)}</td><td>{esc(s.provider)}</td>"
                     f"<td>{esc((s.station_id or '') + (' · ' + s.station_name if s.station_name else ''))}</td>"
                     f"<td>{fmt(km_to_mi(s.distance_km) if s.distance_km is not None else None, '{:.1f} mi')}</td><td>{age}</td>"
                     f"<td style='color:#94a3b8'>{esc(s.note)}</td></tr>")
    st.markdown(f"""<div class="sc-card" style="overflow-x:auto"><table style="width:100%;font-size:.82rem;border-collapse:collapse">
      <thead><tr style="text-align:left;color:#94a3b8"><th>Feature</th><th>Type</th><th>Provider</th><th>Station</th><th>Distance</th><th>Obs. age</th><th>Notes</th></tr></thead>
      <tbody>{''.join(srows)}</tbody></table></div>""", unsafe_allow_html=True)
    if fc.quality["missing"]:
        st.warning("Missing inputs (scores fall back to neutral values and the quality grade drops): " + ", ".join(fc.quality["missing"]))
    st.caption("Measured = instruments · Predicted = official NOAA tables / NWS forecast · Model = numerical model or reanalysis · "
               "Interpolated = between official values · Heuristic = StrikeCast rule-based estimate.")

    st.markdown("##### ↕️ Tide table (local time)")
    tt = [{"Day": f"{pd.Timestamp(e.time).tz_convert(tz):%a %b %d}", "Time": hour_label(pd.Timestamp(e.time).tz_convert(tz), False) + f" ({pd.Timestamp(e.time).tz_convert(tz):%I:%M})",
           "Type": "High" if e.type == "H" else "Low", "Height (ft MLLW)": round(e.height_ft, 2)}
          for e in fc.tide_events if start_ts <= pd.Timestamp(e.time) <= end_ts]
    st.dataframe(pd.DataFrame(tt), hide_index=True, use_container_width=True)

# ================================================================== COMPARE
with tabs[4]:
    st.markdown("Compare the environmental conditions of two dates at the same spot — e.g. a great day vs a slow day. "
                "Observed data (buoys, gauges) are shown separately from model hindcasts.")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    names_all = {s.name: s.id for s in ALL_SPOTS}
    csp = names_all[c1.selectbox("Spot", list(names_all), index=list(names_all.values()).index("robert_moses_democrat_point"), key="cmp_spot")]
    da = c2.date_input("Date A", value=date(2026, 10, 10), max_value=today_local, key="cmp_a")
    db_ = c3.date_input("Date B", value=date(2026, 10, 22), max_value=today_local, key="cmp_b")
    c4.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    go_cmp = c4.button("Compare ▶", type="primary", use_container_width=True)
    ckey = (csp, da, db_)
    if go_cmp or (ss.get("cmp_key") == ckey and "cmp" in ss):
        if ss.get("cmp_key") != ckey:
            bar = st.progress(0.0, text="Rebuilding both days from archives…")
            ss["cmp"] = run_sync(compare_dates(get_spot(csp), da, db_, lambda p, l: bar.progress(min(p, 1.0), text=f"{l} · {p * 100:.0f}%")))
            ss["cmp_key"] = ckey
            bar.empty()
        cmp = ss["cmp"]
        la, lb = f"{da:%b %d, %Y}", f"{db_:%b %d, %Y}"
        st.markdown('<div class="sc-card"><h4>What was different?</h4><ul class="sc-explain">'
                    + "".join(f"<li>{esc(e)}</li>" for e in cmp.explanation) + "</ul></div>", unsafe_allow_html=True)
        st.plotly_chart(charts.compare_chart(cmp.a, cmp.b, la, lb), use_container_width=True, config={"displayModeBar": False})
        t = cmp.table.copy()

        def _v(x, unit=""):
            if isinstance(x, (float, np.floating)):
                return "—" if np.isnan(x) else f"{x:,.1f}{(' ' + unit) if unit else ''}"
            return esc(x)
        def _diff_cell(x) -> str:
            if not isnum(x):
                return "<td></td>"
            return f"<td style='color:{'#4ade80' if x > 0 else '#f87171' if x < 0 else '#94a3b8'}'>{x:+.1f}</td>"

        rows_html = "".join(
            f"<tr><td>{esc(r.metric)}</td><td>{_v(r.A, r.unit)} {kind_badge(r.kind_a)}</td>"
            f"<td>{_v(r.B, r.unit)} {kind_badge(r.kind_b)}</td>{_diff_cell(r['diff'])}</tr>"
            for _, r in t.iterrows())
        st.markdown(f"""<div class="sc-card" style="overflow-x:auto"><table style="width:100%;font-size:.84rem;border-collapse:collapse">
          <thead><tr style="text-align:left;color:#94a3b8"><th>Metric</th><th>{la}</th><th>{lb}</th><th>B − A</th></tr></thead>
          <tbody>{rows_html}</tbody></table></div>""", unsafe_allow_html=True)
        st.caption("Environmental differences are shown for context only — they do not prove why a catch was better or worse.")

# ====================================================================== LOG
with tabs[5]:
    st.markdown("Log every trip — **including skunks**. Each record captures the environmental snapshot for that hour, "
                "so StrikeCast can learn what works *for you*.")
    with st.form("log", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns(4)
        lspot = c1.selectbox("Spot", [s.name for s in ALL_SPOTS])
        ldate = c2.date_input("Date", value=today_local, max_value=today_local + timedelta(days=6))
        ltime = c3.time_input("Start time", value=datetime.now().replace(minute=0, second=0, microsecond=0).time())
        ldur = c4.number_input("Hours fished", 0.5, 24.0, 2.0, 0.5)
        c1, c2, c3, c4 = st.columns(4)
        lsp = c1.selectbox("Target species", SPECIES, format_func=lambda s: species_meta(s)["name"])
        lcaught = c2.number_input("Fish caught", 0, 500, 0)
        llen = c3.number_input("Largest (inches)", 0.0, 70.0, 0.0, 0.5)
        lang = c4.number_input("Anglers", 1, 20, 1)
        c1, c2, c3, c4 = st.columns(4)
        llure = c1.text_input("Lure / bait used")
        lbait = c2.selectbox("Bait seen", ["none", "adult bunker", "peanut bunker", "sand eels", "bay anchovies", "silversides", "other"])
        lbirds = c3.checkbox("Birds diving")
        lblitz = c4.checkbox("Surface blitz")
        lnotes = st.text_area("Notes (water color, structure, what worked)", height=70)
        lconf = st.radio("Record type", ["user_reported", "verified"], horizontal=True)
        submitted = st.form_submit_button("💾 Save trip", type="primary")
    if submitted:
        sid = {s.name: s.id for s in ALL_SPOTS}[lspot]
        when = pd.Timestamp(datetime.combine(ldate, ltime)).tz_localize(TZ).tz_convert("UTC")
        feats = {}
        if sid in forecasts and when.floor("h") in forecasts[sid].hourly.index:
            feats = feature_snapshot(forecasts[sid], when)
        else:
            bar = st.progress(0.0, text="Pulling the environmental snapshot for that date…")
            hist_fc = run_sync(collect([get_spot(sid)], historical_frame(ldate, TZ), lambda p, l: bar.progress(min(p, 1.0), text=l)))
            feats = feature_snapshot(hist_fc[sid], when)
            bar.empty()
        oid = add_observation({"spot_id": sid, "start_utc": when, "duration_h": ldur, "species": lsp,
                               "fish_caught": int(lcaught), "largest_in": llen or None, "lure": llure or None,
                               "bait_seen": None if lbait == "none" else lbait, "birds_diving": lbirds, "blitz": lblitz,
                               "anglers": int(lang), "notes": lnotes or None, "confidence": lconf}, feats)
        st.success(f"Saved trip #{oid} with {len(feats)} environmental features.")
    obs = list_observations()
    st.markdown(f"##### Your log · {len(obs)} trips")
    if not obs.empty:
        show = obs.copy()
        show["start"] = pd.to_datetime(show["start_utc"], utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %I:%M %p")
        st.dataframe(show[["id", "start", "spot_id", "species", "fish_caught", "largest_in", "lure", "bait_seen", "birds_diving",
                           "blitz", "confidence", "has_features"]], hide_index=True, use_container_width=True)
        c1, c2 = st.columns([1, 3])
        did = c1.number_input("Delete trip id", min_value=0, value=0)
        if c1.button("Delete") and did:
            delete_observation(int(did))
            st.rerun()
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Export CSV", export_csv(), file_name="strikecast_log.csv", mime="text/csv", use_container_width=True)
    up = c2.file_uploader("⬆️ Import CSV", type=["csv"], label_visibility="collapsed")
    if up is not None and ss.get("imported") != up.file_id:
        try:
            n = import_csv(up.getvalue())
            ss["imported"] = up.file_id
            st.success(f"Imported {n} records.")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    st.markdown("##### 🤖 Learn from your log")
    st.caption("Trains a logistic-regression model per species with a chronological train/test split. Needs ≥20 trips "
               "(≥5 with fish, ≥5 without). Probabilities are only calibrated once you have 60+ trips.")
    cols = st.columns(len(SPECIES))
    for col, sp in zip(cols, SPECIES):
        if col.button(f"Train {species_meta(sp)['short']}", use_container_width=True, key=f"train_{sp}"):
            rep = train_species(sp)
            (st.success if rep.ok else st.info)(f"{species_meta(sp)['name']}: {rep.message}")
            if rep.ok:
                st.json({"metrics": rep.metrics, "top factors (coef)": rep.importance, "calibrated": rep.calibrated})
        col.caption("✅ model ready" if model_exists(sp) else "no model yet")

# =================================================================== SPECIES
with tabs[6]:
    regs = regulations_config()
    st.markdown(f"Regulations below are a convenience summary (last reviewed {regs.get('last_reviewed')}). "
                f"**Always confirm with [NYSDEC]({regs.get('verify_url')})** and enroll in the free "
                f"[NY Recreational Marine Fishing Registry]({regs.get('registry_url')}).")
    st.plotly_chart(charts.season_chart(focus_spot, SPECIES, pd.Timestamp.now(tz=TZ).day_of_year), use_container_width=True,
                    config={"displayModeBar": False})
    st.caption(f"Seasonal migration curves adjusted for {focus_spot.name} (latitude & water body). Dotted line = today.")
    cols = st.columns(len(SPECIES))
    for col, sp in zip(cols, SPECIES):
        meta = species_meta(sp)
        stt = season_status(sp, today_local)
        badge = ('<span class="sc-badge" style="background:#14532d;color:#bbf7d0">OPEN today</span>' if stt.get("open")
                 else '<span class="sc-badge" style="background:#7f1d1d;color:#fecaca">CLOSED today</span>')
        tips = "".join(f"<li><b>{esc(k.replace('_', ' '))}:</b> {esc(v)}</li>" for k, v in meta.get("tactics", {}).items())
        t = meta["temp_f"]
        col.markdown(f"""<div class="sc-card"><div style="height:4px;background:{meta['color']};border-radius:4px;margin-bottom:.6rem"></div>
          <h4>{esc(meta['name'])}</h4>{badge}
          <div class="sc-cond" style="grid-template-columns:1fr">
            <div><span>Size</span> {esc(stt.get('size', '—'))}</div><div><span>Bag</span> {esc(stt.get('bag', '—'))}</div>
            <div><span>Best water</span> {t[1]}–{t[2]}°F (range {t[0]}–{t[3]}°F)</div></div>
          <div class="sc-sub" style="margin-top:.4rem">{esc(stt.get('note') or '')}</div>
          <ul style="font-size:.8rem;padding-left:1.1rem;margin-top:.5rem">{tips}</ul></div>""", unsafe_allow_html=True)

st.markdown("""<div class="sc-foot">
<b>Data:</b> NOAA CO-OPS tides & currents · NOAA NDBC buoys · NOAA/NWS forecasts & alerts · NOAA NCEI CUDEM bathymetry ·
NOAA ENC charts · <a href="https://open-meteo.com">Open-Meteo</a> (CC BY 4.0) weather & marine models · OpenStreetMap / Nominatim ·
OpenSeaMap · Zippopotam.us · USGS imagery · PyEphem astronomy. All free; no API keys.<br>
<b>Disclaimer:</b> StrikeCast indices are environmental suitability scores, not guarantees or calibrated catch probabilities.
Conditions on jetties and in inlets can be dangerous — always check the official NWS forecast and use judgment.
</div>""", unsafe_allow_html=True)
