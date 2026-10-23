"""Validate StrikeCast wave inputs and surf estimates against independent data.

1. Offshore model skill: Open-Meteo Marine at each NDBC buoy vs the buoy's
   measured Hs / dominant period / direction (last ~45 days, realtime files).
2. Breaking surf at ocean beaches vs NWS Surf Zone Forecasts (independent NWS
   forecaster product, beach season only), for the NEW physics model (with and
   without buoy anchoring) and the OLD heuristic it replaces.
3. Long Island Sound wind chop at Sunken Meadow vs the NWS coastal-waters
   forecast for zone ANZ335.

    python scripts/validate_surf.py                      # defaults below
    python scripts/validate_surf.py 2026-06-01 2026-09-15

Writes docs/validation_results.json and prints a summary used in docs/VALIDATION.md.
Nothing here tunes the model: the numbers are reported as they come out.
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import exposure_config  # noqa: E402
from app.core.geo import destination, wind_components  # noqa: E402
from app.core.http import HttpClient  # noqa: E402
from app.features import surf as phys  # noqa: E402
from app.providers import marine, nws_marine, weather  # noqa: E402
from app.services import surf as svc  # noqa: E402
from app.services.forecast import _hourly, get_spot  # noqa: E402
from app.services.stations import station_meta  # noqa: E402

TZ = "America/New_York"
OCEAN = {"jones_beach_field_2": 1.0, "point_lookout_jetties": 0.85, "robert_moses_democrat_point": 0.8}  # old exposure
BUOY = {"jones_beach_field_2": "44065", "point_lookout_jetties": "44065", "robert_moses_democrat_point": "44025"}
OUT = ROOT / "docs" / "validation_results.json"


def old_heuristic(off_ft, wave_dir, facing, exposure):
    """The pre-upgrade formula (app/features/weather_features.nearshore_waves, ocean branch)."""
    deg_off = np.abs(((wave_dir - facing + 180) % 360) - 180)
    return off_ft * exposure * np.where(np.isfinite(deg_off), np.cos(np.radians(deg_off) / 2) ** 2, 0.6)


async def ndbc_range(http, sid, start: date, end: date) -> pd.DataFrame:
    frames = []
    now = datetime.now(timezone.utc)
    m = date(start.year, start.month, 1)
    while m <= end:
        st = datetime(m.year, m.month, 1, tzinfo=timezone.utc)
        en = (st + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
        if st > now - timedelta(days=44):
            st = max(st, now - timedelta(days=43))
        try:
            df, _ = await marine.ndbc_observations(http, sid, start=st, end=min(en, now))
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"  NDBC {sid} {m:%Y-%m}: {str(exc)[:80]}")
        m = (st + timedelta(days=32)).replace(day=1).date()
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


def skill(a: pd.Series, b: pd.Series) -> dict:
    ok = a.notna() & b.notna()
    if ok.sum() < 5:
        return {"n": int(ok.sum())}
    d = a[ok] - b[ok]
    return {"n": int(ok.sum()), "bias": round(float(d.mean()), 2), "mae": round(float(d.abs().mean()), 2),
            "rmse": round(float(np.sqrt((d ** 2).mean())), 2), "r": round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 2)}


async def offshore_skill(http, days=40) -> dict:
    out = {}
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    pts = {sid: (station_meta(f"ndbc:{sid}")["lat"], station_meta(f"ndbc:{sid}")["lon"]) for sid in ("44065", "44025")}
    res = await marine.openmeteo_marine(http, list(pts.values()), start=start, end=end)
    for (sid, _), (m, _) in zip(pts.items(), res):
        b = (await ndbc_range(http, sid, start, end))
        b = _hourly(b, ["wave_ft", "wave_period_s", "wave_dir", "avg_period_s"]).reindex(m.index)
        dd = ((b["wave_dir"] - m["wave_dir"] + 180) % 360) - 180
        out[sid] = {"period": f"{start}..{end}", "Hs_ft": skill(m["wave_ft"], b["wave_ft"]),
                    "dir_bias_deg_median": round(float(dd.median()), 1), "dir_mae_deg": round(float(dd.abs().mean()), 1),
                    "Tp_over_model_T_median": round(float((b["wave_period_s"] / m["wave_period_s"]).median()), 2)}
        print(f"Offshore skill {sid}: {out[sid]}")
    return out


async def surf_vs_srf(http, start: date, end: date) -> dict:
    spots = [get_spot(s) for s in OCEAN]
    expo = exposure_config()
    old_pts = [destination(s.lat, s.lon, s.facing_deg, 1.0) for s in spots]
    buoys = sorted(set(BUOY.values()))
    bpts = [(station_meta(f"ndbc:{b}")["lat"], station_meta(f"ndbc:{b}")["lon"]) for b in buoys]
    pts = [s.boundary_point() for s in spots] + old_pts + bpts
    print(f"Open-Meteo marine archive {start}..{end} for {len(pts)} points …")
    res = await marine.openmeteo_marine(http, pts, start=start - timedelta(days=1), end=end + timedelta(days=1))
    n = len(spots)
    model_bc = {s.id: res[i][0] for i, s in enumerate(spots)}
    model_old = {s.id: res[n + i][0] for i, s in enumerate(spots)}
    model_buoy = {b: res[2 * n + j][0] for j, b in enumerate(buoys)}
    obs = {}
    for b in buoys:
        raw = await ndbc_range(http, b, start - timedelta(days=1), end + timedelta(days=1))
        obs[b] = _hourly(raw, ["wave_ft", "wave_period_s", "wave_dir"]) if not raw.empty else pd.DataFrame()
    print("NWS surf zone forecasts (IEM archive) …")
    srf = await nws_marine.surf_zone_forecasts(http, datetime.combine(start - timedelta(days=1), datetime.min.time(), timezone.utc),
                                               datetime.combine(end, datetime.min.time(), timezone.utc))
    out = {}
    for s in spots:
        idx = model_bc[s.id].index
        df = model_bc[s.id].copy()
        geom = phys.Geometry.from_dict(expo[s.id])
        b = BUOY[s.id]
        variants = {}
        anchor = {}
        if not obs[b].empty:
            mb = model_buoy[b].reindex(idx)
            anchor = svc.anchor_to_buoy(obs[b]["wave_ft"].reindex(idx), mb["wave_ft"], idx[-1], "historical",
                                        buoy_dir=obs[b]["wave_dir"].reindex(idx), model_dir=mb["wave_dir"])
        for name, anc in (("new_model_only", {}), ("new_buoy_anchored", anchor)):
            c = svc.SurfContext(spot=s, geometry=geom, boundary_depth_m=expo[s.id].get("boundary_depth_m"),
                                anchor=anc, mode="historical", now=idx[-1])
            variants[name] = svc.compute_surf(df, c)[0]["surf_ft"]
        om = model_old[s.id].reindex(idx)
        variants["old_heuristic"] = pd.Series(old_heuristic(om["wave_ft"], om["wave_dir"].fillna(om["swell_dir"]),
                                                            s.facing_deg, OCEAN[s.id]), index=idx)
        # daily daytime (8 AM-6 PM local) mean vs that day's NWS surf forecast
        days = {}
        for sd in srf:
            if sd.zone != s.surf_zone or not (start <= sd.day <= end):
                continue
            day0 = pd.Timestamp(sd.day).tz_localize(TZ).tz_convert("UTC")
            if sd.issued > day0 + pd.Timedelta(hours=8):
                continue
            if sd.day not in days or sd.issued > days[sd.day].issued:
                days[sd.day] = sd
        rows = []
        for d, sd in sorted(days.items()):
            a = pd.Timestamp(d).tz_localize(TZ).tz_convert("UTC") + pd.Timedelta(hours=8)
            seg = slice(a, a + pd.Timedelta(hours=10))
            row = {"day": str(d), "srf_lo": sd.surf_lo_ft, "srf_hi": sd.surf_hi_ft}
            for k, v in variants.items():
                row[k] = round(float(v.loc[seg].mean()), 2) if v.loc[seg].notna().any() else None
            rows.append(row)
        tab = pd.DataFrame(rows)
        summ = {"zone": s.surf_zone, "days": len(tab)}
        if len(tab):
            mid = (tab["srf_lo"] + tab["srf_hi"]) / 2
            for k in variants:
                v = tab[k].astype(float)
                ok = v.notna()
                within = ((v >= tab["srf_lo"] - 0.5) & (v <= tab["srf_hi"] + 0.5))[ok]
                summ[k] = {"within_range_pct": round(100 * float(within.mean()), 1),
                           "bias_vs_mid_ft": round(float((v - mid)[ok].mean()), 2),
                           "mae_vs_mid_ft": round(float((v - mid)[ok].abs().mean()), 2),
                           "below_range_pct": round(100 * float((v < tab["srf_lo"] - 0.5)[ok].mean()), 1)}
        if anchor:
            summ["anchor"] = {k: anchor.get(k) for k in ("ratio_longterm", "dir_offset_longterm_deg", "mae_ft", "bias_ft")}
        out[s.id] = {"summary": summ, "daily": rows}
        print(f"{s.id} vs NWS SRF {s.surf_zone}: {json.dumps(summ)}")
    return out


async def sound_chop(http, start: date, end: date) -> dict:
    s = get_spot("sunken_meadow_field_3")
    geom = phys.Geometry.from_dict(exposure_config()[s.id])
    wx = (await weather.openmeteo_weather(http, [(s.lat, s.lon)], start=start, end=end))[0][0]
    df = wx[["wind_mph", "wind_dir"]].copy()
    on, _ = wind_components(df["wind_mph"].to_numpy(), df["wind_dir"].to_numpy(), s.facing_deg)
    df["onshore_mph"] = on
    out_df, _ = svc.compute_surf(df, svc.SurfContext(spot=s, geometry=geom, boundary_depth_m=None, mode="historical",
                                                     now=df.index[-1]))
    texts = await nws_marine.archived_products(http, "CWFOKX", datetime.combine(start, datetime.min.time(), timezone.utc),
                                               datetime.combine(end, datetime.min.time(), timezone.utc))
    z, _, note = nws_marine.cwf_hourly_from_products(texts, "ANZ335", df.index)
    est = out_df["surf_ft"]
    ok = est.notna() & z["nws_seas_hi_ft"].notna()
    within = ((est >= z["nws_seas_lo_ft"] - 0.5) & (est <= z["nws_seas_hi_ft"] + 0.5))[ok]
    mid = (z["nws_seas_lo_ft"] + z["nws_seas_hi_ft"]) / 2
    res = {"zone": "ANZ335", "hours": int(ok.sum()), "within_range_pct": round(100 * float(within.mean()), 1),
           "bias_vs_mid_ft": round(float((est - mid)[ok].mean()), 2), "note": note,
           "onshore_hours_within_pct": round(100 * float(within[(df["onshore_mph"] > 5)[ok]].mean()), 1)}
    print(f"Sunken Meadow chop vs NWS ANZ335: {res}")
    return res


async def main():
    a = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 6, 1)
    b = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 9, 15)
    async with HttpClient() as http:
        results = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
                   "offshore_model_vs_buoy": await offshore_skill(http),
                   "surf_vs_nws_srf": await surf_vs_srf(http, a, b),
                   "sound_chop_vs_nws": await sound_chop(http, date(2026, 10, 1), date(2026, 10, 22))}
    OUT.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
