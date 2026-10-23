"""Per-spot surf estimation: offshore boundary conditions → nearshore surf.

Pipeline for each hour
----------------------
1. Offshore boundary conditions, best available first (source recorded per hour):
   a. Open-Meteo Marine model at the spot's offshore point (swell + wind-sea
      partitions), scaled by the measured/model ratio at the nearest NDBC buoy
      when that buoy is fresh and consistent (observation anchoring, decaying
      over 12 h into the forecast);
   b. NWS coastal-waters "Wave Detail" components for the spot's zone;
   c. measured NDBC buoy partitions (past/current hours only — never projected).
2. Ocean beaches: shoaling + refraction + sheltering/diffraction + depth-limited
   breaking (app/features/surf.py). Sheltered water: SPM fetch-limited wind chop.
3. Uncertainty range from direction (±10°) and height sensitivity, a confidence
   label, a trend, wind-quality and a separate surf-safety assessment.
4. Validation flags (buoy/model disagreement, NWS zone forecast mismatch,
   implausibly low surf, stale data). Flags lower confidence; they never change
   the estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.features import surf as phys
from app.schemas.env import Spot

FT = phys.FT_PER_M
NAVD88_ABOVE_MLLW_M = 0.7          # Long Island south shore (NOAA VDatum, approx.)
ANCHOR_RATIO_OK = (0.5, 2.0)
ANCHOR_DECAY_H = 12.0
U_KOMAR = 0.15        # scatter of the Komar & Gaughan breaker-height relation in field data
U_SPM = 0.20          # scatter of the SPM fetch-limited growth curves


@dataclass
class SurfContext:
    spot: Spot
    geometry: phys.Geometry | None
    boundary_depth_m: float | None
    profile: dict[str, Any] = field(default_factory=dict)       # slope / bar from bathymetry
    anchor: dict[str, Any] = field(default_factory=dict)        # buoy anchoring summary
    now: pd.Timestamp | None = None
    mode: str = "forecast"
    model_stale: bool = False


# ------------------------------------------------------------ anchoring
def _circ_median(deg: pd.Series) -> float:
    """Robust circular centre of signed angle differences (deg, -180..180)."""
    d = deg.dropna().astype(float)
    if d.empty:
        return np.nan
    return float(np.degrees(np.arctan2(np.sin(np.radians(d)).median(), np.cos(np.radians(d)).median())))


def _blend(idx: pd.DatetimeIndex, hourly: pd.Series, recent: float, longterm: float, last_obs: pd.Timestamp,
           mode: str) -> pd.Series:
    """Correction per hour: measured hours use a 6 h rolling value; later hours relax from the
    recent value to the running (multi-day) bias with a 12 h e-folding time."""
    roll = hourly.rolling(6, min_periods=2, center=True).median()
    if mode == "historical":
        return roll.fillna(longterm)
    dt_h = np.clip((idx - last_obs) / pd.Timedelta(hours=1), 0, None)
    fut = longterm + (recent - longterm) * np.exp(-dt_h / ANCHOR_DECAY_H)
    past = roll.reindex(idx).fillna(longterm)
    return pd.Series(np.where(idx <= last_obs, past, fut), index=idx)


def anchor_to_buoy(buoy_ft: pd.Series, model_at_buoy_ft: pd.Series, now: pd.Timestamp, mode: str,
                   buoy_dir: pd.Series | None = None, model_dir: pd.Series | None = None) -> dict[str, Any]:
    """Observation anchoring: correct the wave model with a co-located NDBC buoy.

    Heights: ratio measured / model at the buoy. Directions: measured minus model
    mean wave direction at the buoy. Each correction = running bias over all
    overlapping hours (up to 4 days) plus the recent (12 h) departure, which
    relaxes to the running bias over ~12 h of forecast. Ratios outside 0.5-2 or
    direction offsets beyond 45 deg are a provider inconsistency: not applied, flagged.
    """
    idx = model_at_buoy_ft.index
    b = buoy_ft.reindex(idx).astype(float)
    m = model_at_buoy_ft.astype(float)
    ok = b.notna() & m.notna() & (m > 0.3) & (b > 0.3)
    info: dict[str, Any] = {"applied": False, "n": int(ok.sum()), "factor": None, "dir_offset": None}
    if ok.sum() < 3:
        info["reason"] = "too few overlapping buoy/model hours"
        return info
    ratio = (b / m).where(ok)
    last_obs = ratio.dropna().index.max()
    recent = ratio.loc[last_obs - pd.Timedelta(hours=12):last_obs].dropna()
    r_recent = float(recent.median()) if len(recent) >= 3 else float(ratio.dropna().median())
    r_long = float(ratio.dropna().median()) if ok.sum() >= 24 else r_recent
    info.update(ratio=round(r_recent, 2), ratio_longterm=round(r_long, 2), last_obs=last_obs,
                mae_ft=round(float((b - m).abs()[ok].mean()), 2), bias_ft=round(float((m - b)[ok].mean()), 2))
    if not (ANCHOR_RATIO_OK[0] <= r_recent <= ANCHOR_RATIO_OK[1]):
        info["reason"] = f"buoy/model ratio {r_recent:.2f} outside {ANCHOR_RATIO_OK}: provider inconsistency, not applied"
        info["inconsistent"] = True
        return info
    info["applied"] = True
    info["factor"] = _blend(idx, ratio.clip(*ANCHOR_RATIO_OK), r_recent, r_long, last_obs, mode).clip(*ANCHOR_RATIO_OK)
    if buoy_dir is not None and model_dir is not None:
        dd = ((buoy_dir.reindex(idx).astype(float) - model_dir.astype(float) + 180) % 360) - 180
        dd = dd.where(ok)
        if dd.notna().sum() >= 3:
            d_recent = _circ_median(dd.loc[last_obs - pd.Timedelta(hours=12):last_obs])
            d_long = _circ_median(dd) if dd.notna().sum() >= 24 else d_recent
            info.update(dir_offset_deg=round(d_recent, 1), dir_offset_longterm_deg=round(d_long, 1))
            if np.isfinite(d_recent) and abs(d_recent) <= 45 and abs(d_long) <= 45:
                info["dir_offset"] = _blend(idx, dd, d_recent, d_long, last_obs, mode).clip(-45, 45)
            else:
                info["dir_reason"] = f"direction offset {d_recent:+.0f} deg too large: not applied"
    return info


def anchor_ratio(buoy_ft: pd.Series, model_at_buoy_ft: pd.Series, now: pd.Timestamp, mode: str) -> tuple[pd.Series, dict]:
    """Height-only anchoring (kept for callers/tests): returns (factor series, info)."""
    info = anchor_to_buoy(buoy_ft, model_at_buoy_ft, now, mode)
    f = info.get("factor")
    return (f if f is not None else pd.Series(1.0, index=model_at_buoy_ft.index)), info


# ------------------------------------------------------------ partitions
def _first(df: pd.DataFrame, *cols: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for c in cols:
        if c in df:
            out = out.combine_first(pd.to_numeric(df[c], errors="coerce"))
    return out


# Spectral-shape period conversions (JONSWAP, gamma = 3.3): when a provider gives
# only mean periods, the peak period is Tp ~ Te / 0.90 for the total sea (energy
# period Tm-1,0) and Tp ~ Tm01 / 0.78 for partitions. Both match NDBC buoys vs
# Open-Meteo here (median measured/model ratios 1.12 and 1.28; docs/VALIDATION.md).
TP_FROM_TE = 1 / 0.90
TP_FROM_TM01 = 1 / 0.78


def _period(df: pd.DataFrame, peak: str, mean: str, conv: float) -> pd.Series:
    return _first(df, peak).combine_first(_first(df, mean) * conv)


def model_partitions(df: pd.DataFrame, factor: pd.Series, dir_offset: pd.Series | None = None) -> tuple[list[dict], pd.Series]:
    """Swell + wind-sea partitions (metres) at the offshore point; returns (parts, valid-hours mask)."""
    off = dir_offset.reindex(df.index).fillna(0.0) if dir_offset is not None else pd.Series(0.0, index=df.index)
    tot = _first(df, "wave_ft") / FT
    sw = _first(df, "swell_ft") / FT
    sw2 = _first(df, "swell2_ft") / FT
    ww = _first(df, "windwave_ft") / FT
    have_parts = sw.notna() | ww.notna()
    # Partition energies add up to the total (Hs^2 = sum of Hi^2). Energy the model
    # reports in the total but not in any partition is carried as its own
    # component (total period / direction) rather than inflating the partitions.
    psum2 = sw.fillna(0) ** 2 + sw2.fillna(0) ** 2 + ww.fillna(0) ** 2
    resid = np.sqrt((tot ** 2 - psum2).clip(lower=0)).where(have_parts & tot.notna())
    resid = resid.where(resid ** 2 > 0.1 * tot ** 2)
    total_T = _period(df, "wave_peak_period_s", "wave_period_s", TP_FROM_TE)
    total_D = (_first(df, "wave_dir") + off) % 360
    parts = [
        {"H": (sw * factor).to_numpy(), "T": _period(df, "swell_peak_period_s", "swell_period_s", TP_FROM_TM01).to_numpy(),
         "dir": ((_first(df, "swell_dir", "wave_dir") + off) % 360).to_numpy(), "sigma": phys.SIGMA_SWELL},
        {"H": (sw2 * factor).to_numpy(), "T": (_first(df, "swell2_period_s") * TP_FROM_TM01).to_numpy(),
         "dir": ((_first(df, "swell2_dir") + off) % 360).to_numpy(), "sigma": phys.SIGMA_SWELL},
        {"H": (ww * factor).to_numpy(),
         "T": _period(df, "windwave_peak_period_s", "windwave_period_s", TP_FROM_TM01).to_numpy(),
         "dir": ((_first(df, "windwave_dir", "wave_dir") + off) % 360).to_numpy(), "sigma": phys.SIGMA_WIND_SEA},
        {"H": (resid * factor).to_numpy(), "T": total_T.to_numpy(), "dir": total_D.to_numpy(), "sigma": 25.0},
    ]
    single = {"H": np.where(have_parts, np.nan, (tot * factor).to_numpy()), "T": total_T.to_numpy(),
              "dir": total_D.to_numpy(), "sigma": 25.0}
    return parts + [single], tot.notna()


def nws_partitions(df: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    parts = []
    for j in (1, 2):
        parts.append({"H": (_first(df, f"nws_c{j}_ft") / FT).to_numpy(), "T": _first(df, f"nws_c{j}_period_s").to_numpy(),
                      "dir": _first(df, f"nws_c{j}_dir").to_numpy(), "sigma": 20.0})
    return parts, _first(df, "nws_c1_ft").notna()


def buoy_partitions(df: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    sw, ww = _first(df, "buoy_swell_ft") / FT, _first(df, "buoy_windwave_ft") / FT
    have = sw.notna() | ww.notna()
    parts = [
        {"H": sw.to_numpy(), "T": _first(df, "buoy_swell_period_s").to_numpy(), "dir": _first(df, "buoy_swell_dir").to_numpy(),
         "sigma": phys.SIGMA_SWELL},
        {"H": ww.to_numpy(), "T": _first(df, "buoy_windwave_period_s").to_numpy(), "dir": _first(df, "buoy_windwave_dir").to_numpy(),
         "sigma": phys.SIGMA_WIND_SEA},
        {"H": np.where(have, np.nan, (_first(df, "buoy_wave_ft") / FT).to_numpy()), "T": _first(df, "buoy_period_s").to_numpy(),
         "dir": _first(df, "buoy_wave_dir").to_numpy(), "sigma": 25.0},
    ]
    return parts, (have | _first(df, "buoy_wave_ft").notna())


def _select(parts_by_src: list[tuple[str, list[dict], pd.Series]], n: int) -> tuple[list[dict], np.ndarray]:
    """Merge partition sets hour by hour in priority order; record the source used."""
    src = np.full(n, None, dtype=object)
    for name, _, valid in parts_by_src:
        v = valid.to_numpy() & (src == None)  # noqa: E711
        src[v] = name
    merged: list[dict] = []
    for name, parts, _ in parts_by_src:
        use = src == name
        for p in parts:
            merged.append({**p, "H": np.where(use, p["H"], np.nan)})
    return merged, src


def _rotate(parts: list[dict], facing: float, deg: float) -> list[dict]:
    """Rotate every partition `deg` toward (+) or away from (−) the shore normal."""
    out = []
    for p in parts:
        d = np.asarray(p["dir"], float)
        diff = ((d - facing + 180) % 360) - 180
        new = facing + np.sign(diff) * np.clip(np.abs(diff) - deg, 0, 180)
        out.append({**p, "dir": new % 360})
    return out


def _boundary_components(out: pd.DataFrame, sets: list, src: np.ndarray) -> None:
    """Swell / wind-wave values actually used at the offshore point (after anchoring),
    hour by hour from whichever source was selected. For NWS 'Wave Detail' the
    longer-period component is reported as swell."""
    n = len(src)
    cols = {f"{pre}_{suf}": np.full(n, np.nan) for pre in ("bc_swell", "bc_swell2", "bc_windwave")
            for suf in ("ft", "tp_s", "dir")}
    for name, parts, _ in sets:
        use = src == name
        if not use.any():
            continue
        if name == "model":
            pairs = ((parts[0], "bc_swell"), (parts[1], "bc_swell2"), (parts[2], "bc_windwave"))
        else:
            a, b = parts[0], parts[1]
            if name == "nws":
                swap = np.nan_to_num(np.asarray(b["T"], float)) > np.nan_to_num(np.asarray(a["T"], float))
                a = {k: np.where(swap, b[k], a[k]) if k != "sigma" else a[k] for k in a}
                b = {k: np.where(swap, parts[0][k], parts[1][k]) if k != "sigma" else b[k] for k in b}
            pairs = ((a, "bc_swell"), (b, "bc_windwave"))
        for p, pre in pairs:
            cols[f"{pre}_ft"] = np.where(use, np.asarray(p["H"], float) * FT, cols[f"{pre}_ft"])
            cols[f"{pre}_tp_s"] = np.where(use, np.asarray(p["T"], float), cols[f"{pre}_tp_s"])
            cols[f"{pre}_dir"] = np.where(use, np.asarray(p["dir"], float), cols[f"{pre}_dir"])
    for k, v in cols.items():
        h = cols[k.replace("_tp_s", "_ft").replace("_dir", "_ft")]
        out[k] = np.where(np.isfinite(h), v, np.nan)


def _offshore_summary(parts: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Total Hs (m), energy-dominant peak period and direction over all partitions."""
    hs = phys.combine_hs(*[p["H"] for p in parts])
    E = np.vstack([np.nan_to_num(np.asarray(p["H"], float)) ** 2 for p in parts])
    k = np.argmax(E, axis=0)
    T = np.vstack([np.asarray(p["T"], float) for p in parts])[k, np.arange(E.shape[1])]
    D = np.vstack([np.asarray(p["dir"], float) for p in parts])[k, np.arange(E.shape[1])]
    return hs, np.where(np.isfinite(hs), T, np.nan), np.where(np.isfinite(hs), D, np.nan)


# ------------------------------------------------------------- main entry
def compute_surf(df: pd.DataFrame, ctx: SurfContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    spot = ctx.spot
    idx = df.index
    n = len(idx)
    now = ctx.now if ctx.now is not None else pd.Timestamp.now(tz="UTC").floor("h")
    out = pd.DataFrame(index=idx)
    meta: dict[str, Any] = {"exposure": spot.exposure_class, "anchor": ctx.anchor, "profile": ctx.profile,
                            "boundary_depth_m": ctx.boundary_depth_m}
    lead_h = np.clip((idx - now) / pd.Timedelta(hours=1), 0, None) if ctx.mode == "forecast" else np.zeros(n)
    onshore = _first(df, "onshore_mph")

    if spot.is_open_ocean:
        factor = pd.Series(1.0, index=idx)
        if ctx.anchor.get("factor") is not None:
            factor = ctx.anchor["factor"].reindex(idx).fillna(1.0)
        dir_off = ctx.anchor.get("dir_offset")
        sets = [("model", *model_partitions(df, factor, dir_off)), ("nws", *nws_partitions(df))]
        if "buoy_wave_ft" in df:
            bp, bv = buoy_partitions(df)
            bv = bv & pd.Series(idx <= now, index=idx)          # measurements only where they exist
            sets.append(("buoy", bp, bv))
        parts, src = _select(sets, n)
        hs_m, tp, dirm = _offshore_summary(parts)
        _boundary_components(out, sets, src)
        slope = ctx.profile.get("slope")
        res = phys.breaking_from_partitions(parts, spot.facing_deg, ctx.geometry, ctx.boundary_depth_m, slope)
        hb = res["hb_m"]
        r_in = phys.breaking_from_partitions(_rotate(parts, spot.facing_deg, 10), spot.facing_deg, ctx.geometry,
                                             ctx.boundary_depth_m, slope)["hb_m"]
        r_out = phys.breaking_from_partitions(_rotate(parts, spot.facing_deg, -10), spot.facing_deg, ctx.geometry,
                                              ctx.boundary_depth_m, slope)["hb_m"]
        # relative uncertainty of the offshore height
        u = np.full(n, 0.25)
        u = np.where(src == "nws", 0.35, np.where(src == "buoy", 0.30, u))
        if ctx.anchor.get("applied"):
            last = ctx.anchor.get("last_obs")
            if last is not None:
                near = (idx <= last + pd.Timedelta(hours=6)) & (src == "model")
                u = np.where(near, 0.18, u)
        if ctx.anchor.get("inconsistent"):
            u = np.where(src == "model", 0.35, u)
        u = u + 0.10 * np.clip(lead_h / 120, 0, 1)
        u = u + np.where(np.nan_to_num(res["oblique_share"]) > 0.5, 0.08, 0)
        u = u + np.where(np.nan_to_num(res["shadow_share"]) > 0.3, 0.08, 0)
        u = u + (0.03 if slope is None else 0) + (0.05 if ctx.model_stale else 0)
        # Hb scales with offshore height^0.8; the breaker relation itself scatters ~15 %
        lo = np.fmin(np.fmin(r_in, r_out), hb) * (1 - u) ** 0.8 * (1 - U_KOMAR)
        hi = np.fmax(np.fmax(r_in, r_out), hb) * (1 + u) ** 0.8 * (1 + U_KOMAR)
        out["surf_ft"] = hb * FT
        out["surf_lo_ft"] = lo * FT
        out["surf_hi_ft"] = hi * FT
        out["surf_kind"] = np.where(np.isfinite(hb), "breaking", None)
        out["breaker_depth_ft"] = res["hb_depth_m"] * FT
        out["surf_oblique_share"] = res["oblique_share"]
        out["surf_shadow_share"] = res["shadow_share"]
        out["offshore_hs_ft"] = hs_m * FT
        out["offshore_tp_s"] = tp
        out["offshore_dir"] = dirm
        out["surf_source"] = src
        Te = 0.9 * tp
        out["wave_power_kw_m"] = phys.wave_power_kw_m(hs_m, Te)
        if slope:
            xi = phys.iribarren(slope, hb, res["T_eff"])
            out["breaker_type"] = phys.breaker_type(xi)
        else:
            out["breaker_type"] = None
        # where waves break at this tide (outer bar vs. beach)
        bar = ctx.profile.get("bar_depth_m")
        if bar is not None and "tide_ft" in df:
            h_bar = bar + _first(df, "tide_ft").to_numpy() / FT - NAVD88_ABOVE_MLLW_M
            out["bar_depth_ft"] = h_bar * FT
            out["break_zone"] = np.where(~np.isfinite(hb), None,
                                         np.where(res["hb_depth_m"] > h_bar, "outer bar", "near the beach"))
        else:
            out["bar_depth_ft"] = np.nan
            out["break_zone"] = None
        # wave–current interaction at inlet mouths (ebb running against the waves)
        if "inlet" in spot.structure or spot.environment == "inlet":
            cur = _first(df, "current_kt").to_numpy()
            align = np.clip(np.cos(np.radians(phys.angle_off_normal(dirm, spot.facing_deg))), 0, 1)
            u_opp = np.where(cur < 0, -cur * 0.514444 * align, 0.0)
            fac, blocked = phys.opposing_current_factor(np.nan_to_num(res["T_eff"], nan=8.0), u_opp)
            out["inlet_wave_ft"] = np.where(np.isfinite(hb), hb * FT * fac ** 0.8, np.nan)
            out["inlet_ebb_factor"] = np.where(np.isfinite(hb), fac, np.nan)
            out["inlet_blocking"] = blocked & np.isfinite(hb)
    else:
        # bays and the Sound: ocean swell cannot reach the shoreline
        wind_ms = _first(df, "wind_mph").to_numpy() / 2.236936
        wdir = _first(df, "wind_dir").to_numpy()
        # 2-hour vector-mean wind: waves lag rapid wind changes
        if n:
            ue = pd.Series(wind_ms * np.sin(np.radians(wdir)), index=idx).rolling(2, min_periods=1).mean()
            un = pd.Series(wind_ms * np.cos(np.radians(wdir)), index=idx).rolling(2, min_periods=1).mean()
            wind_ms = np.where(np.isfinite(wind_ms), np.hypot(ue, un), np.nan)
            wdir = np.where(np.isfinite(wdir), (np.degrees(np.arctan2(ue, un)) + 360) % 360, np.nan)
        if ctx.geometry is None:
            H = T = np.full(n, np.nan)
            lo = hi = H
            fetch_km = np.full(n, np.nan)
        else:
            fetch_km, depth = ctx.geometry.effective_fetch(np.nan_to_num(wdir))
            H, T = phys.fetch_limited(wind_ms, fetch_km * 1000, depth)
            outs = []
            for dU, dD in ((0.8, 0), (1.2, 0), (1.0, 15), (1.0, -15)):
                f2, d2 = ctx.geometry.effective_fetch(np.nan_to_num(wdir) + dD)
                outs.append(phys.fetch_limited(wind_ms * dU, f2 * 1000, d2)[0])
            lo = np.fmin(np.fmin.reduce(np.vstack(outs), axis=0), H) * (1 - U_SPM)
            hi = np.fmax(np.fmax.reduce(np.vstack(outs), axis=0), H) * (1 + U_SPM)
            H = np.where(np.isfinite(wdir), H, np.nan)
        u = np.full(n, 0.35)
        out["surf_ft"] = H * FT
        out["surf_lo_ft"] = lo * FT
        out["surf_hi_ft"] = hi * FT
        out["surf_kind"] = np.where(np.isfinite(H), "wind chop", None)
        out["chop_period_s"] = T
        out["chop_fetch_km"] = fetch_km
        out["surf_source"] = np.where(np.isfinite(H), "wind (fetch-limited)", None)
        # ocean-only quantities are explicitly missing here (same columns for every spot)
        for c in ("offshore_hs_ft", "offshore_tp_s", "offshore_dir", "surf_oblique_share", "surf_shadow_share",
                  "bc_swell_ft", "bc_swell_tp_s", "bc_swell_dir", "bc_swell2_ft", "bc_swell2_tp_s", "bc_swell2_dir",
                  "bc_windwave_ft", "bc_windwave_tp_s", "bc_windwave_dir", "bar_depth_ft"):
            out[c] = np.nan
        out["breaker_type"] = None
        out["break_zone"] = None
        out["wave_power_kw_m"] = np.nan
        out["breaker_depth_ft"] = np.nan

    # ---- confidence, trend, quality, safety ------------------------------
    s = out["surf_ft"].astype(float)
    half = (out["surf_hi_ft"] - out["surf_lo_ft"]) / (2 * s.where(s > 0.3, 0.3))
    flags = [[] for _ in range(n)]
    # validation against the NWS zone forecast (offshore seas, ocean zones)
    if spot.is_open_ocean and "nws_seas_hi_ft" in df:
        lo_n, hi_n = _first(df, "nws_seas_lo_ft"), _first(df, "nws_seas_occ_ft").combine_first(_first(df, "nws_seas_hi_ft"))
        off = out["offshore_hs_ft"]
        mism = (off.notna() & lo_n.notna() & ((off < lo_n - 1.5) | (off > hi_n + 1.5))).to_numpy()
        for i in np.nonzero(mism)[0]:
            flags[i].append(f"model offshore {off.iloc[i]:.1f} ft vs NWS {lo_n.iloc[i]:.0f}–{hi_n.iloc[i]:.0f} ft")
        half = half + np.where(mism, 0.10, 0)
    if not spot.is_open_ocean and "nws_seas_hi_ft" in df:
        lo_n, hi_n = _first(df, "nws_seas_lo_ft"), _first(df, "nws_seas_hi_ft")
        mism = (s.notna() & hi_n.notna() & ((s > hi_n + 1.5) | (s < lo_n - 1.5))).to_numpy()
        for i in np.nonzero(mism)[0]:
            flags[i].append(f"chop {s.iloc[i]:.1f} ft vs NWS zone waves {lo_n.iloc[i]:.0f}–{hi_n.iloc[i]:.0f} ft")
        half = half + np.where(mism, 0.10, 0)
    if spot.is_open_ocean:
        # implausibly low surf for a sizeable, well-aimed offshore swell (diagnostic only)
        off = out["offshore_hs_ft"].to_numpy()
        aimed = phys.angle_off_normal(out["offshore_dir"].to_numpy(), spot.facing_deg) < 45
        low = (off >= 3) & (np.nan_to_num(out["offshore_tp_s"].to_numpy()) >= 7) & aimed & \
              (np.nan_to_num(out["surf_shadow_share"].to_numpy(), nan=1) < 0.3) & (s.to_numpy() < 0.4 * off)
        for i in np.nonzero(low)[0]:
            flags[i].append("implausibly low vs offshore swell — check inputs")
        half = half + np.where(low, 0.25, 0)
        if ctx.anchor.get("inconsistent"):
            for i in range(n):
                flags[i].append(ctx.anchor.get("reason", "buoy/model inconsistent"))
    out["surf_uncertainty"] = half
    out["surf_confidence"] = np.where(s.isna(), None, np.where(half <= 0.30, "high", np.where(half <= 0.45, "medium", "low")))
    out["surf_flags"] = ["; ".join(f) if f else "" for f in flags]
    base = s if s.notna().any() else out["offshore_hs_ft"].astype(float)
    out["surf_trend"] = phys.trend(base)
    on = onshore.to_numpy()
    out["surf_quality"] = np.where(s.isna() | np.isnan(on), None,
                                   np.where(on < -5, "clean (offshore wind)",
                                            np.where(on > 10, "choppy (onshore wind)", "fair (light / side-shore wind)")))
    out["surf_safety"] = surf_safety(out, spot)
    out["nearshore_wave_ft"] = out["surf_ft"]
    meta["sources_used"] = sorted({x for x in out["surf_source"].dropna().unique()})
    return out, meta


def surf_safety(out: pd.DataFrame, spot: Spot) -> np.ndarray:
    """Shore-angler surf SAFETY (separate from the Bite Index): OK / Caution / Dangerous."""
    hb = out["surf_ft"].to_numpy(dtype=float)
    tp = out["offshore_tp_s"].to_numpy(dtype=float) if "offshore_tp_s" in out else np.full(len(out), np.nan)
    tp = np.nan_to_num(tp, nan=0.0)
    rocks = spot.environment == "jetty" or "jetty" in spot.structure or "rocks" in spot.structure
    inlet = out.get("inlet_wave_ft", pd.Series(np.nan, index=out.index)).to_numpy(dtype=float)
    blocking = out.get("inlet_blocking", pd.Series(False, index=out.index)).fillna(False).to_numpy(dtype=bool)
    h = np.fmax(hb, np.nan_to_num(inlet, nan=-1))
    danger = (h >= 6) | (rocks & (h >= 4) & (tp >= 9)) | (blocking & (h >= 3))
    caution = (h >= 3) | (rocks & (h >= 2.5) & (tp >= 8)) | (np.nan_to_num(inlet) >= 2.5)
    res = np.where(danger, "Dangerous", np.where(caution, "Caution", "OK"))
    return np.where(np.isnan(hb), None, res)
