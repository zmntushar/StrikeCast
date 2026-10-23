"""Nearshore wave physics for shore anglers.

Everything here is plain linear wave theory plus standard coastal-engineering
relationships; sources are cited inline and summarised in docs/SURF_MODEL.md.

Terms (kept strictly separate throughout StrikeCast)
----------------------------------------------------
* Offshore significant wave height Hs — mean of the highest third of waves, as
  measured by a buoy or predicted by a wave model, in water 20-40 m deep.
* Swell / wind-wave partitions — the long-period (remote) and short-period
  (local wind) parts of the same sea state. Their ENERGIES add
  (Hs² = Hswell² + Hwind²); their heights never do.
* Breaking wave height Hb ("surf") — significant height where waves break near
  the beach, after shoaling, refraction, sheltering and depth-limited breaking.
* Wind chop — fetch-limited, locally generated waves in bays and the Sound.

Ocean beaches
-------------
1. Each partition is spread over direction with a wrapped-normal spectrum
   (σ ≈ 30° wind sea, 15° swell — Goda 2000 / CEM II-1 typical values).
2. Each directional bin is checked against the spot's ray-cast geometry: land
   between the offshore point and the beach blocks it; bins just inside a
   shadow keep energy by Sommerfeld far-field diffraction.
3. Energy flux E·Cg·cosα is conserved from the offshore depth to breaking over
   straight, parallel contours (linear shoaling + Snell refraction).
4. Breaking height uses the Komar & Gaughan (1972) relation generalised for
   obliquity, Hb = 0.39 g^⅕ (T H0² cosα0 / cosαb)^⅖, with αb from Snell's law
   at the breaker depth hb = Hb/γb (Weggel 1972 breaker index from beach slope).
5. Bins are combined by summing breaking energy flux (∝ Hb^5/2).

Sheltered water (bays, Long Island Sound)
-----------------------------------------
Ocean swell cannot reach these shorelines, so surf is the fetch-limited wind
sea from the SPM (1984) shallow-water growth equations, using the over-water
fetch and mean depth in the direction the wind blows from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

G = 9.80665
FT_PER_M = 3.28084
RHO = 1025.0
KOMAR_K = 0.39                 # Komar & Gaughan (1972), field-calibrated
ADMIT_KM = 10.0                # land beyond this distance cannot shadow the beach from the offshore point
SIGMA_WIND_SEA = 30.0          # directional spread (deg), wind sea
SIGMA_SWELL = 15.0             # directional spread (deg), swell
BIN_STEP = 5.0


# ------------------------------------------------------------ linear theory
def wavenumber(T, h):
    """k (rad/m) from ω² = g k tanh(kh); h=inf/NaN → deep water. Vectorized."""
    T = np.asarray(T, dtype=float)
    h = np.asarray(h, dtype=float)
    w = 2 * np.pi / np.where(T > 0, T, np.nan)
    k0 = w ** 2 / G
    deep = ~np.isfinite(h) | (h <= 0) | (k0 * np.where(np.isfinite(h), h, 0) > 10)
    hh = np.where(deep, 1.0, h)
    k = k0 / np.sqrt(np.tanh(k0 * hh))          # Eckart (1952) first guess
    for _ in range(4):                            # Newton refinement
        f = G * k * np.tanh(k * hh) - w ** 2
        df = G * np.tanh(k * hh) + G * k * hh / np.cosh(np.clip(k * hh, 0, 50)) ** 2
        k = k - f / df
    return np.where(deep, k0, k)


def celerity(T, h):
    """(C, Cg) phase and group speed (m/s)."""
    T = np.asarray(T, dtype=float)
    k = wavenumber(T, h)
    hh = np.asarray(h, dtype=float)
    C = (2 * np.pi / T) / k
    kh2 = 2 * k * np.where(np.isfinite(hh) & (hh > 0), hh, np.inf)
    n = np.where(kh2 > 50, 0.5, 0.5 * (1 + kh2 / np.sinh(np.clip(kh2, 1e-9, 50))))
    return C, n * C


def deep_water(T):
    C0 = G * np.asarray(T, dtype=float) / (2 * np.pi)
    return C0, 0.5 * C0


def breaker_index(slope, Hb, T):
    """Weggel (1972) γb = b − a·Hb/(gT²); 0.78 (McCowan) on a flat bottom."""
    m = np.clip(np.nan_to_num(np.asarray(slope, dtype=float), nan=0.0), 0.0, 0.1)
    a = 43.8 * (1 - np.exp(-19 * m))
    b = 1.56 / (1 + np.exp(-19.5 * m))
    g_ = b - a * np.asarray(Hb, dtype=float) / (G * np.asarray(T, dtype=float) ** 2)
    return np.clip(np.where(m > 0, g_, 0.78), 0.6, 1.2)


# -------------------------------------------------------- directional bins
def spread_offsets(sigma: float, step: float = BIN_STEP, n_sigma: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    off = np.arange(-n_sigma * sigma, n_sigma * sigma + 1e-9, step)
    w = np.exp(-0.5 * (off / sigma) ** 2)
    return off, w / w.sum()


def angle_off_normal(from_deg, facing_deg):
    """Absolute angle (deg) between the direction waves come FROM and the shore normal."""
    return np.abs(((np.asarray(from_deg, dtype=float) - facing_deg + 180) % 360) - 180)


@dataclass
class Geometry:
    """Ray-cast exposure for one spot (see geography.exposure_rays)."""

    dirs: np.ndarray            # ray directions (deg FROM)
    fetch_km: np.ndarray
    depth_m: np.ndarray
    open_ocean: np.ndarray
    edge_depth_m: np.ndarray | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        edge = d.get("edge_depth_m")
        return cls(np.asarray(d["dirs_deg"], float), np.asarray(d["fetch_km"], float),
                   np.asarray(d["mean_depth_m"], float), np.asarray(d["open_ocean"], bool),
                   np.asarray(edge, float) if edge is not None else None)

    def _nearest(self, deg):
        step = 360.0 / len(self.dirs)
        return (np.round(np.asarray(deg, float) % 360 / step).astype(int)) % len(self.dirs)

    def fetch_at(self, deg):
        return self.fetch_km[self._nearest(deg)]

    def admitted(self) -> np.ndarray:
        return self.fetch_km >= ADMIT_KM

    def obstacle_depth(self, facing_deg: float) -> float:
        """Water depth (m) where waves pass the edge of the land that shelters the spot:
        the depth just off the land on blocked rays next to the open sector (the
        tip of a jetty or spit), used to refract waves before the shadow test."""
        adm = self.admitted()
        rel = np.abs(((self.dirs - facing_deg + 180) % 360) - 180)
        near_open = np.array([adm[(i - 1) % len(adm)] or adm[(i + 1) % len(adm)] or
                              adm[(i - 2) % len(adm)] or adm[(i + 2) % len(adm)] for i in range(len(adm))])
        src = self.edge_depth_m if self.edge_depth_m is not None else self.depth_m
        sel = (~adm) & near_open & (rel < 90) & (self.fetch_km > 0) & (src > 0)
        return float(np.clip(np.median(src[sel]), 2.0, 20.0)) if sel.any() else 5.0

    def effective_fetch(self, wind_from_deg, half_width: float = 15.0):
        """(fetch_km, mean_depth_m) averaged over rays within ±half_width of the wind (CEM II-2)."""
        wd = np.asarray(wind_from_deg, dtype=float)
        offs = np.arange(-half_width, half_width + 0.1, 5.0)
        idx = self._nearest(wd[..., None] + offs)
        f = self.fetch_km[idx]
        wet = f > 0
        dsum = np.where(wet, self.depth_m[idx], 0.0).sum(axis=-1)
        cnt = wet.sum(axis=-1)
        depth = np.where(cnt > 0, dsum / np.maximum(cnt, 1), 0.0)
        return f.mean(axis=-1), depth

    def diffraction_weight(self, bin_deg, k):
        """Energy share reaching the spot from direction bin_deg (1 = unobstructed).

        Blocked directions near the edge of an obstacle keep energy by Sommerfeld
        far-field diffraction: Kd ≈ 1 / (2 √(2π k r) · sin(β/2)), capped at the
        shadow-line value 0.5, where β is the angle into the shadow and r the
        distance to the obstacle. Both edges of a short obstacle contribute.
        """
        adm = self.admitted()
        n = len(self.dirs)
        step = 360.0 / n
        b = np.asarray(bin_deg, dtype=float) % 360
        idx = self._nearest(b)
        out = np.where(adm[idx], 1.0, 0.0)
        if adm.all() or not adm.any():
            return out
        # angular distance to the nearest admitted ray on each side
        adm_dirs = self.dirs[adm]
        diff = (adm_dirs[None, :] - b.reshape(-1)[:, None] + 540) % 360 - 180
        cw = np.where(diff > 0, diff, np.inf).min(axis=1)
        ccw = np.where(diff < 0, -diff, np.inf).min(axis=1)
        r_m = np.maximum(self.fetch_km[idx].reshape(-1), 0.2) * 1000
        kk = np.broadcast_to(np.asarray(k, dtype=float), b.shape).reshape(-1)
        energy = np.zeros_like(r_m)
        for d in (cw, ccw):
            beta = np.radians(np.clip(d - step / 2, 0, 180))
            with np.errstate(divide="ignore", invalid="ignore"):
                kd = 1.0 / (2 * np.sqrt(2 * np.pi * kk * r_m) * np.sin(beta / 2))
            kd = np.where(np.isfinite(d), np.minimum(np.nan_to_num(kd, nan=0.5, posinf=0.5), 0.5), 0.0)
            energy += kd ** 2
        blocked = ~adm[idx].reshape(-1)
        res = out.reshape(-1).copy()
        res[blocked] = np.minimum(energy[blocked], 1.0)
        return res.reshape(b.shape)


# --------------------------------------------------------------- breaking
def breaking_from_partitions(parts: list[dict], facing_deg: float, geom: Geometry | None,
                             h_boundary: float | None, slope: float | None = None) -> dict[str, np.ndarray]:
    """Significant breaking height (m) at an ocean beach from offshore wave partitions.

    ``parts``: dicts of equal-length arrays ``H`` (m), ``T`` (s), ``dir`` (deg FROM)
    and a scalar ``sigma`` (deg). ``h_boundary``: water depth (m) where the
    partitions are valid (None = deep water).
    """
    n = len(parts[0]["H"]) if parts else 0
    flux = np.zeros(n)                      # Σ Hb_bin^(5/2) — breaking energy flux, up to a constant
    flux_oblique = np.zeros(n)              # share arriving > 60° off normal (sensitivity / confidence)
    flux_shadow = np.zeros(n)               # share that only arrives by diffraction
    any_valid = np.zeros(n, dtype=bool)
    hb_guess = np.full(n, 1.0)
    T_eff_num = np.zeros(n)
    h1 = h_boundary if h_boundary and h_boundary > 0 else np.inf
    for _ in range(3):                      # iterate the breaker depth used for αb
        flux[:] = 0; flux_oblique[:] = 0; flux_shadow[:] = 0; T_eff_num[:] = 0
        for p in parts:
            H = np.asarray(p["H"], dtype=float)
            T = np.asarray(p["T"], dtype=float)
            D = np.asarray(p["dir"], dtype=float)
            ok = np.isfinite(H) & np.isfinite(T) & np.isfinite(D) & (T > 0) & (H >= 0)
            any_valid |= ok
            Hs, Ts, Ds = np.where(ok, H, 0.0), np.where(ok, T, 8.0), np.where(ok, D, 0.0)
            C1, Cg1 = celerity(Ts, h1)
            C0, Cg0 = deep_water(Ts)
            if geom is not None:
                h_obs = geom.obstacle_depth(facing_deg)
                C_obs, _ = celerity(Ts, h_obs)
                k_obs = wavenumber(Ts, h_obs)
            offs, w = spread_offsets(p.get("sigma", SIGMA_WIND_SEA))
            for off, wi in zip(offs, w):
                bdir = Ds + off
                signed = ((bdir - facing_deg + 180) % 360) - 180
                a1 = np.radians(np.abs(signed))
                incoming = a1 < np.pi / 2
                if geom is not None:
                    # Waves refract toward the shore normal before they reach the
                    # sheltering land (Snell at the obstacle depth); test the shadow there.
                    a_obs = np.degrees(np.arcsin(np.clip(np.sin(a1) * C_obs / C1, -1, 1)))
                    dir_obs = facing_deg + np.sign(signed) * a_obs
                    adm = geom.diffraction_weight(dir_obs, k_obs)
                else:
                    adm = np.ones(n)
                e = wi * adm * Hs ** 2                                  # energy share at the boundary
                sin0 = np.clip(np.sin(a1) * C0 / C1, -1, 1)             # back-refract to deep water
                H0sq_cos0 = e * (Cg1 / Cg0) * np.cos(a1)                # flux conservation: H0² cosα0
                # Snell's law to the breaker depth, with the full dispersion relation
                # (short wind waves are not in shallow water there, so C_b < C0 always)
                Cb, _ = celerity(Ts, hb_guess / 0.78)
                sinb = np.clip(sin0 * Cb / C0, -1, 1)
                cosb = np.sqrt(1 - sinb ** 2)
                fb = np.where(incoming & ok, H0sq_cos0 * Ts / np.maximum(cosb, 0.05), 0.0)
                hb_bin25 = (KOMAR_K * G ** 0.2) ** 2.5 * fb               # Hb_bin^(5/2)
                flux += hb_bin25
                T_eff_num += hb_bin25 * Ts
                flux_oblique += np.where(np.degrees(a1) > 60, hb_bin25, 0.0)
                flux_shadow += np.where(adm < 1.0, hb_bin25, 0.0)
        hb_new = flux ** 0.4
        hb_guess = np.where(hb_new > 0.05, hb_new, 0.05)
    hb = np.where(any_valid, flux ** 0.4, np.nan)
    T_eff = np.where(flux > 0, T_eff_num / np.maximum(flux, 1e-12), np.nan)
    gamma = breaker_index(slope if slope is not None else np.nan, np.nan_to_num(hb), np.nan_to_num(T_eff, nan=8.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        obl = np.where(flux > 0, flux_oblique / flux, np.nan)
        shd = np.where(flux > 0, flux_shadow / flux, np.nan)
    return {"hb_m": hb, "hb_depth_m": hb / gamma, "gamma": gamma, "T_eff": T_eff,
            "oblique_share": obl, "shadow_share": shd}


# ------------------------------------------------------------- wind chop
def fetch_limited(U10_ms, fetch_m, depth_m):
    """SPM (1984) shallow-water fetch-limited growth → (Hs m, Tp s).

    U_A = 0.71 U10^1.23 (wind-stress factor). Deep-water limit is
    gH/U_A² = 1.6e-3 (gF/U_A²)^½.
    """
    U = np.clip(np.asarray(U10_ms, dtype=float), 0, None)
    F = np.clip(np.asarray(fetch_m, dtype=float), 0, None)
    d = np.clip(np.nan_to_num(np.asarray(depth_m, dtype=float), nan=0.0), 0, None)
    UA = 0.71 * U ** 1.23
    with np.errstate(divide="ignore", invalid="ignore"):
        Fd = np.where(UA > 0, G * F / UA ** 2, 0.0)
        dd = np.where(UA > 0, G * np.where(d > 0, d, 50.0) / UA ** 2, 0.0)
        a = np.tanh(0.530 * dd ** 0.75)
        b = np.tanh(0.833 * dd ** 0.375)
        H = np.where(UA > 0, 0.283 * a * np.tanh(0.00565 * Fd ** 0.5 / np.where(a > 0, a, 1)) * UA ** 2 / G, 0.0)
        T = np.where(UA > 0, 7.54 * b * np.tanh(0.0379 * Fd ** (1 / 3) / np.where(b > 0, b, 1)) * UA / G, np.nan)
    H = np.where((F > 0) & (U > 0), H, 0.0)
    T = np.where(H > 0, T, np.nan)
    nan = ~np.isfinite(np.asarray(U10_ms, dtype=float))
    return np.where(nan, np.nan, H), np.where(nan, np.nan, T)


# ---------------------------------------------------- wave-current, power
def opposing_current_factor(T, U_opposing_ms):
    """Wave-height change when waves meet an opposing current (Longuet-Higgins &
    Stewart 1961, deep water): H/H0 = c0 / √(c (c − 2U)), c/c0 = ½(1+√(1−4U/c0)),
    U ≥ 0 opposing. Returns (factor, blocked)."""
    T = np.asarray(T, dtype=float)
    U = np.clip(np.nan_to_num(np.asarray(U_opposing_ms, dtype=float)), 0, None)
    c0 = G * T / (2 * np.pi)
    disc = 1 - 4 * U / c0
    blocked = disc <= 0.05
    c = 0.5 * c0 * (1 + np.sqrt(np.clip(disc, 0.05, None)))
    fac = c0 / np.sqrt(c * np.clip(c - 2 * U, 1e-3, None))
    return np.where(np.isfinite(T), np.clip(fac, 1.0, 2.0), np.nan), blocked & np.isfinite(T)


def wave_power_kw_m(Hs_m, Te_s):
    """Deep-water wave energy flux per metre of crest, P = ρ g² Hs² Te / (64π) (kW/m)."""
    return RHO * G ** 2 * np.asarray(Hs_m, float) ** 2 * np.asarray(Te_s, float) / (64 * np.pi) / 1000


def iribarren(slope, Hb_m, T):
    L0 = G * np.asarray(T, float) ** 2 / (2 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(slope, float) / np.sqrt(np.asarray(Hb_m, float) / L0)


def breaker_type(xi) -> np.ndarray:
    xi = np.asarray(xi, float)
    return np.where(~np.isfinite(xi), None, np.where(xi < 0.4, "spilling", np.where(xi < 2.0, "plunging", "surging")))


# ---------------------------------------------------------------- helpers
def combine_hs(*hs):
    """Total Hs from partitions: energies add, heights do not."""
    arr = np.vstack([np.nan_to_num(np.asarray(h, float)) ** 2 for h in hs])
    tot = np.sqrt(arr.sum(axis=0))
    allnan = np.all(np.vstack([~np.isfinite(np.asarray(h, float)) for h in hs]), axis=0)
    return np.where(allnan, np.nan, tot)


def trend(series: pd.Series, ahead_h: int = 6, rel: float = 0.12, abs_ft: float = 0.25) -> pd.Series:
    """building / steady / subsiding from the change over the next few hours."""
    s = series.astype(float)
    fut = s.shift(-ahead_h).rolling(3, min_periods=1).mean()
    past = s - s.shift(ahead_h)
    delta = (fut - s).where(fut.notna(), past)
    thr = np.maximum(abs_ft, rel * s.abs())
    out = np.where(delta > thr, "building", np.where(delta < -thr, "subsiding", "steady"))
    return pd.Series(np.where(s.isna() | delta.isna(), None, out), index=s.index)
