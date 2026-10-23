"""Hourly tide and current features from official NOAA high/low and max/slack events.

Interpolation between official events uses the standard cosine (half-sine) form.
Max current timing comes from NOAA current predictions when available — we never
assume max flow sits exactly halfway between high and low water. Values produced
here are tagged ``interpolated`` (between official events) or ``proxy`` when the
current is inferred from the rate of tide change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.schemas.env import CurrentEvent, TideEvent


def _hours(index: pd.DatetimeIndex) -> np.ndarray:
    """Epoch hours, independent of the index's datetime resolution (ns/us/ms)."""
    return pd.DatetimeIndex(index).as_unit("ns").asi8 / 3.6e12


def _event_arrays(times, values):
    t = np.array([pd.Timestamp(x).as_unit("ns").value for x in times], dtype="int64") / 3.6e12  # hours
    return t, np.asarray(values, dtype=float)


def tide_features(events: list[TideEvent], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Height, rate, stage and turning-point timing for every timestamp in ``index``."""
    cols = ["tide_ft", "tide_rate_ft_h", "tide_dir", "tide_frac", "tide_stage", "tide_stage_tag",
            "hours_since_turn", "hours_to_turn", "tide_range_ft", "next_turn_type"]
    if len(events) < 2:
        return pd.DataFrame(index=index, columns=cols, dtype=object)
    ev = sorted(events, key=lambda e: e.time)
    et, eh = _event_arrays([e.time for e in ev], [e.height_ft for e in ev])
    etype = [e.type for e in ev]
    th = _hours(index)
    k = np.searchsorted(et, th, side="right") - 1
    valid = (k >= 0) & (k < len(et) - 1)
    k = np.clip(k, 0, len(et) - 2)
    t0, t1, h0, h1 = et[k], et[k + 1], eh[k], eh[k + 1]
    span = np.maximum(t1 - t0, 1e-6)
    f = np.clip((th - t0) / span, 0, 1)
    height = h0 + (h1 - h0) * (1 - np.cos(np.pi * f)) / 2
    rate = (h1 - h0) * np.pi / (2 * span) * np.sin(np.pi * f)
    incoming = np.array([etype[i] == "L" for i in k])
    since = th - t0
    until = t1 - th

    stage = np.empty(len(th), dtype=object)
    tag = np.empty(len(th), dtype=object)
    for n in range(len(th)):
        if not valid[n]:
            stage[n] = tag[n] = None
            continue
        third = "early" if f[n] < 1 / 3 else ("mid" if f[n] < 2 / 3 else "late")
        if incoming[n]:
            stage[n] = f"{third} incoming"
            tag[n] = {"early": "low_incoming", "mid": "mid_incoming", "late": "high_incoming"}[third]
        else:
            stage[n] = f"{third} outgoing"
            tag[n] = {"early": "high_outgoing", "mid": "mid_outgoing", "late": "low_outgoing"}[third]
        if since[n] < 0.6:
            stage[n] = ("low" if incoming[n] else "high") + " slack"
        elif until[n] < 0.6:
            stage[n] = ("high" if incoming[n] else "low") + " slack"

    df = pd.DataFrame({
        "tide_ft": np.where(valid, height, np.nan),
        "tide_rate_ft_h": np.where(valid, rate, np.nan),
        "tide_dir": np.where(valid, np.where(incoming, "incoming", "outgoing"), None),
        "tide_frac": np.where(valid, f, np.nan),
        "tide_stage": stage, "tide_stage_tag": tag,
        "hours_since_turn": np.where(valid, since, np.nan),
        "hours_to_turn": np.where(valid, until, np.nan),
        "tide_range_ft": np.where(valid, np.abs(h1 - h0), np.nan),
        "next_turn_type": np.where(valid, np.where(incoming, "high", "low"), None),
    }, index=index)
    return df


def current_features(events: list[CurrentEvent], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Signed current speed (kt, + flood) and share of the local peak (0-1)."""
    if len(events) < 2:
        return pd.DataFrame(index=index, columns=["current_kt", "current_norm", "current_phase"], dtype=float)
    ev = sorted(events, key=lambda e: e.time)
    et, ev_v = _event_arrays([e.time for e in ev], [e.velocity_kt for e in ev])
    th = _hours(index)
    k = np.clip(np.searchsorted(et, th, side="right") - 1, 0, len(et) - 2)
    valid = (th >= et[0]) & (th <= et[-1])
    t0, t1, v0, v1 = et[k], et[k + 1], ev_v[k], ev_v[k + 1]
    f = np.clip((th - t0) / np.maximum(t1 - t0, 1e-6), 0, 1)
    v = v0 + (v1 - v0) * (1 - np.cos(np.pi * f)) / 2
    # local peak = largest |max event| within +/- 7 h
    peaks = np.array([np.max(np.abs(ev_v[(et >= x - 7) & (et <= x + 7)]), initial=0.0) for x in th])
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(peaks > 0.05, np.abs(v) / peaks, np.nan)
    phase = np.where(np.abs(v) < 0.25, "slack", np.where(v > 0, "flood", "ebb"))
    return pd.DataFrame({
        "current_kt": np.where(valid, v, np.nan),
        "current_norm": np.where(valid, np.clip(norm, 0, 1), np.nan),
        "current_phase": np.where(valid, phase, None),
    }, index=index)


def current_proxy_from_tide(tide: pd.DataFrame) -> pd.Series:
    """Share of peak flow inferred from the rate of tide change (labeled proxy)."""
    rate = tide["tide_rate_ft_h"].astype(float).abs()
    roll = rate.rolling(13, center=True, min_periods=1).max()
    return (rate / roll.replace(0, np.nan)).clip(0, 1)
