"""Fishing-session feedback: store what the window optimizer predicted for each
logged trip, and evaluate the predictions against real outcomes.

Ground rules
------------
* An unsuccessful trip is NOT proof that the prediction was wrong — fish can be
  present and not biting, and tackle, technique, effort and luck all matter.
  Only patterns across many sessions are informative, so every metric here is
  descriptive and gated on sample size.
* Nothing changes the species weights automatically. ``component_signals`` only
  reports which components tracked your results, as input for future calibration.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.features.current_model import SOURCE_TEXT
from app.models.feeding_profiles import COMPONENTS
from app.models.window_optimizer import WindowPick

MODEL_VERSION = "window-v1"
MIN_SESSIONS_FOR_STATS = 8
MIN_SESSIONS_FOR_SIGNALS = 20


def _iso(ts: pd.Timestamp | None) -> str | None:
    return pd.Timestamp(ts).isoformat() if ts is not None else None


def _pick(p: WindowPick | None) -> dict[str, Any] | None:
    return {"start": _iso(p.start), "end": _iso(p.end), "score": p.score} if p else None


def session_prediction(result, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> dict[str, Any]:
    """Snapshot of the optimizer's output for the day plus scores for the interval actually fished."""
    s = pd.Timestamp(start_utc).tz_convert("UTC")
    e = pd.Timestamp(end_utc).tz_convert("UTC")
    seg = result.frame.loc[s:e]
    fished = float(np.nanmean(seg["total"])) if not seg.empty and seg["total"].notna().any() else None
    comps = {c: (float(np.nanmean(seg[c])) if not seg.empty and seg[c].notna().any() else None) for c in COMPONENTS}
    overlap = 0.0
    if result.best is not None:
        a, b = max(s, result.best.start), min(e, result.best.end)
        overlap = max((b - a) / pd.Timedelta(minutes=1), 0.0)
    return {
        "model_version": MODEL_VERSION, "species": result.species, "day": str(result.day),
        "best": _pick(result.best), "second": _pick(result.second),
        "fished_score": round(fished, 1) if fished is not None else None,
        "fished_components": comps, "overlap_best_min": round(overlap),
        "confidence": result.confidence.get("score"), "confidence_label": result.confidence.get("label"),
        "current_source": result.current_source, "current_source_text": SOURCE_TEXT.get(result.current_source),
        "weights": result.profile.weights, "computed_at": _iso(pd.Timestamp(result.computed_at)),
    }


def calibration_frame(sessions: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per session: predicted scores/components and observed outcome rates."""
    rows = []
    for r in sessions:
        p = r.get("prediction") or {}
        hours = r.get("duration_h") or 0
        if not hours and r.get("end_utc") and r.get("start_utc"):
            hours = (pd.Timestamp(r["end_utc"]) - pd.Timestamp(r["start_utc"])) / pd.Timedelta(hours=1)
        hours = max(float(hours or 0), 0.25)
        bites = r.get("bites")
        row = {"id": r.get("id"), "start_utc": r.get("start_utc"), "spot_id": r.get("spot_id"), "species": r.get("species"),
               "hours": hours, "bites": bites, "catches": r.get("fish_caught") or 0,
               "bites_per_h": (bites / hours) if bites is not None else np.nan,
               "catches_per_h": (r.get("fish_caught") or 0) / hours,
               "any_action": int(((bites or 0) + (r.get("fish_caught") or 0)) > 0),
               "fished_score": p.get("fished_score"), "best_score": (p.get("best") or {}).get("score"),
               "overlap_best_min": p.get("overlap_best_min") or 0, "confidence": p.get("confidence"),
               "current_source": p.get("current_source")}
        for c in COMPONENTS:
            row[f"c_{c}"] = (p.get("fished_components") or {}).get(c)
        rows.append(row)
    return pd.DataFrame(rows)


def _rate(df: pd.DataFrame) -> pd.Series:
    """Action per hour: bites when logged, otherwise catches."""
    return df["bites_per_h"].where(df["bites_per_h"].notna(), df["catches_per_h"])


def evaluate(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    df = calibration_frame(sessions)
    out: dict[str, Any] = {"n": len(df), "ready": len(df) >= MIN_SESSIONS_FOR_STATS, "bands": [], "messages": []}
    out["messages"].append("Skunked trips are kept as data but are not treated as proof the forecast was wrong — "
                           "fish can be present and not feeding, and tackle, effort and luck matter.")
    if df.empty:
        out["messages"].append("No sessions with saved predictions yet. Log trips to start evaluating the model.")
        return out
    df = df[df["fished_score"].notna()].copy()
    out["n_scored"] = len(df)
    if df.empty:
        return out
    df["rate"] = _rate(df)
    bands = [(0, 50, "< 50"), (50, 70, "50–69"), (70, 101, "70+")]
    for lo, hi, lab in bands:
        b = df[(df["fished_score"] >= lo) & (df["fished_score"] < hi)]
        out["bands"].append({"band": lab, "sessions": len(b),
                             "action_share": float(b["any_action"].mean()) if len(b) else None,
                             "rate_per_h": float(b["rate"].mean()) if len(b) else None})
    inside = df[df["overlap_best_min"] >= 45]
    outside = df[df["overlap_best_min"] < 45]
    out["inside_best"] = {"sessions": len(inside), "rate_per_h": float(inside["rate"].mean()) if len(inside) else None}
    out["outside_best"] = {"sessions": len(outside), "rate_per_h": float(outside["rate"].mean()) if len(outside) else None}
    if len(df) >= MIN_SESSIONS_FOR_STATS and df["fished_score"].nunique() > 1 and df["rate"].nunique() > 1:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(df["fished_score"], df["rate"])
        out["spearman"] = {"rho": float(rho), "p": float(pval)}
        out["messages"].append(
            f"Rank correlation between predicted suitability and your action rate: ρ = {rho:+.2f} "
            f"(p = {pval:.2f}, {len(df)} sessions). " +
            ("Too early to draw conclusions." if len(df) < 30 or pval > 0.1 else
             "Higher-scored windows have tended to fish better." if rho > 0 else
             "Higher scores have NOT matched your results so far — worth reviewing the species weights."))
    else:
        out["messages"].append(f"Correlation needs at least {MIN_SESSIONS_FOR_STATS} scored sessions "
                               f"(you have {len(df)}).")
    return out


def component_signals(sessions: list[dict[str, Any]]) -> pd.DataFrame | None:
    """Advisory: rank correlation of each component (during the fished interval) with your action rate.

    Returns None until there are enough sessions. Never applied automatically.
    """
    df = calibration_frame(sessions)
    df = df[df["fished_score"].notna()]
    if len(df) < MIN_SESSIONS_FOR_SIGNALS:
        return None
    from scipy.stats import spearmanr
    rate = _rate(df)
    rows = []
    for c in COMPONENTS:
        x = df[f"c_{c}"].astype(float)
        ok = x.notna() & rate.notna()
        if ok.sum() < MIN_SESSIONS_FOR_SIGNALS or x[ok].nunique() < 3:
            continue
        rho, p = spearmanr(x[ok], rate[ok])
        rows.append({"component": c, "rho": float(rho), "p": float(p), "sessions": int(ok.sum())})
    return pd.DataFrame(rows)
