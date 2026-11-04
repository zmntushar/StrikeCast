"""Continuous fixed-length window search over a regular score series.

Every candidate start (5-minute steps) is scored over the WHOLE interval with
trapezoidal averaging, not just its midpoint. The best window is the global
maximum; the runner-up is the best window that does not overlap it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW = pd.Timedelta(minutes=90)
STEP = pd.Timedelta(minutes=5)
MIN_COVERAGE = 0.8   # share of samples that must have a score


@dataclass(frozen=True)
class WindowPick:
    start: pd.Timestamp
    end: pd.Timestamp
    score: float          # 0-100 mean suitability over the interval
    rank: int = 0         # 1 = best, 2 = second best, 0 = user comparison

    @property
    def minutes(self) -> float:
        return (self.end - self.start) / pd.Timedelta(minutes=1)


def interval_score(score: pd.Series, start: pd.Timestamp, length: pd.Timedelta = WINDOW) -> float:
    """Trapezoid-weighted mean of ``score`` over [start, start + length]; NaN if coverage is too thin."""
    seg = score.loc[start:start + length]
    v = seg.to_numpy(dtype=float)
    n_expected = int(length / _step(score)) + 1
    ok = ~np.isnan(v)
    if len(v) < n_expected or ok.sum() < MIN_COVERAGE * n_expected:
        return float("nan")
    w = np.ones(len(v))
    w[0] = w[-1] = 0.5
    w = np.where(ok, w, 0)
    return float(np.sum(np.nan_to_num(v) * w) / np.sum(w))


def _all_scores(score: pd.Series, length: pd.Timedelta) -> np.ndarray:
    """Window score for every grid position at once (same math as ``interval_score``, vectorized)."""
    v = score.to_numpy(dtype=float)
    n = int(length / _step(score)) + 1
    out = np.full(len(v), np.nan)
    if len(v) < n:
        return out
    win = np.lib.stride_tricks.sliding_window_view(v, n)
    ok = ~np.isnan(win)
    w = np.ones(n)
    w[0] = w[-1] = 0.5
    wts = ok * w
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.sum(np.nan_to_num(win) * wts, axis=1) / np.sum(wts, axis=1)
    vals[ok.sum(axis=1) < MIN_COVERAGE * n] = np.nan
    out[:len(vals)] = vals
    return out


def candidate_scores(score: pd.Series, starts: pd.DatetimeIndex, length: pd.Timedelta = WINDOW) -> pd.Series:
    """Window score for every candidate start."""
    pos = score.index.get_indexer(starts)
    allv = _all_scores(score, length)
    vals = np.where(pos >= 0, allv[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=starts)


def rank_windows(score: pd.Series, starts: pd.DatetimeIndex, length: pd.Timedelta = WINDOW, k: int = 2) -> list[WindowPick]:
    """Best ``k`` mutually non-overlapping windows whose start is in ``starts``."""
    vals = candidate_scores(score, starts, length).to_numpy()
    order = [i for i in np.argsort(-np.nan_to_num(vals, nan=-1), kind="stable") if not np.isnan(vals[i])]
    picks: list[WindowPick] = []
    for i in order:
        s = starts[i]
        if all(abs(s - p.start) >= length for p in picks):
            picks.append(WindowPick(start=s, end=s + length, score=round(float(vals[i]), 1), rank=len(picks) + 1))
            if len(picks) == k:
                break
    return picks


def _step(s: pd.Series) -> pd.Timedelta:
    if len(s.index) < 2:
        return STEP
    return pd.Timedelta(s.index[1] - s.index[0])
