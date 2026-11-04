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


def rank_windows(score: pd.Series, starts: pd.DatetimeIndex, length: pd.Timedelta = WINDOW, k: int = 2) -> list[WindowPick]:
    """Best ``k`` mutually non-overlapping windows whose start is in ``starts``."""
    vals = np.array([interval_score(score, s, length) for s in starts])
    order = [i for i in np.argsort(-np.nan_to_num(vals, nan=-1), kind="stable") if not np.isnan(vals[i])]
    picks: list[WindowPick] = []
    for i in order:
        s = starts[i]
        if all(abs(s - p.start) >= length for p in picks):
            picks.append(WindowPick(start=s, end=s + length, score=round(float(vals[i]), 1), rank=len(picks) + 1))
            if len(picks) == k:
                break
    return picks


def candidate_scores(score: pd.Series, starts: pd.DatetimeIndex, length: pd.Timedelta = WINDOW) -> pd.Series:
    """Window score for every candidate start (for charts / diagnostics)."""
    return pd.Series([interval_score(score, s, length) for s in starts], index=starts)


def _step(s: pd.Series) -> pd.Timedelta:
    if len(s.index) < 2:
        return STEP
    return pd.Timedelta(s.index[1] - s.index[0])
