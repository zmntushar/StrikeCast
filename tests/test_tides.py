from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.features.tide_features import current_features, current_proxy_from_tide, tide_features
from app.schemas.env import CurrentEvent, TideEvent


def _t(h, m=0):
    return datetime(2026, 10, 23, h, m, tzinfo=timezone.utc)


EVENTS = [TideEvent(time=_t(0), height_ft=0.2, type="L"), TideEvent(time=_t(6), height_ft=3.0, type="H"),
          TideEvent(time=_t(12), height_ft=0.4, type="L"), TideEvent(time=_t(18), height_ft=2.8, type="H")]


def idx(n=19):
    return pd.date_range(_t(0), periods=n, freq="1h", tz="UTC")


def test_tide_curve_hits_official_events():
    df = tide_features(EVENTS, idx())
    assert df.loc[_t(6), "tide_ft"] == pytest.approx(3.0)
    assert df.loc[_t(12), "tide_ft"] == pytest.approx(0.4)
    # midway between low and high the cosine curve is at the mean height
    assert df.loc[_t(3), "tide_ft"] == pytest.approx((0.2 + 3.0) / 2)


def test_tide_stage_and_turn_timing():
    df = tide_features(EVENTS, idx())
    assert df.loc[_t(1), "tide_dir"] == "incoming"
    assert df.loc[_t(1), "tide_stage_tag"] == "low_incoming"
    assert df.loc[_t(8), "tide_dir"] == "outgoing"
    assert df.loc[_t(8), "hours_since_turn"] == pytest.approx(2)
    assert df.loc[_t(8), "hours_to_turn"] == pytest.approx(4)
    assert df.loc[_t(8), "tide_range_ft"] == pytest.approx(2.6)
    assert df.loc[_t(6), "tide_stage"] == "high slack"
    # rate of change is positive on the flood and negative on the ebb
    assert df.loc[_t(3), "tide_rate_ft_h"] > 0 > df.loc[_t(9), "tide_rate_ft_h"]


def test_outside_event_range_is_missing_not_invented():
    df = tide_features(EVENTS, pd.date_range(_t(20), periods=3, freq="1h", tz="UTC"))
    assert df["tide_ft"].isna().all()


def test_current_interpolation_uses_official_max_and_slack():
    ev = [CurrentEvent(time=_t(0), velocity_kt=0, type="slack"), CurrentEvent(time=_t(3), velocity_kt=2.5, type="flood"),
          CurrentEvent(time=_t(6), velocity_kt=0, type="slack"), CurrentEvent(time=_t(9), velocity_kt=-2.0, type="ebb"),
          CurrentEvent(time=_t(12), velocity_kt=0, type="slack")]
    df = current_features(ev, idx(13))
    assert df.loc[_t(3), "current_kt"] == pytest.approx(2.5)
    assert df.loc[_t(6), "current_kt"] == pytest.approx(0)
    assert df.loc[_t(9), "current_phase"] == "ebb"
    assert df.loc[_t(3), "current_norm"] == pytest.approx(1.0)
    assert 0 < df.loc[_t(1), "current_norm"] < 1


def test_current_peak_not_assumed_at_mid_tide():
    # Inlet-style lag: max flood 2 h after mid-tide. Interpolation must follow the events.
    ev = [CurrentEvent(time=_t(2), velocity_kt=0, type="slack"), CurrentEvent(time=_t(5), velocity_kt=3.0, type="flood"),
          CurrentEvent(time=_t(8), velocity_kt=0, type="slack")]
    df = current_features(ev, pd.date_range(_t(2), periods=7, freq="1h", tz="UTC"))
    assert df["current_kt"].idxmax() == pd.Timestamp(_t(5))


def test_microsecond_index_resolution():
    """pandas 3 can create 'us' resolution indexes — math must not depend on it."""
    ix = idx().as_unit("us")
    df = tide_features(EVENTS, ix)
    assert df.iloc[6]["tide_ft"] == pytest.approx(3.0)


def test_proxy_current_between_zero_and_one():
    df = tide_features(EVENTS, idx())
    proxy = current_proxy_from_tide(df)
    assert np.nanmax(proxy) <= 1 and np.nanmin(proxy) >= 0
