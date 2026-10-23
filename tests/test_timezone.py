from datetime import date, datetime, timezone

import pandas as pd

from app.providers import lunar
from app.services.forecast import forecast_frame, historical_frame


def test_forecast_frame_is_utc_and_hourly():
    f = forecast_frame(3, now=datetime(2026, 10, 23, 15, 37, tzinfo=timezone.utc))
    assert str(f.display_start.tz) == "UTC"
    assert f.display_start == pd.Timestamp("2026-10-23 15:00", tz="UTC")
    assert f.display_end - f.display_start == pd.Timedelta(days=3)
    assert f.index.freq == pd.Timedelta(hours=1)


def test_historical_frame_starts_at_local_midnight_edt():
    f = historical_frame(date(2026, 10, 10), "America/New_York")
    assert f.display_start == pd.Timestamp("2026-10-10 04:00", tz="UTC")  # EDT = UTC-4
    assert f.display_start.tz_convert("America/New_York").hour == 0


def test_historical_frame_across_dst_end():
    # 2026-11-01 is the US fall-back day: 25 local hours.
    f = historical_frame(date(2026, 11, 1), "America/New_York")
    assert f.display_start == pd.Timestamp("2026-11-01 04:00", tz="UTC")
    assert (f.display_end - f.display_start) == pd.Timedelta(hours=24)  # 25 h day -> 25 hourly stamps


def test_sunrise_is_morning_local_time():
    ev = lunar.daily_events(40.62, -73.30, date(2026, 10, 23), "America/New_York")
    rise = pd.Timestamp(ev["sunrise"]).tz_convert("America/New_York")
    sset = pd.Timestamp(ev["sunset"]).tz_convert("America/New_York")
    assert rise.hour == 7 and sset.hour == 18
    assert ev["civil_dawn"] < ev["sunrise"] < ev["sunset"] < ev["civil_dusk"]


def test_moon_phase_known_dates():
    # New moon 2026-10-10, full moon 2026-10-26 (both from published almanacs)
    assert lunar.moon_phase_info(datetime(2026, 10, 10, 16, tzinfo=timezone.utc))["phase_name"] == "New Moon"
    assert lunar.moon_phase_info(datetime(2026, 10, 26, 6, tzinfo=timezone.utc))["phase_name"] == "Full Moon"


def test_light_phase_labels():
    alt = [-20, -5, 5, 30, 5, -5, -20]
    prev = [-25, -20, -5, 5, 30, 5, -5]
    assert list(lunar.light_phase(alt, prev)) == ["night", "dawn", "dawn", "day", "dusk", "dusk", "night"]
