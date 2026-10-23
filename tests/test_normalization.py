import numpy as np
import pandas as pd
import pytest

from app.core.geo import angle_diff, compass, compass_to_deg, haversine_km, wind_components
from app.features.weather_features import nearshore_waves, pressure_tendency
from app.providers.marine import parse_ndbc_stdmet
from app.providers.weather import _parse_nws_speed

NDBC_SAMPLE = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft
2026 10 23 14 50  40  9.0 10.0   1.5     6   4.9  93 1028.2  12.0  16.8   3.8   MM   MM    MM
2026 10 23 14 40  MM   MM   MM   1.8     9   5.2 105     MM    MM  16.3    MM   MM   MM    MM
"""


def test_ndbc_parse_units_and_missing():
    df = parse_ndbc_stdmet(NDBC_SAMPLE)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    last = df.iloc[-1]
    assert last["wave_ft"] == pytest.approx(1.5 * 3.28084)
    assert last["wind_mph"] == pytest.approx(9.0 * 2.236936)
    assert last["water_temp_f"] == pytest.approx(16.8 * 9 / 5 + 32)
    first = df.iloc[0]
    assert np.isnan(first["wind_mph"]) and np.isnan(first["pressure_hpa"])  # "MM" -> explicit NaN


def test_ndbc_parse_garbage_returns_empty():
    assert parse_ndbc_stdmet("<html>404</html>").empty


def test_wind_components_onshore_offshore():
    # South-facing beach (facing 180): south wind is onshore, north wind offshore.
    on, along = wind_components(10, 180, 180)
    assert on == pytest.approx(10) and along == pytest.approx(0, abs=1e-9)
    on, _ = wind_components(10, 0, 180)
    assert on == pytest.approx(-10)
    on, along = wind_components(10, 90, 180)
    assert on == pytest.approx(0, abs=1e-9) and abs(along) == pytest.approx(10)


def test_geo_helpers():
    assert haversine_km(40.62, -73.30, 40.62, -73.30) == 0
    assert 30 < haversine_km(40.6, -73.6, 40.6, -73.2) < 35
    assert compass(0) == "N" and compass(225) == "SW" and compass(None) == "—"
    assert compass_to_deg("SSW") == 202.5
    assert angle_diff(10, 350) == 20 and angle_diff(350, 10) == -20
    assert _parse_nws_speed("5 to 10 mph") == 10 and _parse_nws_speed(None) is None


def test_pressure_tendency():
    p = pd.Series(np.arange(30, dtype=float), index=pd.date_range("2026-10-23", periods=30, freq="1h", tz="UTC"))
    t = pressure_tendency(p)
    assert t["pressure_tend_3h"].iloc[-1] == 3 and t["pressure_tend_24h"].iloc[-1] == 24
    assert np.isnan(t["pressure_tend_24h"].iloc[0])


def test_nearshore_waves_direction_and_exposure():
    ix = pd.date_range("2026-10-23", periods=3, freq="1h", tz="UTC")
    off = pd.Series([4.0, 4.0, 4.0], index=ix)
    head_on = nearshore_waves(off, pd.Series([180.0] * 3, index=ix), pd.Series([0.0] * 3, index=ix), 180, 1.0, "ocean")
    away = nearshore_waves(off, pd.Series([0.0] * 3, index=ix), pd.Series([0.0] * 3, index=ix), 180, 1.0, "ocean")
    sheltered = nearshore_waves(off, pd.Series([180.0] * 3, index=ix), pd.Series([0.0] * 3, index=ix), 180, 0.1, "bay")
    assert head_on.iloc[0] == pytest.approx(4.0)
    assert away.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert sheltered.iloc[0] == pytest.approx(0.4)
