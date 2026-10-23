"""Marine data ingestion: units, parsing, QC, staleness, timezones, provider fallback."""
import asyncio
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from app.core.http import HttpClient
from app.providers import marine, nws_marine

FIX = Path(__file__).parent / "fixtures"


def run(c):
    return asyncio.run(c)


# ------------------------------------------------------------- units / parse
def test_ndbc_spec_parse_units_and_compass_dirs():
    df = marine.parse_ndbc_spec((FIX / "ndbc_44065_20261023.spec").read_text())
    r = df.loc[pd.Timestamp("2026-10-23 16:10", tz="UTC")]
    assert r["wave_ft"] == pytest.approx(1.5 * 3.28084)            # metres -> feet, exactly once
    assert r["swell_ft"] == pytest.approx(0.9 * 3.28084)
    assert r["swell_period_s"] == 7.1 and r["swell_dir"] == 112.5    # "ESE"
    assert r["windwave_dir"] == 90.0 and r["wave_dir"] == 100
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    assert df.loc[pd.Timestamp("2026-10-23 14:10", tz="UTC"), "wave_ft"] != df.loc[pd.Timestamp("2026-10-23 14:10", tz="UTC"), "wave_ft"]


def test_openmeteo_marine_converts_metres_once_and_keeps_partitions():
    payload = {"latitude": 40.5, "longitude": -73.5, "hourly": {
        "time": ["2026-10-23T12:00", "2026-10-23T13:00"],
        "wave_height": [1.0, 1.2], "wave_period": [6.0, 6.5], "wave_direction": [100, 105],
        "swell_wave_height": [0.8, 0.9], "swell_wave_period": [7.0, 7.2], "swell_wave_direction": [110, 112],
        "secondary_swell_wave_height": [0.3, 0.3], "secondary_swell_wave_period": [11, 11],
        "secondary_swell_wave_direction": [150, 150],
        "wind_wave_height": [0.5, 0.6], "wind_wave_period": [3.0, 3.1], "wind_wave_direction": [40, 45],
        "sea_surface_temperature": [17.0, 17.0]}}

    async def go():
        async with HttpClient(use_cache=False, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))) as h:
            return await marine.openmeteo_marine(h, [(40.5, -73.5)], past_days=1, forecast_days=1)

    df, src = run(go())[0]
    assert df["wave_ft"].iloc[0] == pytest.approx(3.28084)
    assert df["swell2_ft"].iloc[0] == pytest.approx(0.3 * 3.28084)
    assert df["windwave_dir"].iloc[1] == 45
    assert df.index[0] == pd.Timestamp("2026-10-23 12:00", tz="UTC")   # GMT request -> UTC index
    assert src.kind.value == "model"


# ------------------------------------------------------------------- QC
def test_qc_removes_spikes_invalid_periods_and_directions():
    ix = pd.date_range("2026-10-23", periods=8, freq="1h", tz="UTC")
    df = pd.DataFrame({"wave_ft": [3.0, 3.1, 3.2, 19.0, 3.3, 3.2, 3.1, -1.0],
                       "wave_period_s": [8, 8, 8, 8, 0.2, 8, 45, 8],
                       "wave_dir": [100, 100, 400, 100, 100, -5, 100, 100]}, index=ix)
    out, flags = marine.qc_waves(df)
    assert np.isnan(out["wave_ft"].iloc[3]) and np.isnan(out["wave_ft"].iloc[7])   # spike, negative
    assert out["wave_ft"].iloc[2] == 3.2                                            # neighbours kept
    assert np.isnan(out["wave_period_s"].iloc[4]) and np.isnan(out["wave_period_s"].iloc[6])
    assert np.isnan(out["wave_dir"].iloc[2]) and np.isnan(out["wave_dir"].iloc[5])
    assert flags


def test_qc_keeps_a_genuine_storm_ramp():
    ix = pd.date_range("2026-10-22", periods=8, freq="1h", tz="UTC")
    df = pd.DataFrame({"wave_ft": [4, 6, 8, 10, 12, 13, 12, 11.0]}, index=ix)
    out, _ = marine.qc_waves(df)
    assert out["wave_ft"].notna().all()


def test_qc_flags_stuck_sensor():
    ix = pd.date_range("2026-10-20", periods=40, freq="1h", tz="UTC")
    out, flags = marine.qc_waves(pd.DataFrame({"wave_ft": [2.3] * 40}, index=ix))
    assert out["wave_ft"].isna().all() and flags


# --------------------------------------------------------------- NWS CWF
def test_cwf_parse_seas_wave_detail_headlines():
    z = nws_marine.parse_cwf((FIX / "cwf_okx_20261023.txt").read_text())
    assert {"ANZ353", "ANZ335", "ANZ345"} <= set(z)
    f = z["ANZ353"]
    assert f.issued == pd.Timestamp("2026-10-23 13:26", tz="UTC")       # 926 AM EDT
    assert f.expires == pd.Timestamp("2026-10-24 02:15", tz="UTC")
    assert any("SMALL CRAFT ADVISORY" in h for h in f.headlines)
    p0 = f.periods[0]
    assert (p0.seas_lo_ft, p0.seas_hi_ft, p0.seas_occ_ft) == (4.0, 6.0, 8.0)
    assert [(c.dir_deg, c.height_ft, c.period_s) for c in p0.components] == [(90.0, 6.0, 9.0)]
    assert z["ANZ345"].periods[0].kind == "waves"


def test_cwf_periods_map_to_local_6am_6pm():
    f = nws_marine.parse_cwf((FIX / "cwf_okx_20261023.txt").read_text())["ANZ353"]
    today, tonight = f.periods[0], f.periods[1]
    assert today.end == pd.Timestamp("2026-10-23 22:00", tz="UTC")     # 6 PM EDT
    assert tonight.end == pd.Timestamp("2026-10-24 10:00", tz="UTC")   # 6 AM EDT


def test_cwf_period_bounds_across_dst_end():
    issued = pd.Timestamp("2026-10-31 20:00", tz="UTC")                 # 4 PM EDT Sat
    b = nws_marine.period_bounds(["TONIGHT", "SUN", "SUN NIGHT"], issued)
    assert b[0][1] == pd.Timestamp("2026-11-01 11:00", tz="UTC")        # 6 AM EST (fall back)
    assert b[1][1] == pd.Timestamp("2026-11-01 23:00", tz="UTC")        # 6 PM EST
    assert b[2][0] == b[1][1]


def test_cwf_stale_detection():
    f = nws_marine.parse_cwf((FIX / "cwf_okx_20261023.txt").read_text())["ANZ353"]
    assert not f.stale(pd.Timestamp("2026-10-23 20:00", tz="UTC"))
    assert f.stale(pd.Timestamp("2026-10-24 06:00", tz="UTC"))


def test_ugc_ranges_expand():
    assert nws_marine.expand_ugc("ANZ330-335>338-340-240215-") == ["ANZ330", "ANZ335", "ANZ336", "ANZ337", "ANZ338", "ANZ340"]


def test_zone_hourly_steps():
    f = nws_marine.parse_cwf((FIX / "cwf_okx_20261023.txt").read_text())["ANZ353"]
    ix = pd.date_range("2026-10-23 14:00", periods=12, freq="1h", tz="UTC")
    h = nws_marine.zone_hourly(f, ix)
    ts = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    assert h.loc[ts("2026-10-23 21:00"), "nws_seas_hi_ft"] == 6.0
    assert h.loc[ts("2026-10-24 01:00"), "nws_seas_hi_ft"] == 4.0
    assert h.loc[ts("2026-10-23 14:00"), "nws_c1_period_s"] == 9.0


def test_srf_surf_height_parse():
    days = nws_marine.parse_srf((FIX / "srf_okx_20260810.txt").read_text())
    nyz179 = [d for d in days if d.zone == "NYZ179"]
    assert nyz179 and nyz179[0].day == pd.Timestamp("2026-08-11").date()   # issued Mon evening -> TUESDAY
    assert (nyz179[0].surf_lo_ft, nyz179[0].surf_hi_ft) == (1.0, 2.0)
    assert nyz179[0].rip_risk == "Low"


def test_iem_product_split():
    blob = "\x01\n861 \nFZUS51 KOKX 220938\nCWFOKX\nA\n\x03\x01\n786 \nFZUS51 KOKX 220641\nCWFOKX\nB\n\x03"
    assert len(nws_marine.split_products(blob)) == 2


# ---------------------------------------------------------- outage fallback
def test_cwf_falls_back_to_tgftp_when_api_down():
    text = (FIX / "cwf_okx_20261023.txt").read_text()

    def handler(req: httpx.Request):
        if "api.weather.gov" in str(req.url):
            return httpx.Response(503)
        return httpx.Response(200, text=text)

    async def go():
        async with HttpClient(use_cache=False, transport=httpx.MockTransport(handler)) as h:
            return await nws_marine.latest_cwf(h, ["ANZ353"])

    zones, url = run(go())
    assert "ANZ353" in zones and "tgftp" in url
