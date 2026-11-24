"""Blitz potential model and the Bite Index fixes that came with it. All inputs are synthetic (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.solar import solar_elevation
from app.models import blitz
from app.models.feeding import pressure_factor, wind_factor
from app.services.forecast import get_spot

TZ = "America/New_York"
SPOT = get_spot("jones_beach_west_end_2")        # ocean jetty at Jones Inlet


def frame(day: str = "2026-11-20", **over) -> pd.DataFrame:
    """A textbook fall blitz setup at a south-shore inlet jetty; ``over`` replaces any column."""
    idx = pd.date_range(pd.Timestamp(day, tz=TZ).tz_convert("UTC"), periods=24, freq="1h")
    n = len(idx)
    hours = np.arange(n)
    df = pd.DataFrame(index=idx)
    df["sun_alt"] = solar_elevation(SPOT.lat, SPOT.lon, idx)
    df["pf_striped_bass_season"] = 0.95
    df["pf_bluefish_season"] = 0.2
    df["water_temp_f"] = 57.5
    df["water_temp_trend_3d"] = -0.8
    df["water_temp_kind"] = "measured"
    for g in ("adult_bunker", "peanut_bunker", "sand_eel", "silverside", "bay_anchovy"):
        df[f"bait_{g}"] = 75.0
    df["bait_flush"] = 0.95
    df["bait_flush_measured"] = True
    df["bay_ref_temp_f"], df["ocean_ref_temp_f"] = 49.0, 58.0
    df["wind_mph"], df["gust_mph"], df["wind_dir"] = 5.0, 8.0, 265.0
    df["onshore_mph"], df["offshore_hours_12h"] = 1.0, 0.0
    df["nearshore_wave_ft"], df["clarity_index"] = 1.5, 0.1
    df["storm_phase"], df["hours_since_storm"] = "none", np.nan
    df["thunder"], df["precip_in"], df["visibility_mi"] = False, 0.0, 10.0
    df["pressure_tend_24h"], df["cloud_pct"] = -3.0, 10.0
    # strong ebb all day (worst case for the tests: only light and weather can stop an alert)
    df["current_norm"], df["current_phase"], df["tide_dir"] = 0.8, "ebb", "outgoing"
    df["tide_ft"], df["tide_stage"] = 2.0 + np.sin(hours / 2), "mid outgoing"
    for k, v in over.items():
        df[k] = v
    return df


def levels(df: pd.DataFrame, **kw) -> pd.Series:
    out = blitz.blitz_scores(df, SPOT, mode=kw.pop("mode", "historical"), **kw)
    out.index = out.index.tz_convert(TZ)
    return out["blitz_level"]


def local_hours(lv: pd.Series, level: str) -> list[int]:
    return list(lv.index[lv == level].hour)


# ------------------------------------------------------------------ the setup itself
def test_textbook_setup_alerts_at_first_and_last_light_only():
    lv = levels(frame())
    alert = local_hours(lv, "alert")
    assert alert, "a perfect fall setup must produce an alert"
    assert all(h in (6, 7, 8, 15, 16, 17) for h in alert), alert     # dawn / dusk only
    assert not ((lv != "") & (lv.index.hour.isin([0, 1, 2, 3, 12, 22]))).any()   # never night or high noon


@pytest.mark.parametrize("what, over", [
    ("strong offshore wind pushes bait out of range", {"wind_mph": 18.0, "gust_mph": 26.0, "onshore_mph": -16.0,
                                                       "offshore_hours_12h": 12.0}),
    ("big surf", {"nearshore_wave_ft": 6.5}),
    ("storm in progress", {"storm_phase": "during"}),
    ("bluebird day after a cold front", {"pressure_tend_24h": 9.0, "cloud_pct": 0.0}),
    ("bait has not left the bays", {"bait_flush": 0.15}),
    ("water too warm for the run", {"water_temp_f": 67.0}),
    ("out of season", {"pf_striped_bass_season": 0.3}),
    ("slack water all day", {"current_norm": 0.0}),
])
def test_one_missing_ingredient_kills_the_alert(what, over):
    lv = levels(frame(**over))
    assert not (lv == "alert").any(), what


def test_flood_tide_at_an_inlet_is_at_most_a_watch():
    lv = levels(frame(current_phase="flood", tide_dir="incoming"))
    assert not (lv == "alert").any()
    assert (lv == "watch").any()


def test_alerts_need_verified_inputs():
    df = frame(water_temp_kind="model")
    out = blitz.blitz_scores(df, SPOT, mode="historical")
    assert not (out["blitz_level"] == "alert").any()
    assert (out["blitz_cap"].str.contains("water temperature")).any()
    df = frame(bait_flush_measured=False)
    assert not (blitz.blitz_scores(df, SPOT, mode="historical")["blitz_level"] == "alert").any()


def test_alerts_more_than_three_days_out_are_held_at_watch():
    df = frame()
    now = df.index[0] - pd.Timedelta(days=4)
    lv = levels(df, now=now, mode="forecast")
    assert not (lv == "alert").any() and (lv == "watch").any()
    lv_soon = levels(df, now=df.index[0] - pd.Timedelta(hours=2), mode="forecast")
    assert (lv_soon == "alert").any()


def test_bay_piers_rarely_flag():
    pier = get_spot("jones_beach_green_island")
    out = blitz.blitz_scores(frame(), pier, mode="historical")
    assert not (out["blitz_level"] == "alert").any()


# ------------------------------------------------------------------ components
def test_bait_flush_follows_the_bay_cooling_below_the_ocean():
    doy = np.array([320, 320, 320, 150])
    fl, measured = blitz.bait_flush(doy, bay_t=np.array([49, 61, 40, 49]), ocean_t=np.array([58, 61, 55, 58]),
                                    bay_trend=np.array([-1, 0, -2, -1]))
    assert fl[0] > 0.9                    # bay 9 F colder than the ocean in November: bait pouring out
    assert fl[1] < 0.5                    # bay as warm as the ocean: no push yet
    assert fl[2] < 0.1                    # bay 40 F: the bait already left
    assert np.isnan(fl[3])                # spring: not applicable
    assert measured[:3].all() and not measured[3]


def test_bait_flush_without_gauges_is_an_estimate():
    fl, measured = blitz.bait_flush(np.array([320]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]),
                                    own_t=np.array([55.0]), own_trend=np.array([-3.0]))
    assert 0 < fl[0] < 1 and not measured[0]


def test_light_peaks_at_first_light():
    idx = pd.date_range(pd.Timestamp("2026-11-20", tz=TZ).tz_convert("UTC"), periods=24, freq="1h")
    e = solar_elevation(SPOT.lat, SPOT.lon, idx)
    lt = pd.Series(blitz.light_component(e, np.full(24, 10.0)), index=idx.tz_convert(TZ))
    assert lt[lt.index.hour == 7].iloc[0] == pytest.approx(1.0)
    assert lt[lt.index.hour == 12].iloc[0] < 0.6
    assert lt[lt.index.hour == 1].iloc[0] < 0.3


def test_sightings_upgrade_watch_but_never_create_a_signal():
    df = frame(current_phase="flood", tide_dir="incoming")     # watch only
    bl = blitz.blitz_scores(df, SPOT, mode="historical")
    s = [{"time": df.index[0], "label": "birds diving logged 3 km away"}]
    up = blitz.apply_sightings(bl, s)
    assert (up["blitz_level"] == "alert").sum() == (bl["blitz_level"] == "watch").sum()
    quiet = blitz.blitz_scores(frame(storm_phase="during"), SPOT, mode="historical")
    assert (blitz.apply_sightings(quiet, s)["blitz_level"] == "").all()


def test_windows_group_contiguous_hours_and_explain_themselves():
    df = frame()
    df = df.join(blitz.blitz_scores(df, SPOT, mode="historical"))
    ws = blitz.blitz_windows(df, SPOT)
    assert ws and ws[0].level == "alert"
    w = ws[0]
    assert (w.end - w.start) >= pd.Timedelta(hours=1)
    assert any("bays" in r for r in w.reasons)                   # names the bait push
    assert all(LEVEL in ("alert", "watch") for LEVEL in (x.level for x in ws))


# ------------------------------------------------------------------ Bite Index fixes
def test_post_front_pressure_rise_is_penalized():
    flat = pressure_factor(np.array([0.5]), 0.25, 0.15, np.array([0.0]))[0]
    post_front = pressure_factor(np.array([0.5]), 0.25, 0.15, np.array([11.0]))[0]
    slow_fall = pressure_factor(np.array([-0.3]), 0.25, 0.15, np.array([-5.0]))[0]
    assert post_front < flat - 0.2                # the 3-hour change alone missed this
    assert slow_fall > flat


def test_strong_offshore_wind_hurts_more():
    on = np.array([-20.0, -6.0, 10.0])
    w = wind_factor(on, [6, 20], "ocean")
    assert w[0] < 0.4 < w[1] < w[2]


def test_feeding_limiting_factor_pulls_score_down():
    from app.models.feeding import feeding_scores
    df = frame().copy()
    df["light_phase"], df["tide_stage_tag"] = "dawn", "high_outgoing"
    df["hours_since_turn"], df["hours_to_turn"] = 2.0, 4.0
    df["pressure_tend_3h"] = 0.0
    bait_av = np.full(len(df), 0.8)
    good = feeding_scores(df, SPOT, "striped_bass", bait_av)["feeding_striped_bass"].mean()
    df["pressure_tend_24h"] = 11.0                       # only the barometer changes
    bad = feeding_scores(df, SPOT, "striped_bass", bait_av)["feeding_striped_bass"].mean()
    assert bad < good * 0.8


def test_out_of_season_species_cannot_score_hot_windows():
    from app.models.feeding_profiles import load_profile
    from app.models.window_scoring import total_score

    prof = load_profile("porgy")
    assert prof.limiting.get("seasonal") and prof.limiting.get("temperature")
    good = pd.DataFrame({c: [0.95] for c in ("solar", "current", "tide", "habitat", "conditions")})
    good["fishability"] = 1.0
    in_season = total_score(good.assign(seasonal=1.0, temperature=1.0), prof)[0]
    gone = total_score(good.assign(seasonal=0.1, temperature=0.5), prof)[0]   # late November porgy
    assert in_season > 80 and gone < 35


def test_bay_gauge_never_stands_in_for_a_sound_spot():
    from app.services.forecast import assemble, forecast_frame

    sound = get_spot("sunken_meadow_field_3")            # temp_sources: Kings Point (Sound), then 44069 (bay)
    frame_ = forecast_frame(1)
    t = pd.date_range(frame_.start, pd.Timestamp.now(tz="UTC"), freq="1h")
    bay = (pd.DataFrame({"water_temp_f": np.full(len(t), 48.0)}, index=t), "url")

    class Raw(dict):
        def get(self, key, default=None):
            return bay if key == "ndbc:44069" else super().get(key, default)

    fc = assemble(sound, frame_, Raw(), 0)
    assert not any(s.feature == "water temperature" and s.station_id == "44069" for s in fc.sources)
    assert not (fc.display["water_temp_f"] == 48.0).any()      # missing (or model SST), never the bay buoy
    assert "measured water temperature" in fc.missing or "water temperature" in fc.missing
