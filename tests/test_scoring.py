import numpy as np
import pandas as pd
import pytest

from app.features.seasonal import curve_value, seasonal_factor, trapezoid
from app.models import bite
from app.models.baitfish import bait_scores
from app.models.common import species_list
from app.models.feeding import current_factor, feeding_scores
from app.models.presence import presence_scores, structure_score
from app.models.storms import detect_storms, storm_multiplier
from app.services.forecast import get_spot


def frame(n=24, **cols):
    ix = pd.date_range("2026-10-23 10:00", periods=n, freq="1h", tz="UTC")
    base = {"water_temp_f": 60.0, "wind_mph": 10.0, "wind_dir": 180.0, "gust_mph": 14.0, "onshore_mph": 8.0,
            "onshore_hours_12h": 6.0, "offshore_hours_12h": 0.0, "nearshore_wave_ft": 2.0, "current_norm": 0.6,
            "tide_dir": "outgoing", "tide_stage_tag": "mid_outgoing", "hours_since_turn": 3.0, "hours_to_turn": 3.0,
            "light_phase": "dusk", "cloud_pct": 50.0, "pressure_tend_3h": -1.0, "water_temp_trend_3d": -1.5,
            "storm_phase": "none", "hours_since_storm": np.nan, "precip_in": 0.0, "visibility_mi": 10.0,
            "thunder": False}
    base.update(cols)
    return pd.DataFrame({k: [v] * n for k, v in base.items()}, index=ix)


def score(spot_id, sp, df):
    spot = get_spot(spot_id)
    b = bait_scores(df, spot)
    df = df.join(b).join(presence_scores(df, spot, sp, b))
    from app.models.presence import bait_availability
    from app.models.common import species_meta
    df = df.join(feeding_scores(df, spot, sp, bait_availability(species_meta(sp), b)))
    df = df.join(bite.fishability(df, spot, sp))
    df[f"bite_{sp}"] = bite.combine(df, sp)
    return df


def test_trapezoid_and_nan():
    v = trapezoid([40, 50, 60, 70, np.nan], 45, 55, 65, 75)
    assert v[0] == 0 and v[1] == pytest.approx(0.5) and v[2] == 1 and v[3] == pytest.approx(0.5)
    assert np.isnan(v[4])


def test_season_curve_wraps_year():
    pts = [[1, 0.2], [180, 1.0], [365, 0.2]]
    assert curve_value(pts, 365.5) == pytest.approx(0.2, abs=0.01)
    assert 0 <= seasonal_factor(pts, np.array([1, 200, 364]), 41.1, "sound").min()


def test_scores_bounded_for_all_species():
    for sp in species_list():
        df = score("robert_moses_democrat_point", sp, frame())
        for c in (f"presence_{sp}", f"feeding_{sp}", f"bite_{sp}"):
            assert df[c].between(0, 100).all(), c


def test_blackfish_needs_rocks():
    beach = score("jones_beach_field_2", "blackfish", frame())["presence_blackfish"].mean()
    jetty = score("point_lookout_jetties", "blackfish", frame())["presence_blackfish"].mean()
    assert jetty > beach * 1.5
    s, tag = structure_score({"structure": {"jetty": 1.0, "beach": 0.1}, "structure_required": True},
                             get_spot("jones_beach_field_2"))
    assert s < 0.2


def test_cold_water_kills_bluefish():
    warm = score("robert_moses_democrat_point", "bluefish", frame(water_temp_f=68.0))["presence_bluefish"].mean()
    cold = score("robert_moses_democrat_point", "bluefish", frame(water_temp_f=48.0))["presence_bluefish"].mean()
    assert cold < 10 < warm


def test_fishability_penalizes_gale_and_thunder():
    calm = score("point_lookout_jetties", "striped_bass", frame())["fishability_striped_bass"].mean()
    gale = score("point_lookout_jetties", "striped_bass", frame(wind_mph=38.0, gust_mph=50.0, nearshore_wave_ft=10.0))
    storm = score("point_lookout_jetties", "striped_bass", frame(thunder=True))
    assert calm == 1.0
    assert gale["fishability_striped_bass"].mean() < 0.1
    assert storm["fishability_striped_bass"].mean() == pytest.approx(0.25)


def test_bass_prefers_low_light_over_midday():
    dusk = score("robert_moses_democrat_point", "striped_bass", frame(light_phase="dusk", cloud_pct=0.0))
    noon = score("robert_moses_democrat_point", "striped_bass", frame(light_phase="day", cloud_pct=0.0))
    assert dusk["feeding_striped_bass"].mean() > noon["feeding_striped_bass"].mean()


def test_current_factor_shapes():
    bass = current_factor(np.array([0.0, 0.5, 1.0]), 0.35, 0.95, 0.3)
    tog = current_factor(np.array([0.0, 0.3, 1.0]), 0.05, 0.5, 0.75)
    assert bass[0] == pytest.approx(0.3) and bass[1] == 1
    assert tog[2] < tog[1]  # blackfish dislike a ripping tide


def test_storm_detection_and_effects():
    df = frame(48, wind_mph=10.0, wave_ft=3.0, pressure_tend_12h=0.0)
    df.iloc[20:26, df.columns.get_loc("wind_mph")] = 32.0
    st = detect_storms(df)
    assert st["storm_active"].iloc[22]
    assert st["storm_phase"].iloc[2] == "none"      # 18 h before onset
    assert st["storm_phase"].iloc[12] == "pre"      # within the 12 h pre-storm window
    assert st["storm_phase"].iloc[30] == "post"
    assert st["hours_since_storm"].iloc[30] == 5
    mult = storm_multiplier(np.array(["during", "post", "none"]), np.array([np.nan, 30, np.nan]),
                            {"during": 0.5, "post": 1.2})
    assert mult[0] == 0.5 and mult[1] > 1.0 and mult[2] == 1.0


def test_best_windows_are_contiguous_and_ranked():
    ix = pd.date_range("2026-10-23", periods=12, freq="1h", tz="UTC")
    s = pd.Series([20, 30, 70, 80, 75, 30, 20, 60, 65, 20, 10, 10], index=ix, dtype=float)
    w = bite.best_windows(s, "striped_bass", "America/New_York")
    assert w[0].peak_score == 80 and w[0].start == ix[2] and w[0].end == ix[5]
    assert all(a.peak_score >= b.peak_score for a, b in zip(w, w[1:]))


def test_explanations_name_factors():
    df = score("robert_moses_democrat_point", "striped_bass", frame())
    df["water_temp_kind"] = "measured"
    txt = bite.narrative(df.iloc[0], "striped_bass", get_spot("robert_moses_democrat_point"))
    assert "striped bass" in txt.lower()
    assert bite.score_label(85)[0] == "Hot" and bite.score_label(10)[0] == "Slow"


def test_regulation_status():
    from datetime import date
    assert bite.season_status("blackfish", date(2026, 10, 23))["open"] is True
    assert bite.season_status("blackfish", date(2026, 7, 1))["open"] is False
