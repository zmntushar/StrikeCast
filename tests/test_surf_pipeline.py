"""Surf pipeline: exposure classification, geography, fallbacks, anchoring, validation flags."""
import numpy as np
import pandas as pd
import pytest

from app.core.config import exposure_config
from app.core.geo import bearing_deg, haversine_km
from app.features.surf import Geometry
from app.services import surf as svc
from app.services.forecast import assemble, get_spot, historical_frame, load_spots
from dashboard.components.ui import surf_headline

FAVS = ["robert_moses_democrat_point", "point_lookout_jetties", "jones_beach_field_2",
        "jones_beach_field_10", "sunken_meadow_field_3"]


def frame(n=24, start="2026-10-23 12:00", **cols):
    ix = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    base = {"wave_ft": 4.5, "swell_ft": 3.6, "swell_period_s": 7.0, "swell_dir": 170.0,
            "windwave_ft": 2.0, "windwave_period_s": 3.5, "windwave_dir": 200.0,
            "wind_mph": 12.0, "wind_dir": 20.0, "onshore_mph": -8.0, "tide_ft": 2.0, "current_kt": -1.5}
    base.update(cols)
    return pd.DataFrame({k: [v] * n for k, v in base.items()}, index=ix)


def ctx(spot_id, **kw):
    sp = get_spot(spot_id)
    ex = exposure_config()[spot_id]
    return svc.SurfContext(spot=sp, geometry=Geometry.from_dict(ex), boundary_depth_m=ex.get("boundary_depth_m"),
                           now=pd.Timestamp("2026-10-23 12:00", tz="UTC"), **kw)


# ------------------------------------------------------ classification / geo
def test_favorites_exposure_classification():
    cls = {s: get_spot(s).exposure_class for s in FAVS}
    assert cls["jones_beach_field_10"] == "sheltered_bay"
    assert cls["sunken_meadow_field_3"] == "sound"
    for s in ("robert_moses_democrat_point", "point_lookout_jetties", "jones_beach_field_2"):
        assert cls[s] == "open_ocean"


def test_all_spots_have_sane_coordinates_and_geometry():
    expo = exposure_config()
    for s in load_spots():
        assert 40.4 < s.lat < 41.3 and -74.1 < s.lon < -71.7, s.id      # Long Island / NY Bight
        assert s.id in expo, f"run scripts/compute_exposure.py for {s.id}"
        if s.is_open_ocean:
            assert (expo[s.id]["boundary_depth_m"] or 0) > 5, s.id       # offshore point is in water
            bl, bo = s.boundary_point()
            assert 5 < haversine_km(s.lat, s.lon, bl, bo) < 20
            if not s.wave_point:
                diff = abs((bearing_deg(s.lat, s.lon, bl, bo) - s.facing_deg + 180) % 360 - 180)
                assert diff < 2


def test_geometry_matches_known_shorelines():
    ex = exposure_config()
    f10 = Geometry.from_dict(ex["jones_beach_field_10"])
    assert f10.fetch_at(180) < 3 and not f10.admitted().any()            # barrier island blocks the ocean
    f2 = Geometry.from_dict(ex["jones_beach_field_2"])
    assert f2.fetch_at(165) >= 100 and f2.fetch_at(0) == 0               # open south, land north
    pl = Geometry.from_dict(ex["point_lookout_jetties"])
    assert pl.fetch_at(110) < 3 and pl.fetch_at(180) >= 100              # Jones Beach West End shadows E/ESE
    sm = Geometry.from_dict(ex["sunken_meadow_field_3"])
    assert 15 < sm.fetch_at(0) < 40 and sm.fetch_at(180) == 0            # across the Sound to Connecticut


# ---------------------------------------------------------------- compute
def test_ocean_surf_is_breaking_estimate_with_range():
    out, _ = svc.compute_surf(frame(), ctx("jones_beach_field_2"))
    r = out.iloc[0]
    assert r["surf_kind"] == "breaking" and r["surf_source"] == "model"
    assert r["surf_lo_ft"] <= r["surf_ft"] <= r["surf_hi_ft"]
    assert r["surf_ft"] >= 0.6 * r["offshore_hs_ft"]                   # not unrealistically low
    assert r["surf_confidence"] in ("high", "medium", "low")
    assert "implausibly" not in r["surf_flags"]


def test_sheltered_spots_ignore_ocean_swell():
    big_swell_no_wind = frame(wave_ft=12.0, swell_ft=11.0, wind_mph=0.0)
    for sid in ("jones_beach_field_10", "sunken_meadow_field_3"):
        out, _ = svc.compute_surf(big_swell_no_wind, ctx(sid))
        assert out["surf_kind"].iloc[0] in ("wind chop", None)
        assert (out["surf_ft"].fillna(0) < 0.05).all(), sid


def test_onshore_wind_builds_chop_in_the_sound_not_offshore_wind():
    on, _ = svc.compute_surf(frame(wind_mph=18.0, wind_dir=20.0), ctx("sunken_meadow_field_3"))
    off, _ = svc.compute_surf(frame(wind_mph=18.0, wind_dir=200.0), ctx("sunken_meadow_field_3"))
    assert on["surf_ft"].iloc[-1] > 1.0 and off["surf_ft"].iloc[-1] < 0.1


def test_spots_differ_for_the_same_offshore_sea():
    f = frame(swell_dir=100.0, windwave_dir=90.0)
    vals = {s: svc.compute_surf(f, ctx(s))[0]["surf_ft"].iloc[0] for s in FAVS}
    assert len({round(v, 1) for v in vals.values()}) >= 4                 # never identical everywhere
    assert vals["jones_beach_field_10"] < vals["jones_beach_field_2"]
    assert vals["point_lookout_jetties"] < vals["jones_beach_field_2"]     # shadowed from the east


def test_missing_everything_gives_unavailable_not_a_number():
    nan = frame(wave_ft=np.nan, swell_ft=np.nan, windwave_ft=np.nan, wind_mph=np.nan, wind_dir=np.nan)
    for sid in ("jones_beach_field_2", "jones_beach_field_10"):
        out, _ = svc.compute_surf(nan, ctx(sid))
        assert out["surf_ft"].isna().all() and out["surf_kind"].isna().all()
        assert surf_headline(out.iloc[0]) == "surf estimate unavailable"


def test_fallback_to_nws_then_buoy_without_projection():
    f = frame(wave_ft=np.nan, swell_ft=np.nan, windwave_ft=np.nan)
    f["nws_c1_ft"], f["nws_c1_period_s"], f["nws_c1_dir"] = 5.0, 9.0, 157.5
    out, _ = svc.compute_surf(f, ctx("jones_beach_field_2"))
    assert (out["surf_source"] == "nws").all() and out["surf_ft"].notna().all()
    g = frame(wave_ft=np.nan, swell_ft=np.nan, windwave_ft=np.nan)
    g["buoy_wave_ft"], g["buoy_period_s"], g["buoy_wave_dir"] = 5.0, 9.0, 160.0
    out, _ = svc.compute_surf(g, ctx("jones_beach_field_2"))
    past = out.index <= pd.Timestamp("2026-10-23 12:00", tz="UTC")
    assert (out.loc[past, "surf_source"] == "buoy").all()
    assert out.loc[~past, "surf_ft"].isna().all()                        # measurements are never projected


# --------------------------------------------------------------- anchoring
def _buoy_case(ratio, n=48):
    ix = pd.date_range("2026-10-21 12:00", periods=n + 24, freq="1h", tz="UTC")
    model = pd.Series(4.0, index=ix)
    buoy = pd.Series(np.where(np.arange(len(ix)) < n, 4.0 * ratio, np.nan), index=ix)
    return buoy, model, ix[n - 1]


def test_anchoring_scales_near_now_and_relaxes():
    buoy, model, last = _buoy_case(1.25)
    info = svc.anchor_to_buoy(buoy, model, last, "forecast")
    f = info["factor"]
    assert info["applied"] and f.loc[last] == pytest.approx(1.25, abs=0.02)
    assert f.iloc[-1] == pytest.approx(1.25, abs=0.05)                    # long-term bias is also 1.25 here


def test_inconsistent_providers_are_flagged_not_forced():
    buoy, model, last = _buoy_case(3.0)
    info = svc.anchor_to_buoy(buoy, model, last, "forecast")
    assert not info["applied"] and info.get("inconsistent") and info["factor"] is None
    c = ctx("jones_beach_field_2", anchor=info)
    out, _ = svc.compute_surf(frame(), c)
    assert "inconsistent" in out["surf_flags"].iloc[0] or "outside" in out["surf_flags"].iloc[0]


def test_direction_offset_from_buoy():
    ix = pd.date_range("2026-10-21", periods=48, freq="1h", tz="UTC")
    info = svc.anchor_to_buoy(pd.Series(5.0, index=ix), pd.Series(5.0, index=ix), ix[-1], "forecast",
                              buoy_dir=pd.Series(110.0, index=ix), model_dir=pd.Series(90.0, index=ix))
    assert info["dir_offset_deg"] == pytest.approx(20.0, abs=0.5)
    wrap = svc.anchor_to_buoy(pd.Series(5.0, index=ix), pd.Series(5.0, index=ix), ix[-1], "forecast",
                              buoy_dir=pd.Series(5.0, index=ix), model_dir=pd.Series(355.0, index=ix))
    assert wrap["dir_offset_deg"] == pytest.approx(10.0, abs=0.5)        # wraps through north


def test_nws_mismatch_lowers_confidence_without_changing_value():
    base, _ = svc.compute_surf(frame(), ctx("jones_beach_field_2"))
    f = frame()
    f["nws_seas_lo_ft"], f["nws_seas_hi_ft"] = 12.0, 15.0
    flagged, _ = svc.compute_surf(f, ctx("jones_beach_field_2"))
    assert flagged["surf_ft"].iloc[0] == pytest.approx(base["surf_ft"].iloc[0])
    assert "NWS" in flagged["surf_flags"].iloc[0]
    assert flagged["surf_uncertainty"].iloc[0] > base["surf_uncertainty"].iloc[0]


def test_no_spikes_or_discontinuities_in_a_smooth_sea():
    n = 48
    f = frame(n=n)
    f["wave_ft"] = np.linspace(3, 6, n)
    f["swell_ft"] = f["wave_ft"] * 0.8
    f["windwave_ft"] = f["wave_ft"] * 0.6
    f["windwave_period_s"] = np.linspace(1.5, 4.0, n)                    # includes very short chop
    f["windwave_dir"] = np.linspace(200, 260, n)
    s = svc.compute_surf(f, ctx("robert_moses_democrat_point"))[0]["surf_ft"]
    assert s.diff().abs().max() < 0.6                                     # ft per hour
    assert (s < 2.5 * f["wave_ft"]).all()


# ------------------------------------------------------------ historical
class _Raw(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def test_historical_day_never_borrows_other_days_buoy_data():
    spot = get_spot("jones_beach_field_2")
    fr = historical_frame(pd.Timestamp("2026-10-10").date(), spot.timezone)
    other = pd.date_range("2026-10-20", periods=24, freq="1h", tz="UTC")
    obs = pd.DataFrame({"wave_ft": 6.0, "wave_period_s": 9.0, "wave_dir": 150.0}, index=other)
    raw = _Raw({"ndbc:44065": (obs, "u"), "ndbc:44025": (obs, "u")})
    fc = assemble(spot, fr, raw, 0)
    assert "buoy_wave_ft" not in fc.display or fc.display["buoy_wave_ft"].isna().all()
    assert "measured waves" in fc.missing
