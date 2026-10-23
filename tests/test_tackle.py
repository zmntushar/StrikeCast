from app.models.tackle import recommend, water_clarity


def row(**kw):
    base = {"current_kt": 1.0, "light_phase": "day", "moon_illum": 50, "nearshore_wave_ft": 1.0,
            "storm_phase": "none", "hours_since_storm": None, "precip_in": 0.0, "bait_top": "sand_eel"}
    base.update(kw)
    return base


def test_heavier_jig_in_deep_fast_water():
    slow = recommend("striped_bass", row(current_kt=0.2), "beach", depth_m=2)
    fast = recommend("striped_bass", row(current_kt=3.0), "inlet", depth_m=8)
    assert fast["weight_oz"] > slow["weight_oz"]
    assert 0.5 <= slow["weight_oz"] <= 3.0 and fast["weight_oz"] <= 3.0
    assert any("Strong current" in n for n in fast["notes"])


def test_color_follows_light_and_clarity():
    assert "black" in recommend("striped_bass", row(light_phase="night", moon_illum=5), "jetty")["color"]
    assert "white" in recommend("striped_bass", row(light_phase="night", moon_illum=95), "jetty")["color"]
    assert "chartreuse" in recommend("fluke", row(nearshore_wave_ft=5.0), "beach")["color"]
    assert water_clarity(row(storm_phase="post", hours_since_storm=6)) == "stained"
    assert water_clarity(row(nearshore_wave_ft=0.5)) == "clear"


def test_species_specific_advice():
    assert "crab" in recommend("blackfish", row(), "jetty")["primary"].lower()
    assert "hi-lo" in recommend("porgy", row(), "pier")["primary"].lower()
    assert "wire" in recommend("bluefish", row(), "inlet")["rig"].lower()
    assert "bucktail" in recommend("fluke", row(), "inlet")["jig"].lower()
    assert "sand-eel" in recommend("striped_bass", row(), "beach")["bait_match"]


def test_missing_values_do_not_crash():
    r = recommend("striped_bass", {"current_kt": float("nan")}, "pier", depth_m=None)
    assert r["primary"] and r["depth_ft"] > 0
