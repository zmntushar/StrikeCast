"""90-minute fishing-window optimizer: the validation checklist from the feature brief.

All inputs are synthetic (no network): semidiurnal tide and current predictions
are generated in the same form NOAA CO-OPS returns them.
"""
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.features import current_model as cm
from app.features.solar import day_length_hours, solar_events
from app.models.feeding_profiles import _blend_curves, load_profile, solar_suitability
from app.models.window_optimizer import WINDOW, WindowPick, rank_windows
from app.schemas.env import CurrentEvent, TideEvent
from app.services.fishing_window import (WindowInputs, compare_window, compute_window, day_grid, inputs_from_forecast,
                                         window_changed)
from dashboard.components.window_chart import time_label, window_figure

TZ = "America/New_York"
M2 = 12.42  # hours, principal lunar semidiurnal period
PAST = datetime(2026, 1, 1, tzinfo=timezone.utc)  # "now" well before the test dates -> whole day eligible


def tides(phase_h: float = 0.0, start: str = "2026-11-02", days: int = 8, lo: float = 0.3, hi: float = 4.4):
    t0 = pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=phase_h)
    n = int(days * 24 / (M2 / 2)) + 2
    return [TideEvent(time=(t0 + pd.Timedelta(hours=M2 / 2 * i)).to_pydatetime(),
                      height_ft=lo if i % 2 == 0 else hi, type="L" if i % 2 == 0 else "H") for i in range(n)]


def currents(tide_events, slack_lag_h: float = 1.5, peak_kt: float = 2.2):
    """Slack lags each tide turn (typical inlet behavior); max flood/ebb halfway between slacks."""
    out = []
    for i, e in enumerate(tide_events[:-1]):
        t = pd.Timestamp(e.time) + pd.Timedelta(hours=slack_lag_h)
        out.append(CurrentEvent(time=t.to_pydatetime(), velocity_kt=0.0, type="slack"))
        flood = e.type == "L"   # after low water the flood runs
        out.append(CurrentEvent(time=(t + pd.Timedelta(hours=M2 / 4)).to_pydatetime(),
                                velocity_kt=peak_kt if flood else -peak_kt, type="flood" if flood else "ebb",
                                direction_deg=20.0 if flood else 200.0))
    return out


def inputs(phase_h=0.0, lat=40.59, lon=-73.57, env="jetty", with_current=False, **kw) -> WindowInputs:
    te = tides(phase_h)
    return WindowInputs(spot_id="test", spot_name="Test jetty", lat=lat, lon=lon, tz=TZ, environment=env,
                        structure=["jetty", "rocks"], tide_events=te, current_station="ACT0000" if with_current else None,
                        current_events=currents(te) if with_current else [], current_is_proxy=False, **kw)


DAY = date(2026, 11, 5)


# 1 --------------------------------------------------------------------------
@pytest.mark.parametrize("species", ["striped_bass", "bluefish", "fluke", "blackfish", "porgy"])
def test_every_recommended_interval_is_exactly_90_minutes(species):
    for d in (DAY, date(2026, 11, 6)):
        r = compute_window(inputs(with_current=True), species, d, now=PAST)
        for p in (r.best, r.second):
            assert p is not None
            assert p.end - p.start == pd.Timedelta(minutes=90)
            assert p.minutes == 90
        assert r.best.start.minute % 5 == 0  # 5-minute increments
        assert abs(r.best.start - r.second.start) >= WINDOW  # runner-up does not overlap
    c = compare_window(r, pd.Timestamp("2026-11-06 13:07", tz=TZ))["pick"]
    assert c.end - c.start == pd.Timedelta(minutes=90)


# 2 --------------------------------------------------------------------------
def test_algorithm_responds_to_species():
    inp = inputs(with_current=True)
    bass = compute_window(inp, "striped_bass", DAY, now=PAST)
    tog = compute_window(inp, "blackfish", DAY, now=PAST)
    assert bass.best.start != tog.best.start
    seg = tog.frame.loc[tog.best.start:tog.best.end]
    assert (seg["sun_alt"] > 0).all()                 # tautog: daylight feeder
    night = tog.frame[tog.frame["sun_alt"] < -18]
    assert night["total"].max() < 25                  # night suitability generally low
    bass_night = bass.frame[bass.frame["sun_alt"] < -18]
    assert bass_night["solar"].min() > 0.7            # bass can feed in darkness


def test_species_profiles_have_distinct_behavior_and_weights():
    profiles = {s: load_profile(s) for s in ["striped_bass", "bluefish", "fluke", "blackfish", "porgy"]}
    assert "crepuscular" in profiles["striped_bass"].categories and "nocturnal" in profiles["striped_bass"].categories
    assert profiles["blackfish"].categories == ("diurnal",)
    assert len({tuple(sorted(p.weights.items())) for p in profiles.values()}) == len(profiles)
    for p in profiles.values():
        assert sum(p.weights.values()) == pytest.approx(1.0)
    # Solar response: not every species peaks at twilight
    e = np.array([-30.0, -4.0, 40.0])
    bass, tog = (solar_suitability(profiles[s], e) for s in ("striped_bass", "blackfish"))
    assert bass[1] > bass[2] and tog[2] > tog[1] > tog[0]


def test_new_species_falls_back_to_category_templates():
    curve = _blend_curves(("diurnal", "nocturnal"))
    xs, ys = zip(*curve)
    assert min(xs) == -90 and max(xs) == 90 and all(0 <= y <= 1 for y in ys)


# 3 --------------------------------------------------------------------------
def test_algorithm_responds_to_location():
    a = compute_window(inputs(phase_h=0.0), "fluke", DAY, now=PAST)
    b = compute_window(inputs(phase_h=3.0, lat=41.07, lon=-71.86, env="beach"), "fluke", DAY, now=PAST)
    assert a.best.start != b.best.start
    assert not np.allclose(a.frame["sun_alt"], b.frame["sun_alt"])   # different coordinates


# 4 --------------------------------------------------------------------------
def test_algorithm_responds_to_date():
    inp = inputs(with_current=True)
    starts = [compute_window(inp, "fluke", date(2026, 11, d), now=PAST).best.start for d in (4, 5, 6)]
    local_times = {s.tz_convert(TZ).strftime("%H:%M") for s in starts}
    assert len(local_times) > 1                         # tides shift ~50 min/day, so does the window
    assert len({s.tz_convert(TZ).date() for s in starts}) == 3


# 5 --------------------------------------------------------------------------
def test_solar_cycle_accounts_for_seasons():
    june = solar_events(40.6, -73.5, date(2026, 6, 21), TZ)
    dec = solar_events(40.6, -73.5, date(2026, 12, 21), TZ)
    for ev in (june, dec):
        order = ["astronomical_dawn", "nautical_dawn", "civil_dawn", "sunrise", "solar_noon", "sunset",
                 "civil_dusk", "nautical_dusk", "astronomical_dusk"]
        times = [ev[k] for k in order]
        assert all(t is not None for t in times) and times == sorted(times)
    assert day_length_hours(june) > 15 > 9.5 > day_length_hours(dec)
    assert pd.Timestamp(june["sunrise"]).tz_convert(TZ).hour == 5
    assert pd.Timestamp(dec["sunrise"]).tz_convert(TZ).hour == 7
    # Same clock time, different light: 6:00 AM is daylight in June, darkness in December.
    rj = compute_window(inputs(), "fluke", date(2026, 6, 21), now=PAST)
    rd = compute_window(inputs(), "fluke", date(2026, 12, 21), now=PAST)
    tj = pd.Timestamp("2026-06-21 06:00", tz=TZ).tz_convert("UTC")
    td = pd.Timestamp("2026-12-21 06:00", tz=TZ).tz_convert("UTC")
    assert rj.frame.loc[tj, "sun_alt"] > 0 > rd.frame.loc[td, "sun_alt"]
    assert rj.frame.loc[tj, "solar"] > rd.frame.loc[td, "solar"]


# 6 --------------------------------------------------------------------------
def test_tide_and_current_are_separate_variables():
    with_cur = compute_window(inputs(with_current=True), "blackfish", DAY, now=PAST)
    assert with_cur.current_source == cm.STATION
    f = with_cur.frame
    day = f.iloc[:-18]
    # Slack current is NOT at high/low water: it follows the station's own predictions.
    high = [pd.Timestamp(e.time) for e in with_cur.inputs.tide_events if e.type == "H"
            and day.index[0] <= pd.Timestamp(e.time) <= day.index[-1]]
    for h in high:
        assert abs(f.loc[h.floor("5min"), "current_kt"]) > 0.5
    assert len(with_cur.slack) >= 3
    # Changing only the current predictions changes the current component but not the tide component.
    other = replace(with_cur.inputs, current_events=currents(with_cur.inputs.tide_events, slack_lag_h=0.2, peak_kt=0.9))
    r2 = compute_window(other, "blackfish", DAY, now=PAST)
    pd.testing.assert_series_equal(f["tide"], r2.frame["tide"])
    pd.testing.assert_series_equal(f["tide_ft"], r2.frame["tide_ft"])
    assert not np.allclose(f["current"], r2.frame["current"])
    # Without a current station, no velocity is invented from tide height.
    no_cur = compute_window(inputs(), "blackfish", DAY, now=PAST)
    assert no_cur.current_source == cm.TIDE_PHASE
    assert no_cur.frame["current_kt"].isna().all()
    assert no_cur.frame["current_norm"].between(0, 1).all()


# 7 --------------------------------------------------------------------------
def test_missing_current_data_reduces_confidence():
    a = compute_window(inputs(with_current=True), "striped_bass", DAY, now=PAST)
    b = compute_window(inputs(), "striped_bass", DAY, now=PAST)
    proxy = compute_window(replace(inputs(with_current=True), current_is_proxy=True), "striped_bass", DAY, now=PAST)
    assert a.confidence["score"] > proxy.confidence["score"] > b.confidence["score"]
    assert any("approximate" in t for t in b.confidence["estimated"])
    none = compute_window(replace(inputs(), tide_events=[]), "striped_bass", DAY, now=PAST)
    assert none.current_source == cm.MISSING
    assert "tidal current" in none.confidence["missing"] and "tide predictions" in none.confidence["missing"]
    assert none.confidence["score"] < b.confidence["score"]


# 8 --------------------------------------------------------------------------
def test_recommended_window_can_cross_midnight():
    # Late evening "today": only windows starting 11 PM or later remain, and they run past midnight.
    late = pd.Timestamp("2026-11-05 23:00", tz=TZ).tz_convert("UTC").to_pydatetime()
    r = compute_window(inputs(), "striped_bass", DAY, now=late)
    assert not r.elapsed
    s, e = r.best.start.tz_convert(TZ), r.best.end.tz_convert(TZ)
    assert s.date() == DAY and e.date() == date(2026, 11, 6)
    assert np.isfinite(r.best.score)
    # Pure optimizer: a score peak straddling midnight is found exactly.
    grid, starts = day_grid(DAY, TZ)
    peak = pd.Timestamp("2026-11-05 23:40", tz=TZ).tz_convert("UTC")
    score = pd.Series(np.where((grid >= peak) & (grid <= peak + WINDOW), 90.0, 20.0), index=grid)
    best = rank_windows(score, starts)[0]
    assert best.start == peak and best.end.tz_convert(TZ).date() == date(2026, 11, 6)


def test_daylight_saving_day_has_25_hours_of_starts():
    grid, starts = day_grid(date(2026, 11, 1), TZ)   # DST ends: 25-hour local day
    assert len(starts) == 25 * 12
    assert grid[-1] - grid[0] == pd.Timedelta(hours=25) + WINDOW


def test_today_prefers_upcoming_windows():
    noon = pd.Timestamp("2026-11-05 12:00", tz=TZ).tz_convert("UTC").to_pydatetime()
    r = compute_window(inputs(), "blackfish", DAY, now=noon)
    assert r.best.start >= pd.Timestamp(noon) and r.second.start >= pd.Timestamp(noon)
    done = compute_window(inputs(), "blackfish", DAY, now=pd.Timestamp("2026-11-06 06:00", tz=TZ).to_pydatetime())
    assert done.elapsed and done.best is not None   # whole day passed: best-of-day shown for reference


# 9 --------------------------------------------------------------------------
def test_charts_are_synchronized_and_rectangle_spans_both():
    r = compute_window(inputs(with_current=True), "striped_bass", DAY, now=PAST)
    cmpw = compare_window(r, pd.Timestamp("2026-11-05 12:00", tz=TZ))["pick"]
    fig = window_figure(r, compare=cmpw)
    lay = fig.layout
    assert lay.xaxis.matches == "x2" or lay.xaxis2.matches == "x"     # one shared horizontal axis
    sun_trace = next(t for t in fig.data if t.name == "Sun elevation")
    tide_trace = next(t for t in fig.data if t.name == "Tide height (predicted)")
    assert sun_trace.xaxis == "x" and tide_trace.xaxis == "x2"
    assert list(sun_trace.x) == list(tide_trace.x)                     # identical time samples
    assert any(t.name.startswith("Predicted current") and t.yaxis == "y3" for t in fig.data)  # own current axis
    rect = next(s for s in lay.shapes if s.name == "best_window")
    assert rect.yref == "paper" and rect.y0 == 0 and rect.y1 == 1      # spans both panels
    assert pd.Timestamp(rect.x1) - pd.Timestamp(rect.x0) == pd.Timedelta(minutes=90)
    assert any(s.name == "compare_window" for s in lay.shapes)
    assert any(s.name == "second_window" for s in lay.shapes)


# 10 -------------------------------------------------------------------------
def test_window_boundaries_display_local_times():
    assert time_label(pd.Timestamp("2026-07-10 10:15", tz="UTC"), TZ) == "6:15 AM"    # EDT (UTC-4)
    assert time_label(pd.Timestamp("2026-12-10 11:15", tz="UTC"), TZ) == "6:15 AM"    # EST (UTC-5)
    assert time_label(pd.Timestamp("2026-12-10 23:45", tz="UTC"), TZ) == "6:45 PM"
    r = compute_window(inputs(), "fluke", DAY, now=PAST)
    fig = window_figure(r)
    ann = {a.name: a for a in fig.layout.annotations if a.name}
    assert time_label(r.best.start, TZ) in ann["best_window_start"].text
    assert time_label(r.best.end, TZ) in ann["best_window_end"].text
    x0 = pd.Timestamp(next(s for s in fig.layout.shapes if s.name == "best_window").x0)
    assert x0 == r.best.start.tz_convert(TZ).tz_localize(None)       # drawn at local wall-clock time


# 11 -------------------------------------------------------------------------
def test_environmental_updates_trigger_recalculation():
    r1 = compute_window(inputs(with_current=True), "fluke", DAY, now=PAST)
    # A revised forecast: weather turns rough during the recommended window.
    grid = pd.date_range("2026-11-01", "2026-11-10", freq="1h", tz="UTC")
    bad = (grid >= r1.best.start - pd.Timedelta(hours=2)) & (grid <= r1.best.end + pd.Timedelta(hours=2))
    hourly = pd.DataFrame({"ff_fluke_wind": np.where(bad, 0.0, 0.9), "ff_fluke_waves": np.where(bad, 0.0, 0.9),
                           "fishability_fluke": np.where(bad, 0.3, 1.0)}, index=grid)
    r2 = compute_window(replace(r1.inputs, hourly=hourly), "fluke", DAY, now=PAST)
    assert window_changed(r1.best, r2.best)
    assert not (r2.best.start < r1.best.end and r1.best.start < r2.best.end)  # moved off the rough period
    assert not window_changed(r1.best, r1.best)
    assert window_changed(None, r1.best)


# 12 -------------------------------------------------------------------------
def test_comparison_window_does_not_overwrite_recommendation():
    r = compute_window(inputs(with_current=True), "striped_bass", DAY, now=PAST)
    best, second, total = r.best, r.second, r.frame["total"].copy()
    for t in ("2026-11-05 03:00", "2026-11-05 12:30", "2026-11-05 23:55"):
        c = compare_window(r, pd.Timestamp(t, tz=TZ))
        assert c["pick"].rank == 0 and c["pick"].score <= best.score + 1e-9
        assert c["factors"]
    assert r.best == best and r.second == second
    pd.testing.assert_series_equal(r.frame["total"], total)
    with pytest.raises(FrozenInstanceError):
        r.best = WindowPick(start=best.start, end=best.end, score=0)  # type: ignore[misc]


# -------------------------------------------------------------- explanation
def test_explanation_names_the_conditions():
    r = compute_window(inputs(with_current=True), "striped_bass", DAY, now=PAST)
    ex = r.explanation
    assert "tide" in ex["summary"] and "current" in ex["summary"]
    assert {f["component"] for f in ex["factors"]} >= {"solar", "current", "tide"}
    assert all(f["kind"] for f in ex["factors"])
    assert 0 <= r.confidence["score"] <= 100 and r.confidence["label"] in ("High", "Medium", "Low")


def test_works_from_a_forecast_with_no_data():
    """Integration: the real forecast object, every provider offline."""
    from app.core.http import FetchError
    from app.services.forecast import assemble, forecast_frame, get_spot

    class _AllFail(dict):
        def get(self, key, default=None):
            return FetchError("offline")

    fc = assemble(get_spot("point_lookout_jetties"), forecast_frame(3), _AllFail(), 0)
    assert fc.current_events == []
    inp = inputs_from_forecast(fc)
    day = pd.Timestamp.now(tz=TZ).date()
    r = compute_window(inp, "blackfish", day)
    assert r.current_source == cm.MISSING and r.confidence["label"] == "Low"
    assert "tide predictions" in r.confidence["missing"]


# ------------------------------------------------------------ consistency
def test_time_of_day_limit_restricts_window_starts():
    r = compute_window(inputs(), "fluke", DAY, now=PAST, tod=(5, 9))
    for p in (r.best, r.second):
        assert 5 <= p.start.tz_convert(TZ).hour < 9
    none = compute_window(inputs(), "fluke", DAY, now=PAST, tod=(3, 3))
    assert none.best is None and none.second is None


def test_vectorized_window_scores_match_interval_score():
    from app.models.window_optimizer import candidate_scores, interval_score
    r = compute_window(inputs(with_current=True), "striped_bass", DAY, now=PAST)
    fast = candidate_scores(r.frame["total"], r.starts).to_numpy()
    slow = np.array([interval_score(r.frame["total"], s) for s in r.starts])
    assert np.allclose(fast, slow, equal_nan=True)


def test_main_page_popover_shows_the_optimizer_windows():
    """The hover card must list exactly the 90-minute windows the Best 90 min tab recommends."""
    from types import SimpleNamespace
    from dashboard.components.ui import span_text, species_popover
    r = compute_window(inputs(with_current=True), "striped_bass", DAY, now=PAST)
    fc = SimpleNamespace(spot=SimpleNamespace(timezone=TZ, short="Test"), hourly=pd.DataFrame())
    html = species_popover(fc, "striped_bass", pd.DataFrame(), [(r, r.best)], {}, {}, "")
    assert span_text(r.best.start, r.best.end, TZ) in html
    assert f"{r.best.score:.0f}/100" in html
    assert "Best 90-minute windows" in html
