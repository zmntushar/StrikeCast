"""Fishing-session feedback: storage of predictions, evaluation, and schema migration."""
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from app.database.db import _migrate
from app.services.fishing_window import compute_window
from app.services.observations import add_observation, delete_observation, list_observations, sessions_with_predictions
from app.services.session_eval import component_signals, evaluate, session_prediction
from tests.test_fishing_window import DAY, PAST, TZ, inputs


def test_session_saves_prediction_and_outcomes():
    r = compute_window(inputs(with_current=True), "porgy", DAY, now=PAST)
    start = r.best.start
    end = r.best.end + pd.Timedelta(minutes=30)
    pred = session_prediction(r, start, end)
    assert pred["overlap_best_min"] == 90
    assert pred["best"]["score"] == r.best.score and pred["fished_score"] is not None
    oid = add_observation({"spot_id": "test", "start_utc": start, "end_utc": end, "species": "porgy", "bites": 6,
                           "fish_caught": 2, "lure": "clam strips", "water_notes": "clear, light chop"}, None, pred)
    try:
        df = list_observations()
        row = df[df["id"] == oid].iloc[0]
        assert row["bites"] == 6 and row["has_prediction"]
        assert row["duration_h"] == 2.0                      # derived from start/end
        assert row["water_notes"] == "clear, light chop"
        s = [x for x in sessions_with_predictions("porgy") if x["id"] == oid][0]
        assert s["prediction"]["model_version"] == "window-v1"
    finally:
        delete_observation(oid)


def test_skunks_are_not_treated_as_model_failures():
    r = compute_window(inputs(with_current=True), "porgy", DAY, now=PAST)
    sess = [{"id": 1, "start_utc": r.best.start, "duration_h": 2, "bites": 0, "fish_caught": 0,
             "prediction": session_prediction(r, r.best.start, r.best.end)}]
    ev = evaluate(sess)
    assert ev["n"] == 1 and not ev["ready"]
    assert any("not treated as proof" in m for m in ev["messages"])
    assert "spearman" not in ev
    assert component_signals(sess) is None                # too few sessions: no calibration advice


def test_evaluation_with_enough_sessions():
    r = compute_window(inputs(with_current=True), "fluke", DAY, now=PAST)
    sess = []
    for i, h in enumerate(range(0, 24, 2)):
        s = pd.Timestamp(f"2026-11-05 {h:02d}:00", tz=TZ).tz_convert("UTC")
        p = session_prediction(r, s, s + pd.Timedelta(minutes=90))
        sess.append({"id": i, "start_utc": s, "duration_h": 1.5, "fish_caught": 0,
                     "bites": int(round((p["fished_score"] or 0) / 10)), "prediction": p})
    ev = evaluate(sess)
    assert ev["ready"] and ev["spearman"]["rho"] > 0.9
    assert sum(b["sessions"] for b in ev["bands"]) == len(sess)


def test_migration_adds_new_columns_to_old_database(tmp_path):
    eng = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE observations (id INTEGER PRIMARY KEY, spot_id VARCHAR(64), species VARCHAR(32))"))
        c.execute(text("INSERT INTO observations (spot_id, species) VALUES ('x', 'porgy')"))
    _migrate(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("observations")}
    assert {"end_utc", "bites", "prediction_json"} <= cols
    with eng.begin() as c:
        assert c.execute(text("SELECT count(*) FROM observations")).scalar() == 1   # data kept
    _migrate(eng)   # idempotent
