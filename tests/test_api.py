from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_and_static_endpoints():
    assert client.get("/health").json() == {"status": "ok"}
    spots = client.get("/spots").json()
    ids = {s["id"] for s in spots}
    for fav in ("jones_beach_field_10", "sunken_meadow_field_3", "point_lookout_jetties",
                "jones_beach_west_end_2", "robert_moses_democrat_point"):
        assert fav in ids
    assert all(s["favorite"] for s in spots if s["id"] in {"jones_beach_field_10", "jones_beach_west_end_2",
                                                          "robert_moses_democrat_point"})
    assert not next(s for s in spots if s["id"] == "jones_beach_field_2")["favorite"]
    assert set(client.get("/species").json()) == {"striped_bass", "bluefish", "fluke", "blackfish", "porgy"}
    assert "regulations" in client.get("/regulations").json()


def test_unknown_spot_404():
    assert client.get("/forecast/nowhere").status_code == 404
