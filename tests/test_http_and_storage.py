import asyncio

import httpx
import pytest

from app.core.http import FetchError, HttpClient
from app.database.db import HttpCache, session
from app.services.observations import add_observation, export_csv, import_csv, list_observations


def run(c):
    return asyncio.run(c)


def test_cache_hit_avoids_second_request():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(200, json={"ok": calls["n"]})

    async def go():
        async with HttpClient(transport=httpx.MockTransport(handler)) as http:
            a = await http.request("https://example.test/a", provider="T", ttl=600)
            b = await http.request("https://example.test/a", provider="T", ttl=600)
            return a, b

    a, b = run(go())
    assert calls["n"] == 1 and not a.from_cache and b.from_cache and b.data == {"ok": 1}


def test_stale_cache_used_when_provider_down():
    async def seed():
        async with HttpClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"v": 1}))) as http:
            await http.request("https://example.test/b", provider="T", ttl=1)

    run(seed())
    with session() as db:  # force expiry
        for row in db.query(HttpCache).filter(HttpCache.url.like("%example.test/b%")):
            row.expires_at = row.fetched_at
        db.commit()

    async def down():
        async with HttpClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as http:
            return await http.request("https://example.test/b", provider="T", ttl=1)

    res = run(down())
    assert res.stale and res.data == {"v": 1}


def test_client_error_raises_without_cache():
    async def go():
        async with HttpClient(transport=httpx.MockTransport(lambda r: httpx.Response(404))) as http:
            await http.request("https://example.test/nothing", provider="T", ttl=60)

    with pytest.raises(FetchError):
        run(go())


def test_observation_roundtrip_csv():
    oid = add_observation({"spot_id": "point_lookout_jetties", "start_utc": "2026-10-20T15:00:00Z", "species": "blackfish",
                           "fish_caught": 3, "largest_in": 17.5, "lure": "green crab", "birds_diving": False},
                          {"water_temp_f": 61.0, "bite_blackfish": 80.0})
    assert oid > 0
    df = list_observations()
    assert (df["species"] == "blackfish").any()
    csv = export_csv()
    n_before = len(df)
    assert import_csv(csv) == n_before
    assert len(list_observations()) == 2 * n_before
    assert "features_json" in csv.decode()


def test_import_csv_requires_columns():
    with pytest.raises(ValueError):
        import_csv(b"foo,bar\n1,2\n")
