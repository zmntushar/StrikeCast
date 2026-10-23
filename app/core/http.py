"""Async HTTP client with retry/backoff, rate limiting and a SQLite response cache.

Every response records provider, URL and fetch time so the UI can show data age.
If the network fails and an expired cache entry exists, it is returned and
flagged as ``stale`` instead of failing (stale-if-error) — never silently.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from app.core.config import get_settings, user_agent
from app.core.logging import get_logger, log_kv
from app.database.db import HttpCache, session

log = get_logger("http")

# Minimum spacing between requests per host (seconds) — provider usage policies.
HOST_MIN_INTERVAL = {
    "nominatim.openstreetmap.org": 1.1,
    "overpass-api.de": 2.0,
}


class FetchError(RuntimeError):
    pass


@dataclass
class FetchResult:
    data: Any
    url: str
    provider: str
    fetched_at: datetime
    from_cache: bool = False
    stale: bool = False


def _ssl_context():
    """Prefer the operating-system trust store (works behind TLS-inspecting
    antivirus / corporate proxies); fall back to certifi's bundle."""
    try:
        import ssl

        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:  # noqa: BLE001
        return True


def _cache_key(method: str, url: str, params: dict | None, body: str | None) -> str:
    raw = f"{method}|{url}|{json.dumps(params or {}, sort_keys=True)}|{body or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()


class HttpClient:
    """Shared async client. Use ``async with HttpClient() as http:``."""

    def __init__(self, use_cache: bool | None = None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        s = get_settings()
        self._transport = transport  # injectable for tests
        self.use_cache = s.cache_enabled if use_cache is None else use_cache
        self._client: httpx.AsyncClient | None = None
        self._sem = asyncio.Semaphore(s.http_max_concurrency)
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._host_last: dict[str, float] = {}

    async def __aenter__(self) -> "HttpClient":
        s = get_settings()
        self._client = httpx.AsyncClient(
            transport=self._transport,
            verify=_ssl_context(),
            timeout=httpx.Timeout(s.http_timeout_s),
            headers={"User-Agent": user_agent(), "Accept": "application/json, text/plain, */*"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------ cache
    def _cache_get(self, key: str) -> HttpCache | None:
        if not self.use_cache:
            return None
        with session() as db:
            return db.get(HttpCache, key)

    def _cache_put(self, key: str, provider: str, url: str, body: str, ttl: int) -> datetime:
        now = datetime.now(timezone.utc)
        if self.use_cache and ttl > 0:
            with session() as db:
                db.merge(HttpCache(key=key, provider=provider, url=url, fetched_at=now,
                                   expires_at=now + timedelta(seconds=ttl), body=body))
                db.commit()
        return now

    # ---------------------------------------------------------------- request
    async def _throttle(self, host: str) -> None:
        gap = HOST_MIN_INTERVAL.get(host)
        if not gap:
            return
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            loop = asyncio.get_running_loop()
            wait = self._host_last.get(host, 0) + gap - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            self._host_last[host] = loop.time()

    async def request(
        self,
        url: str,
        *,
        provider: str,
        params: dict[str, Any] | None = None,
        ttl: int = 3600,
        method: str = "GET",
        data: dict[str, Any] | None = None,
        as_json: bool = True,
        headers: dict[str, str] | None = None,
    ) -> FetchResult:
        body_str = urlencode(data) if data else None
        key = _cache_key(method, url, params, body_str)
        full_url = url + ("?" + urlencode(params) if params else "")
        cached = self._cache_get(key)
        now = datetime.now(timezone.utc)
        if cached is not None:
            exp = cached.expires_at if cached.expires_at.tzinfo else cached.expires_at.replace(tzinfo=timezone.utc)
            if exp > now:
                fa = cached.fetched_at if cached.fetched_at.tzinfo else cached.fetched_at.replace(tzinfo=timezone.utc)
                payload = json.loads(cached.body) if as_json else cached.body
                return FetchResult(payload, full_url, provider, fa, from_cache=True)

        assert self._client is not None, "use 'async with HttpClient()'"
        s = get_settings()
        host = urlparse(url).netloc
        last_err: Exception | None = None
        for attempt in range(s.http_max_retries + 1):
            try:
                await self._throttle(host)
                async with self._sem:
                    resp = await self._client.request(method, url, params=params, data=data, headers=headers)
                if resp.status_code == 429 or resp.status_code >= 500:
                    retry_after = resp.headers.get("Retry-After")
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}" + (f" retry-after={retry_after}" if retry_after else ""),
                        request=resp.request, response=resp)
                resp.raise_for_status()
                text = resp.text
                payload = json.loads(text) if as_json else text
                fetched = self._cache_put(key, provider, full_url, text, ttl)
                log_kv(log, logging.DEBUG, "fetched", provider=provider, status=resp.status_code, attempt=attempt)
                return FetchResult(payload, full_url, provider, fetched)
            except (httpx.HTTPStatusError, httpx.TransportError, json.JSONDecodeError) as exc:
                last_err = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    break  # client error: retrying will not help
                if attempt < s.http_max_retries:
                    delay = min(8.0, 0.8 * 2 ** attempt) + random.uniform(0, 0.4)
                    ra = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After") if status == 429 else None
                    if ra and str(ra).isdigit():
                        delay = max(delay, min(30.0, float(ra)))
                    await asyncio.sleep(delay)
        log_kv(log, logging.WARNING, "fetch failed", provider=provider, url=full_url[:160], error=str(last_err)[:200])
        if cached is not None:
            fa = cached.fetched_at if cached.fetched_at.tzinfo else cached.fetched_at.replace(tzinfo=timezone.utc)
            payload = json.loads(cached.body) if as_json else cached.body
            return FetchResult(payload, full_url, provider, fa, from_cache=True, stale=True)
        raise FetchError(f"{provider}: {last_err}")
