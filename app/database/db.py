"""SQLite (SQLAlchemy) storage: HTTP response cache + fishing observations.

The schema uses only portable column types so the same models work on PostgreSQL
by changing STRIKECAST_DATABASE_URL.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class HttpCache(Base):
    __tablename__ = "http_cache"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    url: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    body: Mapped[str] = mapped_column(Text)


class Observation(Base):
    """A fishing trip / sighting record (Phase 3)."""

    __tablename__ = "observations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    spot_id: Mapped[str] = mapped_column(String(64), index=True)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    start_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    duration_h: Mapped[float] = mapped_column(Float, default=2.0)
    species: Mapped[str] = mapped_column(String(32), index=True)
    fish_caught: Mapped[int] = mapped_column(Integer, default=0)
    largest_in: Mapped[float | None] = mapped_column(Float, nullable=True)
    lure: Mapped[str | None] = mapped_column(String(128), nullable=True)
    bait_seen: Mapped[str | None] = mapped_column(String(128), nullable=True)
    birds_diving: Mapped[bool] = mapped_column(Boolean, default=False)
    blitz: Mapped[bool] = mapped_column(Boolean, default=False)
    anglers: Mapped[int] = mapped_column(Integer, default=1)
    water_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # verified | user_reported | inferred
    confidence: Mapped[str] = mapped_column(String(16), default="user_reported")
    source: Mapped[str] = mapped_column(String(64), default="strikecast_log")
    # JSON snapshot of environmental features at the time of the trip
    features_json: Mapped[str | None] = mapped_column(Text, nullable=True)


_engine: Engine | None = None
_Session: sessionmaker | None = None
_lock = Lock()


def get_engine() -> Engine:
    global _engine, _Session
    with _lock:
        if _engine is None:
            url = get_settings().database_url
            if url.startswith("sqlite:///"):
                Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(url, future=True, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
            if url.startswith("sqlite"):
                @event.listens_for(_engine, "connect")
                def _pragmas(dbapi_conn, _):  # noqa: ANN001
                    cur = dbapi_conn.cursor()
                    cur.execute("PRAGMA journal_mode=WAL")
                    cur.execute("PRAGMA synchronous=NORMAL")
                    cur.close()
            Base.metadata.create_all(_engine)
            _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def session():
    get_engine()
    assert _Session is not None
    return _Session()


def reset_engine_for_tests(url: str) -> None:
    """Point the app at a different database (used by tests)."""
    global _engine, _Session
    with _lock:
        _engine = None
        _Session = None
    get_settings().database_url = url  # type: ignore[misc]
    get_engine()
