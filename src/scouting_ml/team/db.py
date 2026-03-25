"""Database helpers for ScoutML Team Edition."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from scouting_ml.core.runtime_config import load_api_runtime_config
from scouting_ml.team.models import Base


_LOCK = threading.Lock()
_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None
_ENGINE_URL: str | None = None


def team_mode_enabled() -> bool:
    """Return whether team mode is enabled by runtime configuration."""
    config = load_api_runtime_config()
    return bool(config.team_mode and config.database_url.strip())


def get_database_url() -> str:
    """Return the configured SQLAlchemy database URL for team mode."""
    config = load_api_runtime_config()
    return config.database_url.strip()


def _engine_kwargs(url: str) -> dict:
    kwargs: dict = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return kwargs


def get_engine() -> Engine:
    """Return a process-wide SQLAlchemy engine for the configured database."""
    global _ENGINE, _SESSION_FACTORY, _ENGINE_URL
    url = get_database_url()
    if not url:
        raise RuntimeError("Team mode database URL is not configured.")
    if _ENGINE is not None and _ENGINE_URL == url:
        return _ENGINE
    with _LOCK:
        if _ENGINE is not None and _ENGINE_URL == url:
            return _ENGINE
        _ENGINE = create_engine(url, **_engine_kwargs(url))
        _SESSION_FACTORY = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False, expire_on_commit=False)
        _ENGINE_URL = url
        return _ENGINE


def get_session_factory() -> sessionmaker[Session]:
    """Return a cached sessionmaker bound to the team engine."""
    global _SESSION_FACTORY
    if _SESSION_FACTORY is not None:
        return _SESSION_FACTORY
    get_engine()
    assert _SESSION_FACTORY is not None
    return _SESSION_FACTORY


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a SQLAlchemy session and commit or roll back automatically."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all_tables() -> None:
    """Create the Team Edition schema if it is missing."""
    if not team_mode_enabled():
        return
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def reset_team_db_caches() -> None:
    """Clear cached engine/session state. Useful for tests."""
    global _ENGINE, _SESSION_FACTORY, _ENGINE_URL
    with _LOCK:
        if _ENGINE is not None:
            _ENGINE.dispose()
        _ENGINE = None
        _SESSION_FACTORY = None
        _ENGINE_URL = None


__all__ = [
    "create_all_tables",
    "get_database_url",
    "get_engine",
    "get_session_factory",
    "reset_team_db_caches",
    "session_scope",
    "team_mode_enabled",
]
