"""Engine and session factory. One session per unit of work; repositories never commit."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from research_assistant.config import get_settings


def libpq_url(sqlalchemy_url: str) -> str:
    """The libpq form of a SQLAlchemy psycopg URL (for psycopg.connect and the checkpointer)."""
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


def make_engine(url: str | None = None) -> Engine:
    return create_engine(url or get_settings().database_url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@lru_cache
def default_session_factory() -> sessionmaker[Session]:
    return make_session_factory(make_engine())


@contextmanager
def session_scope(factory: Callable[[], Session] | None = None) -> Iterator[Session]:
    """Yield a session; commit on success, roll back on any exception."""
    session = (factory or default_session_factory())()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
