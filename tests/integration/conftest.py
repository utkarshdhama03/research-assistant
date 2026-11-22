"""Integration fixtures: real Postgres test database, migrated once per session.

Each test runs inside an outer transaction that is rolled back, so tests never see each other's rows.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, delete, make_url, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from research_assistant.config import get_settings
from research_assistant.db.models import ResearchRun

ROOT = Path(__file__).resolve().parents[2]


def _test_db_url() -> str:
    s = get_settings()
    test_url, main_url = make_url(s.test_database_url), make_url(s.database_url)
    # These tests run `alembic downgrade base`; never point them at a real database.
    if not (test_url.database or "").endswith("_test"):
        pytest.exit(f"TEST_DATABASE_URL database must end in '_test', got {test_url.database!r}", 2)
    if (test_url.host, test_url.port, test_url.database) == (main_url.host, main_url.port, main_url.database):
        pytest.exit("TEST_DATABASE_URL must differ from DATABASE_URL", 2)
    return s.test_database_url


def alembic_config(db_url: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["db_url"] = db_url
    cfg.attributes["configure_logger"] = False
    return cfg


@pytest.fixture(scope="session")
def db_url() -> str:
    return _test_db_url()


# Held for the whole pytest session. Two sessions on one test database break each other: teardown deletes
# every run created since the test started (the other session's too) and the engine fixture rebuilds the
# schema. Seen 2026-11-22: two concurrent runs of test_memory.py failed with "run ... not found".
_SESSION_LOCK_KEY = 0x52415F54455354  # "RA_TEST"


@pytest.fixture(scope="session")
def engine(db_url: str) -> Iterator[Engine]:
    eng = create_engine(db_url)
    try:
        guard = eng.connect()
    except OperationalError as e:
        pytest.fail(f"test database unreachable ({e.orig}); run `docker compose up -d`", pytrace=False)
    locked = guard.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": _SESSION_LOCK_KEY}).scalar()
    guard.commit()  # a session-level advisory lock outlives the transaction
    if not locked:
        guard.close()
        pytest.exit("another pytest session is using the test database; run one at a time", 2)
    try:
        cfg = alembic_config(db_url)
        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")
        yield eng
    finally:
        guard.close()  # releases the lock
        eng.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    conn = engine.connect()
    outer = conn.begin()
    sess = Session(bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False)
    try:
        yield sess
    finally:
        sess.close()
        outer.rollback()
        conn.close()


class TrackingSessionFactory:
    """Session factory that counts open sessions, so fakes can assert none is open during I/O."""

    def __init__(self, engine: Engine) -> None:
        self._maker = sessionmaker(bind=engine, expire_on_commit=False)
        self.open = 0
        self.checks = 0

    def __call__(self) -> Session:
        session = self._maker()
        self.open += 1
        original_close = session.close
        closed = False

        def close() -> None:
            nonlocal closed
            if not closed:
                closed = True
                self.open -= 1
            original_close()

        session.close = close  # type: ignore[method-assign]
        return session

    def assert_idle(self, what: str) -> None:
        self.checks += 1
        assert self.open == 0, f"{what} was called while {self.open} database session(s) were open"


@pytest.fixture
def committing_factory(engine: Engine) -> Iterator[TrackingSessionFactory]:
    """Sessions that really commit. Teardown deletes every run created during the test (the cascade
    removes its rows), and runs even when the test fails."""
    database = engine.url.database or ""
    if not database.endswith("_test"):  # same guard as db_url(): never commit into a real database
        pytest.exit(f"committing tests need a *_test database, got {database!r}", 2)
    with engine.connect() as conn:
        before = set(conn.scalars(select(ResearchRun.id)))
    factory = TrackingSessionFactory(engine)
    try:
        yield factory
    finally:
        with engine.begin() as conn:
            stmt = delete(ResearchRun)
            if before:
                stmt = stmt.where(ResearchRun.id.not_in(before))
            conn.execute(stmt)


@pytest.fixture(scope="session")
def checkpoint_tables(engine: Engine, db_url: str) -> None:
    """The LangGraph checkpointer's tables in the test database (created by its own setup())."""
    from research_assistant.graph.checkpoint import postgres_checkpointer

    with postgres_checkpointer(db_url) as saver:
        saver.setup()


@pytest.fixture
def checkpointer(checkpoint_tables: None, engine: Engine, db_url: str) -> Iterator[Any]:
    """A strict PostgresSaver on the test database. Teardown removes every checkpoint and heartbeat."""
    from research_assistant.graph.checkpoint import postgres_checkpointer

    try:
        with postgres_checkpointer(db_url) as saver:
            yield saver
    finally:
        with engine.begin() as conn:
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints", "worker_heartbeats"):
                conn.execute(text(f"DELETE FROM {table}"))
