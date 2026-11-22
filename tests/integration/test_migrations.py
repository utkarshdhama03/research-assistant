from alembic import command
from sqlalchemy import Engine, inspect, text

from tests.integration.conftest import alembic_config

TABLES = {
    "research_runs",
    "sub_questions",
    "sources",
    "chunks",
    "claims",
    "claim_evidence",
    "reports",
    "run_events",
    "worker_heartbeats",
}


def test_downgrade_base_then_upgrade_head(engine: Engine, db_url: str) -> None:
    cfg = alembic_config(db_url)

    command.downgrade(cfg, "base")
    assert TABLES.isdisjoint(inspect(engine).get_table_names())

    command.upgrade(cfg, "head")
    assert TABLES <= set(inspect(engine).get_table_names())
    with engine.connect() as conn:
        assert conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar()
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "c7f3a9e1b2d6"


def test_downgrade_maps_new_statuses_to_failed(engine: Engine, db_url: str) -> None:
    cfg = alembic_config(db_url)
    command.upgrade(cfg, "head")
    with engine.begin() as conn:
        for status in ("queued", "awaiting_approval", "cancelled", "completed"):
            conn.execute(
                text("INSERT INTO research_runs (question, status) VALUES (:q, :s)"),
                {"q": status, "s": status},
            )
    try:
        command.downgrade(cfg, "e6b36bf2589f")
        with engine.connect() as conn:
            rows = dict(
                conn.execute(
                    text("SELECT question, status || ':' || coalesce(error, '') FROM research_runs")
                ).all()
            )
        assert rows == {
            "queued": "failed:downgraded from queued",
            "awaiting_approval": "failed:downgraded from awaiting_approval",
            "cancelled": "failed:downgraded from cancelled",
            "completed": "completed:",
        }
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM research_runs"))
        command.upgrade(cfg, "head")
