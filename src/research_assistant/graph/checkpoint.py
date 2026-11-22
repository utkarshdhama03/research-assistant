"""LangGraph Postgres checkpointer for plan-approval pauses.

Its tables (checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations) live in the main
database and are created by PostgresSaver.setup() at worker start, not by Alembic (see the decision
log). Deserialization is strict: only LangGraph's built-in safe types are allowed, which covers our
graph state (UUIDs, lists and dicts of them, ints).
"""

from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import Connection
from psycopg.rows import DictRow, dict_row

from research_assistant.db.session import libpq_url


def strict_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=None)


@contextmanager
def postgres_checkpointer(database_url: str) -> Iterator[PostgresSaver]:
    """A checkpointer on its own autocommit connection (what PostgresSaver requires)."""
    conn: Connection[DictRow]
    with Connection.connect(
        libpq_url(database_url), autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        yield PostgresSaver(conn, serde=strict_serializer())
