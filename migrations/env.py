"""Alembic environment. The database URL never lives in alembic.ini.

URL precedence: Config.attributes["db_url"] (tests) > `alembic -x db_url=...` > DATABASE_URL from config.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from research_assistant.config import get_settings
from research_assistant.db.models import Base

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Created and migrated by langgraph-checkpoint-postgres (PostgresSaver.setup()), not by Alembic.
CHECKPOINTER_TABLES = frozenset(
    {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
)


def include_object(obj: object, name: str | None, type_: str, reflected: bool, compare_to: object) -> bool:
    """Keep the checkpointer's tables (and their indexes) out of autogenerate and `alembic check`."""
    table = getattr(obj, "table", None)
    table_name = name if type_ == "table" else getattr(table, "name", None)
    return table_name not in CHECKPOINTER_TABLES


def _db_url() -> str:
    url = config.attributes.get("db_url") or context.get_x_argument(as_dictionary=True).get("db_url")
    return str(url or get_settings().database_url)


def run_migrations_offline() -> None:
    context.configure(
        url=_db_url(),
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_db_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, include_object=include_object
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
