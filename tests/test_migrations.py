from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DBAPIError


def test_migration_and_audit_append_only(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'migration.db'}"
    monkeypatch.setenv("RUSAMODELS_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", url)
    from rusamodels.config import get_settings
    get_settings.cache_clear()
    command.upgrade(Config("alembic.ini"), "head")
    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    assert {"model_catalog", "model_versions", "artifacts", "upload_sessions",
            "upload_parts", "model_jobs", "audit_events", "alembic_version"} <= tables
    with engine.connect() as connection:
        columns = {column["name"] for column in inspect(engine).get_columns("model_jobs")}
    assert {"attempt_count", "max_attempts", "lease_owner", "lease_epoch",
            "lease_expires_at", "last_heartbeat_at", "next_attempt_at",
            "cancel_requested"} <= columns
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO audit_events (id, actor_subject, actor_role, action, resource, outcome, request_id, metadata) "
            "VALUES ('a1','ci','administrator','test','/','success','req-1','{}')"))
    try:
        with engine.begin() as connection:
            connection.execute(text("UPDATE audit_events SET outcome='changed' WHERE id='a1'"))
    except DBAPIError:
        pass
    else:
        raise AssertionError("audit UPDATE was not blocked")
    engine.dispose()
    get_settings.cache_clear()
