"""
Stage 1 migration: adds an event_id column and a unique index to each of the
three local event tables, so event_consumer.py's
"INSERT ... ON CONFLICT (event_id) DO NOTHING" can dedupe a message that
gets redelivered after a consumer restart or an unacked requeue.

Idempotent (ADD COLUMN IF NOT EXISTS / CREATE UNIQUE INDEX IF NOT EXISTS):
safe to run against a brand-new table, an already-migrated one, or one that
already holds rows from the older batch producer (event_producer.py) - those
existing rows get event_id = NULL, and Postgres's unique index allows any
number of NULLs, so they're unaffected.

Run standalone with:
    python -m src.live.migrate_event_tables

(as -m, not as a plain script: it imports src.live.safety, which only
resolves when the repo root is on sys.path - the same reason the FastAPI
apps in this repo are run via `uvicorn module:app` rather than as a plain
script.) Also called by src/live/control_api.py before starting the
generator, so a forgotten manual run can't leave the column missing.
"""

import psycopg2

from src.live.safety import local_pg_dsn

EVENT_TABLES = ["login_events", "support_tickets", "feature_usage_logs"]


def migrate_event_tables(conn):
    with conn.cursor() as cur:
        for table in EVENT_TABLES:
            cur.execute(f"ALTER TABLE IF EXISTS {table} ADD COLUMN IF NOT EXISTS event_id UUID")
            cur.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {table}_event_id_key ON {table} (event_id)"
            )
    conn.commit()


def main():
    conn = psycopg2.connect(**local_pg_dsn())
    try:
        migrate_event_tables(conn)
    finally:
        conn.close()
    print(f"Migrated event_id column + unique index on: {', '.join(EVENT_TABLES)}")


if __name__ == "__main__":
    main()
