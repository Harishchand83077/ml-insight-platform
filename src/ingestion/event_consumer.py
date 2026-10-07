"""
Consumes from all three event queues and writes each message into Postgres
(the postgres-ml container / ml_insight database), one table per event type.

Order of operations per message, which is what makes a restart or a
redelivery safe: (1) callback() executes the INSERT within the open,
uncommitted transaction - "attempted" but not yet durable, and not yet
acked, so the broker still owns redelivery if the process dies right here;
(2) once COMMIT_EVERY messages (or FLUSH_IDLE_SECONDS of idle time) have
accumulated, flush_pending() calls conn.commit() - now durable; (3) only
after that commit succeeds does flush_pending() ack (multiple=True, every
message up to and including the last one in the batch). A crash between (1)
and (2) loses nothing: the uncommitted INSERTs roll back with the
connection, and the unacked messages are redelivered. A crash between (2)
and (3) also loses nothing, but CAN cause redelivery of messages that were
already committed - that's exactly what the event_id column + unique index
+ ON CONFLICT (event_id) DO NOTHING (below) exists to make harmless: a
redelivered message lands on the same row, not a new one.
"""

import json
import logging
import os
import time

import pika
import psycopg2

QUEUES = ["login_events", "support_tickets", "feature_usage_logs"]
LOG_EVERY = 1000
COMMIT_EVERY = 200  # commit in batches instead of per-row; committing every
# single insert forces a disk sync per message and caps throughput at a few
# hundred rows/sec. Messages are ack'd only after a successful commit, so a
# crash mid-batch just redelivers the still-unacked messages - no data is
# lost, just possibly reprocessed.
FLUSH_IDLE_SECONDS = 2  # also flush a partial batch after this long without a
# new message, so a small trickle of messages (or the tail of a burst, fewer
# than COMMIT_EVERY) doesn't sit unacked/uncommitted indefinitely.

PG_DSN = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": os.environ.get("PG_PORT", "5432"),
    "dbname": os.environ.get("PG_DATABASE", "ml_insight"),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", "devpassword"),
}

TABLE_SCHEMAS = {
    "login_events": """
        CREATE TABLE IF NOT EXISTS login_events (
            id SERIAL PRIMARY KEY,
            customer_id TEXT,
            timestamp TIMESTAMP,
            session_duration_seconds INTEGER,
            device TEXT
        )
    """,
    "support_tickets": """
        CREATE TABLE IF NOT EXISTS support_tickets (
            id SERIAL PRIMARY KEY,
            customer_id TEXT,
            timestamp TIMESTAMP,
            category TEXT,
            resolved BOOLEAN,
            resolution_time_hours DOUBLE PRECISION
        )
    """,
    "feature_usage_logs": """
        CREATE TABLE IF NOT EXISTS feature_usage_logs (
            id SERIAL PRIMARY KEY,
            customer_id TEXT,
            timestamp TIMESTAMP,
            feature_name TEXT,
            usage_count INTEGER
        )
    """,
}

# event_id is NULL for messages from the older batch producer
# (event_producer.py), which predates it - Postgres allows any number of
# NULLs through a unique index, so those rows are unaffected. It's a real
# uuid for messages from the live generator (src/live/generator.py), and
# ON CONFLICT (event_id) DO NOTHING is what makes a redelivered message
# (after a consumer restart, or an unacked requeue) a no-op instead of a
# duplicate row - see migrate_event_tables() in main() below for the column
# and unique index this relies on.
INSERT_STATEMENTS = {
    "login_events": """
        INSERT INTO login_events (event_id, customer_id, timestamp, session_duration_seconds, device)
        VALUES (%(event_id)s, %(customer_id)s, %(timestamp)s, %(session_duration_seconds)s, %(device)s)
        ON CONFLICT (event_id) DO NOTHING
    """,
    "support_tickets": """
        INSERT INTO support_tickets (event_id, customer_id, timestamp, category, resolved, resolution_time_hours)
        VALUES (%(event_id)s, %(customer_id)s, %(timestamp)s, %(category)s, %(resolved)s, %(resolution_time_hours)s)
        ON CONFLICT (event_id) DO NOTHING
    """,
    "feature_usage_logs": """
        INSERT INTO feature_usage_logs (event_id, customer_id, timestamp, feature_name, usage_count)
        VALUES (%(event_id)s, %(customer_id)s, %(timestamp)s, %(feature_name)s, %(usage_count)s)
        ON CONFLICT (event_id) DO NOTHING
    """,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("event_consumer")


def flush_pending(conn, channel, pending):
    if pending["count"] > 0:
        conn.commit()
        channel.basic_ack(delivery_tag=pending["last_tag"], multiple=True)
        pending["count"] = 0
        pending["last_flush"] = time.monotonic()


def make_callback(conn, channel, queue_name, counters, pending):
    insert_sql = INSERT_STATEMENTS[queue_name]

    def callback(ch, method, properties, body):
        message = json.loads(body)
        message.setdefault("event_id", None)  # older batch producer's messages have none
        with conn.cursor() as cur:
            cur.execute(insert_sql, message)

        pending["count"] += 1
        pending["last_tag"] = method.delivery_tag

        counters[queue_name] += 1
        counters["total"] += 1

        if pending["count"] >= COMMIT_EVERY:
            flush_pending(conn, channel, pending)

        if counters["total"] % LOG_EVERY == 0:
            logger.info(
                "Consumed %d messages total (login_events=%d, support_tickets=%d, feature_usage_logs=%d)",
                counters["total"],
                counters["login_events"],
                counters["support_tickets"],
                counters["feature_usage_logs"],
            )

    return callback


def main():
    conn = psycopg2.connect(**PG_DSN)
    with conn.cursor() as cur:
        for schema in TABLE_SCHEMAS.values():
            cur.execute(schema)
        # Idempotent migration, run every startup so it's never missed: adds
        # event_id (absent from tables created before the live generator
        # existed) and the unique index ON CONFLICT (event_id) above needs.
        # Self-contained here (no cross-import) so this stays runnable as a
        # plain script, same as every other file in src/ingestion and
        # src/pipelines; src/live/migrate_event_tables.py is the same
        # migration for standalone/live-generator use.
        for queue_name in QUEUES:
            cur.execute(f"ALTER TABLE IF EXISTS {queue_name} ADD COLUMN IF NOT EXISTS event_id UUID")
            cur.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {queue_name}_event_id_key ON {queue_name} (event_id)"
            )
    conn.commit()

    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
    channel = connection.channel()
    channel.basic_qos(prefetch_count=COMMIT_EVERY * 2)

    counters = {"total": 0, "login_events": 0, "support_tickets": 0, "feature_usage_logs": 0}
    pending = {"count": 0, "last_tag": None, "last_flush": time.monotonic()}

    for queue_name in QUEUES:
        channel.queue_declare(queue=queue_name, durable=True)
        channel.basic_consume(
            queue=queue_name,
            on_message_callback=make_callback(conn, channel, queue_name, counters, pending),
        )

    logger.info("Waiting for messages on %s. To exit press CTRL+C", QUEUES)
    try:
        while True:
            connection.process_data_events(time_limit=1)
            if pending["count"] > 0 and time.monotonic() - pending["last_flush"] > FLUSH_IDLE_SECONDS:
                flush_pending(conn, channel, pending)
    except KeyboardInterrupt:
        pass
    finally:
        flush_pending(conn, channel, pending)
        connection.close()
        conn.close()
        logger.info("Stopped. Final counts: %s", counters)


if __name__ == "__main__":
    main()
