"""
Consumes from all three event queues and writes each message into Postgres
(the postgres-ml container / ml_insight database), one table per event type.
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

INSERT_STATEMENTS = {
    "login_events": """
        INSERT INTO login_events (customer_id, timestamp, session_duration_seconds, device)
        VALUES (%(customer_id)s, %(timestamp)s, %(session_duration_seconds)s, %(device)s)
    """,
    "support_tickets": """
        INSERT INTO support_tickets (customer_id, timestamp, category, resolved, resolution_time_hours)
        VALUES (%(customer_id)s, %(timestamp)s, %(category)s, %(resolved)s, %(resolution_time_hours)s)
    """,
    "feature_usage_logs": """
        INSERT INTO feature_usage_logs (customer_id, timestamp, feature_name, usage_count)
        VALUES (%(customer_id)s, %(timestamp)s, %(feature_name)s, %(usage_count)s)
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
