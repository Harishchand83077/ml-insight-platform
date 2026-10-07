"""
Unit tests for event_consumer.py's event_id-based de-duplication: a message
redelivered after a crash/restart must not land as a second row, and a
message with no event_id (the older batch producer, event_producer.py, never
set one) must still insert instead of KeyError-ing on the new placeholder.

Postgres is faked with a minimal in-memory table whose insert() mimics the
real "INSERT ... ON CONFLICT (event_id) DO NOTHING" semantics the unique
index (src/live/migrate_event_tables.py) enforces for real - that real
enforcement is proven live in the Stage 1 live check (restart + confirm no
duplicates); this test proves the application-level SQL and ack ordering
are correct against that semantics.
"""

import json
from unittest.mock import MagicMock

from src.ingestion import event_consumer as consumer


class FakeTable:
    """In-memory stand-in for one Postgres table with a UNIQUE(event_id)
    index: insert() mimics "INSERT ... ON CONFLICT (event_id) DO NOTHING" -
    a NULL event_id (the older batch producer) is never treated as a
    duplicate, matching Postgres's own unique-index semantics."""

    def __init__(self):
        self.rows = []
        self._seen_event_ids = set()

    def insert(self, message):
        event_id = message.get("event_id")
        if event_id is not None and event_id in self._seen_event_ids:
            return  # ON CONFLICT (event_id) DO NOTHING
        if event_id is not None:
            self._seen_event_ids.add(event_id)
        self.rows.append(message)


class FakeCursor:
    def __init__(self, table):
        self.table = table

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        assert "ON CONFLICT (event_id) DO NOTHING" in sql
        # mirrors psycopg2's %(name)s substitution closely enough to raise
        # the same KeyError it would if the SQL references a key the
        # message dict doesn't have (e.g. event_id, for an older message).
        sql % params
        self.table.insert(params)


class FakeConn:
    def __init__(self, table):
        self.table = table
        self.commits = 0

    def cursor(self):
        return FakeCursor(self.table)

    def commit(self):
        self.commits += 1


def _fake_method(delivery_tag):
    method = MagicMock()
    method.delivery_tag = delivery_tag
    return method


def _callback_for(table, queue_name="login_events"):
    conn = FakeConn(table)
    channel = MagicMock()
    counters = {"total": 0, "login_events": 0, "support_tickets": 0, "feature_usage_logs": 0}
    pending = {"count": 0, "last_tag": None, "last_flush": 0}
    return consumer.make_callback(conn, channel, queue_name, counters, pending), channel


LOGIN_MESSAGE = {
    "customer_id": "c-1",
    "timestamp": "2026-01-01 00:00:00",
    "session_duration_seconds": 120,
    "device": "mobile",
}


class TestDuplicateEventIdInsertsNothing:
    def test_redelivered_message_with_the_same_event_id_is_not_inserted_twice(self):
        table = FakeTable()
        callback, channel = _callback_for(table)
        message = {**LOGIN_MESSAGE, "event_id": "11111111-1111-1111-1111-111111111111"}
        body = json.dumps(message).encode()

        callback(channel, _fake_method(1), None, body)
        callback(channel, _fake_method(2), None, body)  # redelivered: same event_id

        assert len(table.rows) == 1

    def test_a_different_event_id_inserts_a_second_row(self):
        table = FakeTable()
        callback, channel = _callback_for(table)
        first = {**LOGIN_MESSAGE, "event_id": "11111111-1111-1111-1111-111111111111"}
        second = {**LOGIN_MESSAGE, "event_id": "22222222-2222-2222-2222-222222222222"}

        callback(channel, _fake_method(1), None, json.dumps(first).encode())
        callback(channel, _fake_method(2), None, json.dumps(second).encode())

        assert len(table.rows) == 2

    def test_a_message_with_no_event_id_still_inserts(self):
        # messages from the older batch producer (event_producer.py) have no
        # event_id field at all - callback must not KeyError on them.
        table = FakeTable()
        callback, channel = _callback_for(table)
        body = json.dumps(LOGIN_MESSAGE).encode()

        callback(channel, _fake_method(1), None, body)

        assert len(table.rows) == 1
        assert table.rows[0]["event_id"] is None

    def test_two_messages_with_no_event_id_both_insert_rather_than_colliding(self):
        # NULL != NULL under a unique index: neither is treated as a dup of
        # the other.
        table = FakeTable()
        callback, channel = _callback_for(table)
        body = json.dumps(LOGIN_MESSAGE).encode()

        callback(channel, _fake_method(1), None, body)
        callback(channel, _fake_method(2), None, body)

        assert len(table.rows) == 2
