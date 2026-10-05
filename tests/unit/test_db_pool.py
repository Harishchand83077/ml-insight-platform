"""
Unit tests for the pooled-connection helpers in src/common/db.py and their
use in auth.py, feedback.py, and feature_cache.py. No database: the pool
and connections are fakes. A fake connection raises OperationalError on its
first use, the way a Supabase-closed idle connection does.
"""

from unittest.mock import MagicMock, patch

import psycopg2
import pytest

from src.common import db
from src.serving import auth, feature_cache, feedback


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append(sql)
        if self.conn.fail_on_execute > 0:
            self.conn.fail_on_execute -= 1
            raise psycopg2.OperationalError("server closed the connection unexpectedly")

    def fetchone(self):
        return self.conn.row


class FakeConn:
    """fail_on_execute: how many execute() calls raise OperationalError first."""

    def __init__(self, name, fail_on_execute=0, row=None):
        self.name = name
        self.fail_on_execute = fail_on_execute
        self.row = row
        self.executed = []
        self.commits = 0

    def cursor(self, cursor_factory=None):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class FakePool:
    def __init__(self, *conns):
        self.free = list(conns)
        self.returned = []  # (conn, close)

    def getconn(self):
        return self.free.pop(0)

    def putconn(self, conn, close=False):
        self.returned.append((conn, close))


# --- run_read: idempotent reads, retried once -----------------------------

class TestRunRead:
    def test_retries_once_on_a_fresh_connection_after_a_lost_connection(self):
        dead = FakeConn("dead", fail_on_execute=1)
        fresh = FakeConn("fresh", row=("u1", "a@example.com"))
        pool = FakePool(dead, fresh)

        def query(conn):
            with conn.cursor() as cur:
                cur.execute("SELECT id, email FROM users WHERE id = %s", ("u1",))
                return cur.fetchone()

        result = db.run_read(pool, query)

        assert result == ("u1", "a@example.com")
        assert pool.returned == [(dead, True), (fresh, False)]

    def test_second_lost_connection_is_raised(self):
        pool = FakePool(FakeConn("a", fail_on_execute=1), FakeConn("b", fail_on_execute=1))

        def query(conn):
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

        with pytest.raises(psycopg2.OperationalError):
            db.run_read(pool, query)
        assert all(close for _, close in pool.returned)

    def test_non_connection_error_is_not_retried(self):
        conn = FakeConn("ok")
        pool = FakePool(conn)
        calls = []

        def query(c):
            calls.append(c)
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            db.run_read(pool, query)
        assert len(calls) == 1
        assert pool.returned == [(conn, False)]


# --- checkout_for_write: ping first, never retry the write -----------------

class TestCheckoutForWrite:
    def test_dead_connection_is_replaced_by_a_ping_before_the_write(self):
        dead = FakeConn("dead", fail_on_execute=1)  # the ping fails
        fresh = FakeConn("fresh")
        pool = FakePool(dead, fresh)

        with db.checkout_for_write(pool) as conn:
            assert conn is fresh
            with conn.cursor() as cur:
                cur.execute("INSERT INTO feedback VALUES (1)")
            conn.commit()

        assert pool.returned == [(dead, True), (fresh, False)]
        assert fresh.commits == 1
        assert "INSERT INTO feedback VALUES (1)" in fresh.executed

    def test_write_is_not_retried_when_it_fails_mid_flight(self):
        # Ping succeeds, the INSERT then dies. The write must not run again.
        conn = FakeConn("c", fail_on_execute=0)
        pool = FakePool(conn)
        body_runs = []

        with pytest.raises(psycopg2.OperationalError):
            with db.checkout_for_write(pool) as c:
                body_runs.append(1)
                c.fail_on_execute = 1  # the INSERT's execute raises
                with c.cursor() as cur:
                    cur.execute("INSERT INTO audit_logs VALUES (1)")

        assert len(body_runs) == 1
        assert pool.returned == [(conn, True)]

    def test_gives_up_when_no_live_connection_is_found(self):
        pool = FakePool(FakeConn("a", fail_on_execute=1), FakeConn("b", fail_on_execute=1))
        body_runs = []

        with pytest.raises(psycopg2.OperationalError):
            with db.checkout_for_write(pool):
                body_runs.append(1)
        assert body_runs == []

    def test_statement_error_returns_connection_to_pool_without_closing(self):
        conn = FakeConn("c")
        pool = FakePool(conn)

        with pytest.raises(ValueError):
            with db.checkout_for_write(pool):
                raise ValueError("constraint-style failure, connection still fine")

        assert pool.returned == [(conn, False)]


# --- keepalives on the pool's connections ----------------------------------

class TestMakePool:
    def test_pool_is_created_with_tcp_keepalives(self):
        with patch.object(db.psycopg2.pool, "SimpleConnectionPool") as pool_cls, \
             patch.object(db, "get_database_url", return_value="postgresql://example"):
            db.make_pool(1, 5)

        args, kwargs = pool_cls.call_args
        assert args == (1, 5, "postgresql://example")
        assert kwargs["keepalives"] == 1
        assert kwargs["keepalives_idle"] == 30
        assert kwargs["keepalives_interval"] == 10
        assert kwargs["keepalives_count"] == 3


# --- the three call sites, each with a connection that fails on first use --

class TestAuthReadRecovers:
    def test_get_user_by_id_recovers_from_a_closed_idle_connection(self):
        pool = FakePool(
            FakeConn("dead", fail_on_execute=1),
            FakeConn("fresh", row=("user-1", "a@example.com")),
        )
        with patch.object(auth, "_get_pool", return_value=pool):
            user = auth.get_user_by_id("user-1")

        assert user == {"id": "user-1", "email": "a@example.com"}


class TestFeedbackWriteDoesNotDuplicate:
    def test_store_feedback_inserts_exactly_once_after_a_dead_connection(self):
        dead = FakeConn("dead", fail_on_execute=1)  # ping fails before any INSERT
        fresh = FakeConn("fresh")
        pool = FakePool(dead, fresh)

        with patch.object(feedback, "_get_pool", return_value=pool):
            feedback.store_feedback("user-1", "session-1", "hello", "up")

        inserts = [s for s in fresh.executed if "INSERT INTO feedback" in s]
        assert len(inserts) == 1
        assert fresh.commits == 1


class TestFeatureCacheReadRecovers:
    def test_get_customer_features_recovers_on_a_fresh_connection(self):
        pool = FakePool(
            FakeConn("dead", fail_on_execute=1),
            FakeConn("fresh", row={"customer_id": "7590-VHVEG", "tenure": 3}),
        )
        fake_redis = MagicMock()
        fake_redis.get.return_value = None  # cache miss, so the Postgres path runs

        with patch.object(feature_cache, "_get_pool", return_value=pool), \
             patch.object(feature_cache, "get_redis_client", return_value=fake_redis):
            features, hit = feature_cache.get_customer_features("7590-VHVEG")

        assert hit is False
        assert features["tenure"] == 3
        fake_redis.setex.assert_called_once()
