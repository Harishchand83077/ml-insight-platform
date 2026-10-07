"""
Global guard: no unit test may reach a real Postgres or Redis service.

Patches psycopg2.connect and redis.Redis.from_url for the duration of every
unit test, so a test that forgets to mock the right thing - e.g. patching
run_read() without noticing that tools.py's callers also call
get_shared_pool() first, which eagerly opens real connections via
psycopg2.pool.SimpleConnectionPool.__init__ - fails immediately with a
clear message, instead of hanging on, or (worse) quietly succeeding
against, a real local, Supabase, or Upstash service.

A test that mocks psycopg2.connect or redis.Redis.from_url itself (or
anything that calls them - get_shared_pool, make_pool, get_redis_client,
_get_pool, ...) is unaffected: the guard only fires if something gets all
the way down to the real library call, which a properly mocked test never
does. unittest.mock.patch applied inside a test always saves and restores
whatever was in place when it was entered, including this fixture's patch,
so nesting is safe regardless of which one "wins" - the test's own patch
always does, for its duration.
"""

import psycopg2
import pytest
import redis


class RealServiceConnectionAttempted(AssertionError):
    pass


@pytest.fixture(autouse=True)
def _forbid_real_service_connections(monkeypatch):
    def _boom_postgres(*args, **kwargs):
        raise RealServiceConnectionAttempted(
            "unit test attempted a real service connection (psycopg2.connect) - "
            "mock the pool getter (e.g. get_shared_pool, _get_pool) or a fake pool, "
            "not just run_read/checkout_for_write: tools.py's callers construct the "
            "real pool (which connects immediately) before run_read ever runs"
        )

    def _boom_redis(*args, **kwargs):
        raise RealServiceConnectionAttempted(
            "unit test attempted a real service connection (redis.Redis.from_url) - "
            "mock get_redis_client (or _get_redis_client), not just the function that calls it"
        )

    monkeypatch.setattr(psycopg2, "connect", _boom_postgres)
    monkeypatch.setattr(redis.Redis, "from_url", _boom_redis)
