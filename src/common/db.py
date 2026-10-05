"""
Shared Postgres connection helper. get_database_url() returns a single
connection string usable both for psycopg2.connect() and for
psycopg2.pool.*ConnectionPool() (which take the same dsn as their first
positional arg after min/max size) - callers don't need to build a DSN
dict themselves, and there's exactly one code path whether the caller
wants a single connection or a pool.

DATABASE_URL / SUPABASE_DB_URL, when set, are used as-is (any valid
libpq connection string/URI) - psycopg2 natively supports postgresql://
URIs, including query-string options like sslmode=require (which
Supabase requires), so there's deliberately no manual URL parsing here
that could silently drop those options.

This is what makes pointing a deployment at a different Postgres (local
dev -> Supabase, or anywhere else) just an env var change rather than a
code change: set DATABASE_URL (or SUPABASE_DB_URL) and nothing in the
calling code changes.
"""

import logging
import os
from contextlib import contextmanager
from urllib.parse import quote

import psycopg2
import psycopg2.pool
from dotenv import load_dotenv

logger = logging.getLogger("db")

load_dotenv()  # so a standalone script (e.g. scripts/migrate_to_supabase.py)
# picks up .env without having to remember to call this itself

# Matches the PG_HOST/PG_PORT/... convention already used throughout this
# project (load_static_data.py, build_features.py, ...). Used only when
# neither DATABASE_URL nor SUPABASE_DB_URL is set.
LOCAL_DEFAULTS = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": os.environ.get("PG_PORT", "5432"),
    "dbname": os.environ.get("PG_DATABASE", "ml_insight"),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", "devpassword"),
}


def get_database_url():
    """The main app's Postgres connection target, as a single connection
    string - despite LOCAL_DEFAULTS' name, this is what a real deployment
    (e.g. Render) uses too. Checks DATABASE_URL, then SUPABASE_DB_URL (so
    the same var a deployment's dashboard sets under that name just
    works, no second var to wire up), then falls back to a URL built from
    the individual PG_HOST/PG_PORT/... vars (local dev default)."""
    url = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")
    if url:
        return url

    d = LOCAL_DEFAULTS
    user = quote(d["user"], safe="")
    password = quote(d["password"], safe="")
    return f"postgresql://{user}:{password}@{d['host']}:{d['port']}/{d['dbname']}"


def connect_local():
    """A single ready psycopg2 connection to get_database_url()'s target."""
    return psycopg2.connect(get_database_url())


def connect_supabase():
    """Connects to Supabase via SUPABASE_DB_URL specifically - always
    required, no fallback (there's no sensible local default for a hosted
    DB). Used by scripts/migrate_to_supabase.py, which needs the local and
    Supabase connections simultaneously so can't use the generic
    connect_local() for both. If your Supabase password contains special
    characters, percent-encode them in the URL (e.g. urllib.parse.quote)."""
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError(
            "SUPABASE_DB_URL is not set. Add it to your .env, e.g.:\n"
            "SUPABASE_DB_URL=postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres"
        )
    return psycopg2.connect(url)


# --- Pooled connections -------------------------------------------------
#
# Supabase closes idle connections on its side. psycopg2 doesn't notice
# until the next query fails with OperationalError ("server closed the
# connection unexpectedly"), and SimpleConnectionPool will hand that dead
# connection out again on the next getconn(). The helpers below handle it.
#
# Retry policy. The split between reads and writes is deliberate:
#
# - Reads (feature lookup, user lookups) are idempotent. If the connection
#   dies, run_read() discards it and retries the query once on a fresh one.
#
# - Writes (signup, feedback, audit) are NOT retried after the statement
#   has been sent. The connection can die after the server has committed
#   but before the client receives the acknowledgement, so a blind retry
#   could insert the same row twice. Instead, checkout_for_write() pings
#   the connection before any write is sent and replaces it if the ping
#   fails. Nothing has been written at that point, so swapping connections
#   can't duplicate anything.
#
# TCP keepalives make a dead connection show up sooner, so the pool
# notices an idle drop before a user request does.

KEEPALIVE_OPTIONS = {
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
}

# Errors meaning the connection itself is gone, as opposed to the statement
# failing (IntegrityError, etc.), which leaves the connection usable.
CONNECTION_LOST_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)


def make_pool(minconn, maxconn):
    """A SimpleConnectionPool whose connections use TCP keepalives."""
    return psycopg2.pool.SimpleConnectionPool(minconn, maxconn, get_database_url(), **KEEPALIVE_OPTIONS)


def _return_connection(pool, conn, lost):
    # close=True for a dead connection, so the pool doesn't keep it. For
    # any other error, plain putconn: psycopg2 rolls back an open transaction.
    pool.putconn(conn, close=lost)


def _ping(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except CONNECTION_LOST_ERRORS:
        return False


def run_read(pool, query):
    """Runs query(conn) and returns its result. Read-only, idempotent use
    only: if the connection is lost, it is discarded and the query runs
    once more on a fresh connection. Do not use this for writes."""
    for attempt in (1, 2):
        conn = pool.getconn()
        try:
            result = query(conn)
        except CONNECTION_LOST_ERRORS:
            _return_connection(pool, conn, lost=True)
            if attempt == 2:
                raise
            logger.warning("Pooled connection lost during read; retrying once on a fresh connection")
            continue
        except Exception:
            _return_connection(pool, conn, lost=False)
            raise
        _return_connection(pool, conn, lost=False)
        return result


@contextmanager
def checkout_for_write(pool):
    """Context manager yielding a pooled connection that has been pinged
    before use. The caller runs its write and commits inside the block. A
    lost connection is discarded and the error propagates, with no retry of
    the write itself (see the policy comment above)."""
    conn = None
    for _ in range(2):
        candidate = pool.getconn()
        if _ping(candidate):
            conn = candidate
            break
        _return_connection(pool, candidate, lost=True)
    if conn is None:
        raise psycopg2.OperationalError("could not get a live connection from the pool")

    try:
        yield conn
    except CONNECTION_LOST_ERRORS:
        _return_connection(pool, conn, lost=True)
        raise
    except Exception:
        _return_connection(pool, conn, lost=False)
        raise
    _return_connection(pool, conn, lost=False)
