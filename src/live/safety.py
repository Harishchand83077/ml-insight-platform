"""
Safety guard shared by every Stage 1 live-data component (src/live/*).

Stage 1 is a local development stream: a background generator reads the
local customers table and publishes synthetic events to a local RabbitMQ,
and a local consumer writes them to local Postgres. None of this must ever
reach Supabase or Upstash - a misconfigured PG_HOST/SUPABASE_DB_URL-style env
var pointing at a hosted service would otherwise mean the live generator
pollutes real production data, or a migration runs against it by accident.

assert_local_host() refuses anything but localhost/127.0.0.1, checked at the
point each component resolves its target host - not just at import time -
so a call always fails loudly before it would otherwise connect.
"""

import os

ALLOWED_HOSTS = {"localhost", "127.0.0.1"}


class NonLocalHostError(RuntimeError):
    """Raised when a Stage 1 component would otherwise connect to a
    non-local host. Stage 1 must never touch Supabase or Upstash."""


def assert_local_host(host, what):
    if host not in ALLOWED_HOSTS:
        raise NonLocalHostError(
            f"refusing to start: {what} host '{host}' is not local "
            f"(allowed: {sorted(ALLOWED_HOSTS)}). Stage 1 live-data "
            "components must never touch Supabase or Upstash."
        )
    return host


def local_pg_dsn():
    """Postgres connection dict for Stage 1 components. Deliberately not
    src.common.db.get_database_url(): that helper prefers DATABASE_URL /
    SUPABASE_DB_URL, which in a deployed-style .env point at Supabase - this
    always uses the local PG_HOST/PG_PORT/... vars (same convention as
    event_consumer.py, load_static_data.py), and refuses to proceed unless
    the resolved host is local."""
    host = assert_local_host(os.environ.get("PG_HOST", "localhost"), "database")
    return {
        "host": host,
        "port": os.environ.get("PG_PORT", "5432"),
        "dbname": os.environ.get("PG_DATABASE", "ml_insight"),
        "user": os.environ.get("PG_USER", "postgres"),
        "password": os.environ.get("PG_PASSWORD", "devpassword"),
    }


def local_rabbitmq_host():
    host = os.environ.get("RABBITMQ_HOST", "localhost")
    return assert_local_host(host, "RabbitMQ")
