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

import os
from urllib.parse import quote

import psycopg2
from dotenv import load_dotenv

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
