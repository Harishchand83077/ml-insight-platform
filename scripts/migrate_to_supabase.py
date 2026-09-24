"""
One-time migration: copies `customers` and `customer_features` from local
Postgres to Supabase Postgres. The raw event tables (login_events,
support_tickets, feature_usage_logs) are deliberately NOT migrated -
customer_features already has everything the app needs, aggregated from
them during build_features.py.

Source: local Postgres, via src/common/db.py:connect_local() (DATABASE_URL
if set, else the individual PG_HOST/PG_PORT/... vars every other script in
this project already uses).
Destination: Supabase, via connect_supabase() - reads SUPABASE_DB_URL only,
no default. Never hardcode the connection string; add it to your local
.env (see .env.example) - it's gitignored, never committed.

The CREATE TABLE statements below mirror the schemas defined in
load_static_data.py (customers) and build_features.py (customer_features)
- keep them in sync if those change.

Run: python scripts/migrate_to_supabase.py
"""

import sys
from pathlib import Path

# scripts/ lives outside the src package, so add the project root to
# sys.path - running this file directly only puts scripts/ itself on the
# path, and `from src...` would otherwise fail.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psycopg2.extras import execute_values  # noqa: E402

from src.common.db import connect_local, connect_supabase  # noqa: E402

TABLES = ["customers", "customer_features"]

CREATE_TABLE_SQL = {
    "customers": """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            gender TEXT,
            senior_citizen INTEGER,
            partner TEXT,
            dependents TEXT,
            tenure INTEGER,
            phone_service TEXT,
            multiple_lines TEXT,
            internet_service TEXT,
            online_security TEXT,
            online_backup TEXT,
            device_protection TEXT,
            tech_support TEXT,
            streaming_tv TEXT,
            streaming_movies TEXT,
            contract TEXT,
            paperless_billing TEXT,
            payment_method TEXT,
            monthly_charges NUMERIC,
            total_charges NUMERIC,
            churn TEXT
        )
    """,
    "customer_features": """
        CREATE TABLE IF NOT EXISTS customer_features (
            customer_id TEXT PRIMARY KEY,
            tenure INTEGER,
            monthly_charges NUMERIC,
            contract TEXT,
            payment_method TEXT,
            internet_service TEXT,
            senior_citizen INTEGER,
            partner TEXT,
            dependents TEXT,
            num_addons_active INTEGER,
            total_logins_90d INTEGER,
            recent_30d_vs_older_60d_ratio NUMERIC,
            avg_session_duration_recent_30d NUMERIC,
            num_tickets INTEGER,
            pct_unresolved NUMERIC,
            avg_resolution_time_hours NUMERIC,
            avg_usage_count NUMERIC
        )
    """,
}


def fetch_table(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {table}")
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return columns, rows


def load_table(conn, table, columns, rows):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL[table])
        cur.execute(f"TRUNCATE TABLE {table}")
        if rows:
            insert_sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s"
            execute_values(cur, insert_sql, rows)
    conn.commit()


def row_count(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return cur.fetchone()[0]


def main():
    print("Connecting to local Postgres...")
    local_conn = connect_local()

    print("Connecting to Supabase...")
    try:
        supabase_conn = connect_supabase()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    mismatches = []
    try:
        for table in TABLES:
            print(f"\n--- {table} ---")
            columns, rows = fetch_table(local_conn, table)
            print(f"Fetched {len(rows)} rows from local {table}")

            load_table(supabase_conn, table, columns, rows)
            print(f"Loaded into Supabase {table}")

            local_count = row_count(local_conn, table)
            supabase_count = row_count(supabase_conn, table)
            match = local_count == supabase_count
            print(f"Row count check: local={local_count} supabase={supabase_count} [{'OK' if match else 'MISMATCH'}]")
            if not match:
                mismatches.append(table)
    finally:
        local_conn.close()
        supabase_conn.close()

    if mismatches:
        print(f"\nMigration FAILED - row count mismatch for: {', '.join(mismatches)}")
        sys.exit(1)

    print("\nMigration complete - all row counts match.")


if __name__ == "__main__":
    main()
