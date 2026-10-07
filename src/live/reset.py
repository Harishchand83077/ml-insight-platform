"""
Stage 2 event-table reset: truncates and reloads the three local event
tables from the seed-42 synthetic CSVs (data/synthetic/*.csv - the exact,
reproducible output of src/data_gen/generate_events.py, verified
byte-identical across re-runs in Stage 1), so a live/drift experiment can
always be undone back to the known baseline.

table_checksum() lets a caller prove two resets land on identical state: it
hashes every row (in id order, made deterministic by TRUNCATE ... RESTART
IDENTITY + a fixed CSV row order on reload) rather than relying on row
counts alone, which would miss a reset that reloaded the same number of
different or reordered rows.
"""

import os

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

EVENT_TABLES = ["login_events", "support_tickets", "feature_usage_logs"]
SYNTHETIC_CSV_DIR = "data/synthetic"

TABLE_COLUMNS = {
    "login_events": ["customer_id", "timestamp", "session_duration_seconds", "device"],
    "support_tickets": ["customer_id", "timestamp", "category", "resolved", "resolution_time_hours"],
    "feature_usage_logs": ["customer_id", "timestamp", "feature_name", "usage_count"],
}


def _load_rows(table):
    csv_path = os.path.join(SYNTHETIC_CSV_DIR, f"{table}.csv")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df[TABLE_COLUMNS[table]].replace({np.nan: None})
    return [tuple(row) for row in df.itertuples(index=False, name=None)]


def reset_event_tables(conn):
    """Truncates and reloads all three local event tables from the seed-42
    synthetic CSVs. event_id stays NULL on every reloaded row, matching the
    original batch producer's baseline - no live-generator events have run
    against this data yet."""
    with conn.cursor() as cur:
        for table in EVENT_TABLES:
            cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY")
            columns = TABLE_COLUMNS[table]
            insert_sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s"
            execute_values(cur, insert_sql, _load_rows(table))
    conn.commit()


def table_checksum(conn, table, order_by="id"):
    """A single md5 hex digest of every row (all columns) in order_by order
    - "id" for the event tables (RESTART IDENTITY makes it deterministic
    across resets), "customer_id" for customer_features (no serial id
    there). Two resets of identical input data produce identical
    checksums; a reset that silently reloaded different, missing, or
    reordered rows would not match."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT md5(string_agg(t.*::text, '|' ORDER BY t.{order_by})) FROM {table} t")
        return cur.fetchone()[0]
