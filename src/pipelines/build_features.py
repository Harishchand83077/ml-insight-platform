"""
Builds a model-ready feature table by combining the `customers` table with
the three event tables (login_events, support_tickets, feature_usage_logs)
in Postgres. Writes the result to a new `customer_features` table and to
data/processed/features.csv for inspection.

"Recent" is relative to END_DATE below, which must match the reference date
used by src/data_gen/generate_events.py when it generated the synthetic
event timestamps (2024-06-30) - otherwise the 30/60-day windows wouldn't
line up with the actual event data.

Customers with zero rows in a given event table get 0 for every feature
derived from that table (not NULL), since "no events" is itself meaningful
signal (e.g. a customer who never logged in).
"""

import os

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

END_DATE = pd.Timestamp("2024-06-30")
FEATURES_CSV = "data/processed/features.csv"

PG_DSN = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": os.environ.get("PG_PORT", "5432"),
    "dbname": os.environ.get("PG_DATABASE", "ml_insight"),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", "devpassword"),
}

ADDON_COLUMNS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

FEATURES_TABLE = "customer_features"
STAGING_TABLE = "customer_features_new"

CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS {table} (
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
"""

FEATURE_COLUMNS = [
    "customer_id",
    "tenure",
    "monthly_charges",
    "contract",
    "payment_method",
    "internet_service",
    "senior_citizen",
    "partner",
    "dependents",
    "num_addons_active",
    "total_logins_90d",
    "recent_30d_vs_older_60d_ratio",
    "avg_session_duration_recent_30d",
    "num_tickets",
    "pct_unresolved",
    "avg_resolution_time_hours",
    "avg_usage_count",
]

EVENT_DERIVED_COLUMNS = [
    "total_logins_90d",
    "recent_30d_vs_older_60d_ratio",
    "avg_session_duration_recent_30d",
    "num_tickets",
    "pct_unresolved",
    "avg_resolution_time_hours",
    "avg_usage_count",
]


def build_customer_features(customers):
    customers = customers.copy()
    customers["num_addons_active"] = customers[ADDON_COLUMNS].eq("Yes").sum(axis=1)

    keep = [
        "customer_id",
        "tenure",
        "monthly_charges",
        "contract",
        "payment_method",
        "internet_service",
        "senior_citizen",
        "partner",
        "dependents",
        "num_addons_active",
    ]
    return customers[keep]


def build_login_features(login_events, as_of=END_DATE):
    """as_of anchors the 'recent 30d' / 'older 60d' windows (default: END_DATE,
    today's behaviour, unchanged for every existing caller). Stage 2's
    simulated clock passes sim-now here so the windows track simulated time
    instead of the fixed historical reference date."""
    if login_events.empty:
        return pd.DataFrame(
            columns=["customer_id", "total_logins_90d", "recent_30d_vs_older_60d_ratio", "avg_session_duration_recent_30d"]
        )

    days_ago = (as_of - login_events["timestamp"]).dt.days
    recent = login_events[days_ago < 30]
    older = login_events[(days_ago >= 30) & (days_ago < 90)]

    total_logins = login_events.groupby("customer_id").size().rename("total_logins_90d")
    recent_count = recent.groupby("customer_id").size()
    older_count = older.groupby("customer_id").size()
    avg_session_recent = recent.groupby("customer_id")["session_duration_seconds"].mean()

    agg = pd.DataFrame(total_logins)
    agg["recent_count"] = recent_count
    agg["older_count"] = older_count
    agg["avg_session_duration_recent_30d"] = avg_session_recent
    agg = agg.fillna(0)

    ratio = agg["recent_count"] / agg["older_count"]
    agg["recent_30d_vs_older_60d_ratio"] = ratio.replace([np.inf, -np.inf], 0).fillna(0)

    agg = agg.drop(columns=["recent_count", "older_count"]).reset_index()
    return agg


def build_ticket_features(support_tickets):
    if support_tickets.empty:
        return pd.DataFrame(columns=["customer_id", "num_tickets", "pct_unresolved", "avg_resolution_time_hours"])

    grouped = support_tickets.groupby("customer_id")
    num_tickets = grouped.size().rename("num_tickets")
    pct_unresolved = grouped["resolved"].apply(lambda s: (~s).mean()).rename("pct_unresolved")
    # resolution_time_hours is NULL for unresolved tickets, so .mean() already
    # only averages over resolved ones (NaNs are skipped).
    avg_resolution_time_hours = grouped["resolution_time_hours"].mean().rename("avg_resolution_time_hours")

    agg = pd.concat([num_tickets, pct_unresolved, avg_resolution_time_hours], axis=1)
    agg["avg_resolution_time_hours"] = agg["avg_resolution_time_hours"].fillna(0)
    return agg.reset_index()


def build_usage_features(feature_usage_logs):
    if feature_usage_logs.empty:
        return pd.DataFrame(columns=["customer_id", "avg_usage_count"])

    return (
        feature_usage_logs.groupby("customer_id")["usage_count"]
        .mean()
        .rename("avg_usage_count")
        .reset_index()
    )


def build_feature_table(customers, login_events, support_tickets, feature_usage_logs, as_of=END_DATE):
    """Pure composition step (no I/O): joins the per-source feature builders
    and zero-fills customers with no rows in a given event table. as_of is
    forwarded to build_login_features only - ticket/usage features aren't
    windowed by recency at all (see build_ticket_features/build_usage_features)."""
    features = build_customer_features(customers)
    features = features.merge(build_login_features(login_events, as_of=as_of), on="customer_id", how="left")
    features = features.merge(build_ticket_features(support_tickets), on="customer_id", how="left")
    features = features.merge(build_usage_features(feature_usage_logs), on="customer_id", how="left")

    features[EVENT_DERIVED_COLUMNS] = features[EVENT_DERIVED_COLUMNS].fillna(0)
    features["total_logins_90d"] = features["total_logins_90d"].astype(int)
    features["num_tickets"] = features["num_tickets"].astype(int)

    return features[FEATURE_COLUMNS]


def write_features_atomically(conn, features):
    """Replaces customer_features with `features` in a single transaction,
    without ever making readers wait on or see an empty or partial table.

    Why a staging table and rename, not TRUNCATE + INSERT: TRUNCATE takes an
    ACCESS EXCLUSIVE lock, so every reader blocks until the INSERT commits.
    Measured on the old code: 3 reads out of 265 waited up to 648 ms. Here,
    all the slow work (the INSERT into customer_features_new) happens while
    readers keep using the old table. The only lock taken on the live name
    is for the two RENAMEs, which are quick. Readers never see zero rows:
    if anything fails, the transaction rolls back and the old table stays
    as it was.
    """
    rows = [tuple(row) for row in features.itertuples(index=False, name=None)]
    insert_sql = f"INSERT INTO {STAGING_TABLE} ({', '.join(FEATURE_COLUMNS)}) VALUES %s"
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL.format(table=FEATURES_TABLE))  # first run only; no-op after
        cur.execute(f"DROP TABLE IF EXISTS {STAGING_TABLE}")  # leftover from an interrupted run
        cur.execute(CREATE_TABLE_SQL.format(table=STAGING_TABLE))
        execute_values(cur, insert_sql, rows)
        cur.execute(f"ALTER TABLE {FEATURES_TABLE} RENAME TO customer_features_old")
        cur.execute(f"ALTER TABLE {STAGING_TABLE} RENAME TO {FEATURES_TABLE}")
        cur.execute("DROP TABLE customer_features_old")
        # Give the new table's primary-key index the same name the old one had.
        cur.execute(f"ALTER INDEX {STAGING_TABLE}_pkey RENAME TO {FEATURES_TABLE}_pkey")
    conn.commit()


def run_build(conn, as_of=END_DATE):
    """Reads customers + the three event tables, builds the feature table
    (windowed at as_of), writes it to FEATURES_CSV, and atomically replaces
    customer_features. Returns the built DataFrame. Shared by main() (the
    CLI entry point - as_of=END_DATE, today's behaviour, unchanged) and
    Stage 2's POST /pipeline/rebuild-features (as_of=simulated now), so
    there's one build+write code path regardless of caller."""
    customers = pd.read_sql("SELECT * FROM customers", conn)
    login_events = pd.read_sql("SELECT * FROM login_events", conn, parse_dates=["timestamp"])
    support_tickets = pd.read_sql("SELECT * FROM support_tickets", conn, parse_dates=["timestamp"])
    feature_usage_logs = pd.read_sql("SELECT * FROM feature_usage_logs", conn, parse_dates=["timestamp"])

    features = build_feature_table(customers, login_events, support_tickets, feature_usage_logs, as_of=as_of)

    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
    features.to_csv(FEATURES_CSV, index=False)

    write_features_atomically(conn, features)
    return features


def main():
    conn = psycopg2.connect(**PG_DSN)
    features = run_build(conn)
    conn.close()

    print(f"Final shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")


if __name__ == "__main__":
    main()
