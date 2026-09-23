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

CREATE_TABLE_SQL = """
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


def build_login_features(login_events):
    if login_events.empty:
        return pd.DataFrame(
            columns=["customer_id", "total_logins_90d", "recent_30d_vs_older_60d_ratio", "avg_session_duration_recent_30d"]
        )

    days_ago = (END_DATE - login_events["timestamp"]).dt.days
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


def build_feature_table(customers, login_events, support_tickets, feature_usage_logs):
    """Pure composition step (no I/O): joins the per-source feature builders
    and zero-fills customers with no rows in a given event table."""
    features = build_customer_features(customers)
    features = features.merge(build_login_features(login_events), on="customer_id", how="left")
    features = features.merge(build_ticket_features(support_tickets), on="customer_id", how="left")
    features = features.merge(build_usage_features(feature_usage_logs), on="customer_id", how="left")

    features[EVENT_DERIVED_COLUMNS] = features[EVENT_DERIVED_COLUMNS].fillna(0)
    features["total_logins_90d"] = features["total_logins_90d"].astype(int)
    features["num_tickets"] = features["num_tickets"].astype(int)

    return features[FEATURE_COLUMNS]


def main():
    conn = psycopg2.connect(**PG_DSN)

    customers = pd.read_sql("SELECT * FROM customers", conn)
    login_events = pd.read_sql("SELECT * FROM login_events", conn, parse_dates=["timestamp"])
    support_tickets = pd.read_sql("SELECT * FROM support_tickets", conn, parse_dates=["timestamp"])
    feature_usage_logs = pd.read_sql("SELECT * FROM feature_usage_logs", conn, parse_dates=["timestamp"])

    features = build_feature_table(customers, login_events, support_tickets, feature_usage_logs)

    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
    features.to_csv(FEATURES_CSV, index=False)

    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        cur.execute("TRUNCATE TABLE customer_features")
        rows = [tuple(row) for row in features.itertuples(index=False, name=None)]
        insert_sql = f"INSERT INTO customer_features ({', '.join(FEATURE_COLUMNS)}) VALUES %s"
        execute_values(cur, insert_sql, rows)
    conn.commit()
    conn.close()

    print(f"Final shape: {features.shape}")
    print(f"Columns: {list(features.columns)}")


if __name__ == "__main__":
    main()
