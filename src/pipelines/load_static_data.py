"""
One-off batch load of the static Telco customer table
(data/raw/telco_churn.csv.csv) into Postgres as `customers`. This data
doesn't change per-event, so it's loaded directly rather than streamed
through RabbitMQ like the synthetic event tables.

TotalCharges has 11 blank values in the source CSV, all for customers with
tenure == 0 (confirmed during EDA - they are brand-new customers who
haven't been billed yet), so those are loaded as 0 instead of NULL.
"""

import os

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

RAW_CSV = "data/raw/telco_churn.csv.csv"

PG_DSN = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": os.environ.get("PG_PORT", "5432"),
    "dbname": os.environ.get("PG_DATABASE", "ml_insight"),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", "devpassword"),
}

CREATE_TABLE_SQL = """
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
"""

# raw CSV column -> customers table column
COLUMN_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}


def clean_total_charges(df):
    """Blank/whitespace-only TotalCharges values (all tenure=0 customers,
    confirmed during EDA) become 0 rather than NULL, dropped, or
    mean-imputed - they're legitimately zero-charge, not missing data."""
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip().replace("", "0")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    return df


def load_dataframe():
    df = pd.read_csv(RAW_CSV)

    blank_total_charges = df["TotalCharges"].astype(str).str.strip().eq("").sum()
    print(f"Blank TotalCharges values (set to 0): {blank_total_charges}")

    df = clean_total_charges(df)

    return df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]


def main():
    df = load_dataframe()

    conn = psycopg2.connect(**PG_DSN)
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        cur.execute("TRUNCATE TABLE customers")

        columns = list(df.columns)
        rows = [tuple(row) for row in df[columns].itertuples(index=False, name=None)]
        insert_sql = f"INSERT INTO customers ({', '.join(columns)}) VALUES %s"
        execute_values(cur, insert_sql, rows)

    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM customers")
        row_count = cur.fetchone()[0]

    conn.close()
    print(f"Loaded {row_count} rows into customers (source CSV had {len(df)} rows)")


if __name__ == "__main__":
    main()
