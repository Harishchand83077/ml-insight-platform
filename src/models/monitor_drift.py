"""
Data drift check for the churn feature set, using Evidently. Reference is
the exact X_train used in train_baseline.py (same split, same
random_state=42). "Current" simulates a later snapshot: a random 20%
sample of customer_features with mild synthetic drift injected into three
numeric columns - avg_usage_count down 15%, avg_session_duration_recent_30d
down 10%, avg_resolution_time_hours up 20% - modeling declining engagement
and slower support over time. Saves the full report to
reports/drift_report.html and prints a per-column drift summary.

Evidently 0.7's DataDriftPreset auto-picks a statistical test per column:
a p-value test (drift when p < threshold) for small samples, or a distance
metric like Wasserstein/Jensen-Shannon (drift when score > threshold) for
larger ones - both conventions are handled below.
"""

import os

import pandas as pd
import psycopg2
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split

from train_baseline import BINARY_YES_NO_COLS, CATEGORICAL_COLS, NUMERIC_COLS, PG_DSN, load_dataset

REPORT_PATH = "reports/drift_report.html"
CURRENT_SAMPLE_FRAC = 0.2
CURRENT_SAMPLE_SEED = 123

DRIFT_SHIFTS = {
    "avg_usage_count": 0.85,  # -15%, declining engagement
    "avg_session_duration_recent_30d": 0.90,  # -10%, shorter sessions
    "avg_resolution_time_hours": 1.20,  # +20%, slower support
}

FEATURE_COLUMNS = CATEGORICAL_COLS + BINARY_YES_NO_COLS + NUMERIC_COLS


def build_current_dataset():
    conn = psycopg2.connect(**PG_DSN)
    df = pd.read_sql("SELECT * FROM customer_features", conn)
    conn.close()

    df["partner"] = (df["partner"] == "Yes").astype(int)
    df["dependents"] = (df["dependents"] == "Yes").astype(int)

    current = df.sample(frac=CURRENT_SAMPLE_FRAC, random_state=CURRENT_SAMPLE_SEED)[FEATURE_COLUMNS].copy()
    for col, factor in DRIFT_SHIFTS.items():
        current[col] = current[col] * factor

    return current


def is_drifted(method, value, threshold):
    return value < threshold if "p_value" in method.lower() else value > threshold


def main():
    X, y = load_dataset()
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    current = build_current_dataset()

    report = Report([DataDriftPreset()])
    snapshot = report.run(current, X_train)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    snapshot.save_html(REPORT_PATH)

    result = snapshot.dict()

    print(
        f"Reference rows: {len(X_train)}  Current rows: {len(current)} "
        f"(20% sample, drift injected into {list(DRIFT_SHIFTS)})\n"
    )
    print(f"{'column':<35}{'method':<28}{'score':>10}{'threshold':>12}  drifted")

    drifted_cols = []
    for m in result["metrics"]:
        cfg = m.get("config", {})
        if cfg.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        column, method, threshold, value = cfg["column"], cfg["method"], cfg["threshold"], m["value"]
        drifted = is_drifted(method, value, threshold)
        if drifted:
            drifted_cols.append(column)
        print(f"{column:<35}{method:<28}{value:>10.4f}{threshold:>12.3f}  {drifted}")

    for m in result["metrics"]:
        if m.get("config", {}).get("type") == "evidently:metric_v2:DriftedColumnsCount":
            count, share = m["value"]["count"], m["value"]["share"]
            dataset_drift = "DETECTED" if share >= 0.5 else "not detected"
            print(
                f"\nDrifted columns: {int(count)}/{len(FEATURE_COLUMNS)} ({share:.1%}) "
                f"- dataset drift {dataset_drift} (threshold 50%)"
            )

    print(f"Columns with drift: {drifted_cols if drifted_cols else 'none'}")
    print(f"Full report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
