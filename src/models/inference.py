"""
The one piece of train_baseline.py that the serving path (src/serving/api.py)
actually needs at request time: turning a customer_features row into the
DataFrame shape the trained pipeline expects. Split out into its own
module - deliberately with no mlflow/matplotlib/psycopg2 imports - so
importing it doesn't drag training-only dependencies into the deployed
service's baseline memory footprint. train_baseline.py imports the column
lists from here too, so there's still one source of truth for the
pipeline's expected feature columns.
"""

import pandas as pd

CATEGORICAL_COLS = ["contract", "payment_method", "internet_service"]
BINARY_YES_NO_COLS = ["partner", "dependents"]
NUMERIC_COLS = [
    "tenure",
    "monthly_charges",
    "senior_citizen",
    "num_addons_active",
    "total_logins_90d",
    "recent_30d_vs_older_60d_ratio",
    "avg_session_duration_recent_30d",
    "num_tickets",
    "pct_unresolved",
    "avg_resolution_time_hours",
    "avg_usage_count",
]


def prepare_model_input(features):
    """Turn one customer_features row (as returned by get_customer_features,
    with partner/dependents still "Yes"/"No") into the single-row DataFrame
    shape the training pipeline was fit on, for serving-time inference."""
    row = dict(features)
    row["partner"] = 1 if row["partner"] == "Yes" else 0
    row["dependents"] = 1 if row["dependents"] == "Yes" else 0
    columns = CATEGORICAL_COLS + BINARY_YES_NO_COLS + NUMERIC_COLS
    return pd.DataFrame([{col: row[col] for col in columns}])
