"""
Trains two baseline churn classifiers - Logistic Regression and XGBoost -
on customer_features (joined with the Churn label from customers), with
default/reasonable hyperparameters and no tuning. Logs params, metrics,
the model artifact, and a feature-importance plot for each to a local
MLflow tracking store (sqlite:///mlruns.db, view with:
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000).

Churn is ~26%/74% imbalanced (confirmed during EDA), so PR-AUC and F1 are
treated as the primary metrics - accuracy alone would be misleading here
(a model that always predicts "No churn" scores ~74% accuracy for free).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import psycopg2
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PG_DSN = {
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": os.environ.get("PG_PORT", "5432"),
    "dbname": os.environ.get("PG_DATABASE", "ml_insight"),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", "devpassword"),
}

MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "churn-baseline"
PLOT_DIR = "data/processed/plots"

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

# Positive/negative bars need distinct hues (polarity); magnitude-only bars
# use a single hue - never a rainbow across categories.
COLOR_POSITIVE = "#c0392b"  # pushes toward churn
COLOR_NEGATIVE = "#2e6f9e"  # pushes toward retention
COLOR_MAGNITUDE = "#2e6f9e"


def load_dataset():
    conn = psycopg2.connect(**PG_DSN)
    df = pd.read_sql(
        """
        SELECT cf.*, c.churn
        FROM customer_features cf
        JOIN customers c ON cf.customer_id = c.customer_id
        """,
        conn,
    )
    conn.close()

    df["partner"] = (df["partner"] == "Yes").astype(int)
    df["dependents"] = (df["dependents"] == "Yes").astype(int)
    y = (df["churn"] == "Yes").astype(int)
    X = df[CATEGORICAL_COLS + BINARY_YES_NO_COLS + NUMERIC_COLS]
    return X, y


def prepare_model_input(features):
    """Turn one customer_features row (as returned by get_customer_features,
    with partner/dependents still "Yes"/"No") into the single-row DataFrame
    shape the training pipeline was fit on, for serving-time inference."""
    row = dict(features)
    row["partner"] = 1 if row["partner"] == "Yes" else 0
    row["dependents"] = 1 if row["dependents"] == "Yes" else 0
    columns = CATEGORICAL_COLS + BINARY_YES_NO_COLS + NUMERIC_COLS
    return pd.DataFrame([{col: row[col] for col in columns}])


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder="passthrough",
    )


def compute_metrics(y_test, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }


def plot_importance(names, values, title, out_path, diverging):
    order = np.argsort(np.abs(values))[::-1][:20]  # top 20 by magnitude
    names = np.array(names)[order][::-1]
    values = np.array(values)[order][::-1]
    colors = [COLOR_POSITIVE if v > 0 else COLOR_NEGATIVE for v in values] if diverging else COLOR_MAGNITUDE

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, values, color=colors, height=0.6)
    ax.set_title(title)
    ax.set_xlabel("Coefficient" if diverging else "Importance")
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_and_log_logistic_regression(X_train, X_test, y_train, y_test, feature_names_fn):
    preprocessor = build_preprocessor()
    model = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = Pipeline(
        [("preprocess", preprocessor), ("scale", StandardScaler()), ("model", model)]
    )

    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_params({f"model__{k}": v for k, v in model.get_params().items()})
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        coefficients = pipeline.named_steps["model"].coef_[0]
        plot_path = f"{PLOT_DIR}/logistic_regression_importance.png"
        plot_importance(
            feature_names, coefficients, "Logistic Regression - coefficients (top 20 by |value|)", plot_path, diverging=True
        )
        mlflow.log_artifact(plot_path)

    return metrics


def train_and_log_xgboost(X_train, X_test, y_train, y_test):
    preprocessor = build_preprocessor()
    model = XGBClassifier(random_state=42, eval_metric="logloss")
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params({f"model__{k}": v for k, v in model.get_params().items() if v is not None})
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        # log the full pipeline (preprocessing + model), not just the bare
        # classifier, so a serving caller can pass it raw feature rows.
        # skops (mlflow's sklearn serializer) refuses to load XGBoost's
        # types unless explicitly told they're trusted.
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
        )

        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        plot_path = f"{PLOT_DIR}/xgboost_importance.png"
        plot_importance(
            feature_names, importances, "XGBoost - feature importances (top 20)", plot_path, diverging=False
        )
        mlflow.log_artifact(plot_path)

    return metrics


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    lr_metrics = train_and_log_logistic_regression(X_train, X_test, y_train, y_test, None)
    xgb_metrics = train_and_log_xgboost(X_train, X_test, y_train, y_test)

    print(f"\nTrain rows: {len(X_train)}  Test rows: {len(X_test)}  Churn rate (test): {y_test.mean():.3f}\n")
    print(f"{'metric':<12}{'logistic_regression':>22}{'xgboost':>14}")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"{key:<12}{lr_metrics[key]:>22.4f}{xgb_metrics[key]:>14.4f}")


if __name__ == "__main__":
    main()
