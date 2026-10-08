"""
Stage 3b promotion gate: freeze a holdout, train a retrain candidate with
production's own hyperparameters, evaluate candidate vs production on
that frozen holdout (paired bootstrap on the PR-AUC difference), and
decide whether to promote - using the margin Stage 3a's measurements
grounded (0.02), not a guess.

Local only: freeze_holdout() reads through src.live.safety.local_pg_dsn(),
so this refuses to run against anything but a local Postgres, same as
every other Stage 1/2/3 component.

Margin rationale (Stage 3a): across 6 measured scenarios, the PAIRED
bootstrap SE of the PR-AUC difference ranged 0.013-0.020. 0.02 sits just
above that whole observed range, so a candidate can't clear the margin on
noise alone at this holdout size - the interval-excludes-zero check
enforces statistical significance independently.
"""

import datetime

import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split

from src.live.safety import local_pg_dsn
from src.models.inference import BINARY_YES_NO_COLS, CATEGORICAL_COLS, NUMERIC_COLS
from src.models.train_baseline import build_xgboost_pipeline

FEATURE_COLUMNS = CATEGORICAL_COLS + BINARY_YES_NO_COLS + NUMERIC_COLS

DEFAULT_LABEL_TABLE = "customers"
CONCEPT_DRIFT_LABEL_TABLE = "stage3a_concept_drift_labels"

PROMOTION_MARGIN = 0.02
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


class HoldoutMismatchError(AssertionError):
    """holdout["X_test"]'s customer_id column and holdout["y_test"]'s
    index don't agree - the holdout itself was built or edited
    inconsistently, so scoring against it would silently misalign rows."""


def _load_labeled_table(label_table, strength=None):
    conn = psycopg2.connect(**local_pg_dsn())
    try:
        if label_table == DEFAULT_LABEL_TABLE:
            df = pd.read_sql(
                """
                SELECT cf.*, c.churn AS label
                FROM customer_features cf
                JOIN customers c ON cf.customer_id = c.customer_id
                """,
                conn,
            )
            df["label"] = (df["label"] == "Yes").astype(int)
        elif label_table == CONCEPT_DRIFT_LABEL_TABLE:
            if strength is None:
                raise ValueError(f"strength is required when label_table={label_table!r}")
            df = pd.read_sql(
                """
                SELECT cf.*, s.scenario_churn AS label
                FROM customer_features cf
                JOIN stage3a_concept_drift_labels s ON cf.customer_id = s.customer_id
                WHERE s.strength = %(strength)s
                """,
                conn,
                params={"strength": strength},
            )
            df["label"] = df["label"].astype(int)
        else:
            raise ValueError(f"unknown label_table {label_table!r}")
    finally:
        conn.close()

    df["partner"] = (df["partner"] == "Yes").astype(int)
    df["dependents"] = (df["dependents"] == "Yes").astype(int)
    return df


def freeze_holdout(label_table=DEFAULT_LABEL_TABLE, strength=None, test_size=0.2, random_state=42):
    """Snapshots customer_features joined to labels (default:
    customers.churn; pass label_table=CONCEPT_DRIFT_LABEL_TABLE and a
    strength to use Stage 3a's local-only relabeled table instead), splits
    80/20 stratified, and returns the frozen holdout: train/test feature
    frames (indexed by customer_id) and the metadata every evaluate()/
    decide() call reports back (rows, positives, frozen_at, label_table)."""
    df = _load_labeled_table(label_table, strength=strength)

    X = df[["customer_id"] + FEATURE_COLUMNS].set_index("customer_id")
    y = pd.Series(df["label"].values, index=X.index, name="label")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "metadata": {
            "rows": len(X_test),
            "positives": int(y_test.sum()),
            "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "label_table": label_table,
            "strength": strength,
            "train_rows": len(X_train),
            "random_state": random_state,
        },
    }


def train_candidate(holdout):
    """Fits a fresh pipeline - build_xgboost_pipeline(), imported from
    train_baseline.py, not copied - on the holdout's train split."""
    pipeline = build_xgboost_pipeline()
    pipeline.fit(holdout["X_train"][FEATURE_COLUMNS], holdout["y_train"])
    return pipeline


def _predict_proba(model, X_test):
    return model.predict_proba(X_test[FEATURE_COLUMNS])[:, 1]


def paired_bootstrap_pr_auc_diff(y_test, proba_production, proba_candidate, n_resamples=BOOTSTRAP_N_RESAMPLES, seed=BOOTSTRAP_SEED):
    """candidate - production, resampled WITH replacement, the SAME
    resampled indices for both models each draw (paired)."""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y_test)
    proba_production = np.asarray(proba_production)
    proba_candidate = np.asarray(proba_candidate)
    n = len(y_arr)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_s = y_arr[idx]
        if y_s.sum() == 0 or y_s.sum() == n:
            continue
        pr_prod = average_precision_score(y_s, proba_production[idx])
        pr_cand = average_precision_score(y_s, proba_candidate[idx])
        diffs.append(pr_cand - pr_prod)
    diffs = np.array(diffs)
    return {
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "se": float(diffs.std(ddof=1)),
        "n_resamples_used": int(len(diffs)),
    }


def evaluate(candidate, production, holdout):
    """Scores both models on holdout's SAME test rows. Before scoring,
    asserts X_test's customer_id column and y_test's index agree exactly -
    a holdout built or hand-edited so the two disagree (the thing that
    would silently misalign a prediction with the wrong label) raises
    HoldoutMismatchError instead of producing a number that looks fine."""
    X_test, y_test = holdout["X_test"], holdout["y_test"]
    if list(X_test.index) != list(y_test.index):
        raise HoldoutMismatchError(
            "holdout X_test and y_test disagree on customer_id order/membership - "
            f"X_test has {len(X_test)} rows, y_test has {len(y_test)}, "
            f"first mismatch at {next((i for i, (a, b) in enumerate(zip(X_test.index, y_test.index)) if a != b), 'length')}"
        )

    prod_proba = _predict_proba(production, X_test)
    cand_proba = _predict_proba(candidate, X_test)

    prod_pred = (prod_proba >= 0.5).astype(int)
    cand_pred = (cand_proba >= 0.5).astype(int)

    production_metrics = {
        "pr_auc": float(average_precision_score(y_test, prod_proba)),
        "f1": float(f1_score(y_test, prod_pred)),
    }
    candidate_metrics = {
        "pr_auc": float(average_precision_score(y_test, cand_proba)),
        "f1": float(f1_score(y_test, cand_pred)),
    }
    bootstrap = paired_bootstrap_pr_auc_diff(y_test, prod_proba, cand_proba)

    return {
        "production": production_metrics,
        "candidate": candidate_metrics,
        "pr_auc_diff_point": candidate_metrics["pr_auc"] - production_metrics["pr_auc"],
        "f1_diff_point": candidate_metrics["f1"] - production_metrics["f1"],
        "bootstrap": bootstrap,
        "holdout_metadata": holdout["metadata"],
    }


def decide(pr_auc_diff_point, ci_lower, ci_upper, holdout_metadata=None, margin=PROMOTION_MARGIN):
    """Promote only if the interval excludes zero on the positive side
    (ci_lower > 0) AND the point difference clears the margin. Both
    conditions are independent and both must hold - a huge point diff with
    an interval that still touches zero is exactly as unpromotable as a
    tiny, "significant" diff that never gets near the margin."""
    interval_excludes_zero_positive = ci_lower > 0
    meets_margin = pr_auc_diff_point >= margin

    margin_shortfall = max(0.0, margin - pr_auc_diff_point)
    interval_shortfall = max(0.0, -ci_lower)  # how far below zero the lower bound sits, 0 if it's already > 0

    promoted = interval_excludes_zero_positive and meets_margin

    reasons = []
    if not interval_excludes_zero_positive:
        reasons.append(
            f"95% interval [{ci_lower:.4f}, {ci_upper:.4f}] does not exclude zero on the positive side"
        )
    if not meets_margin:
        reasons.append(f"point difference {pr_auc_diff_point:.4f} is below the required margin {margin:.4f}")
    if promoted:
        reasons.append(
            f"point difference {pr_auc_diff_point:.4f} >= margin {margin:.4f} and "
            f"95% interval [{ci_lower:.4f}, {ci_upper:.4f}] excludes zero"
        )

    return {
        "promoted": promoted,
        "pr_auc_diff_point": pr_auc_diff_point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "margin": margin,
        "margin_shortfall": margin_shortfall,
        "interval_shortfall": interval_shortfall,
        "holdout_metadata": holdout_metadata or {},
        "reasons": reasons,
    }
