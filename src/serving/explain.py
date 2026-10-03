"""
Per-customer feature contributions for the churn model: which features
moved this customer's predicted risk up or down, and by how much.

Contributions come from XGBoost's own booster.predict(..., pred_contribs=True),
which computes exact TreeSHAP values in C++ inside xgboost - the same numbers
shap.TreeExplainer returns for XGBoost models (checked to 0.0 difference per
feature), without importing the shap package or its numba/llvmlite
dependencies. Values are in log-odds (the model's margin), not probability
points. The last column is the bias term; bias + sum(contributions) equals the
raw margin for that customer, so a probability is recoverable exactly by a
sigmoid.

One-hot encoded categorical columns (contract, payment_method,
internet_service) are summed back into their source column before ranking, so
"contract=Month-to-month" is one contribution, not three one-hot columns.
"""

import numpy as np
import xgboost as xgb

from src.common.production_model import get_pipeline
from src.models.inference import CATEGORICAL_COLS, prepare_model_input
from src.serving.feature_cache import get_customer_features

TOP_K = 5


def _source_column(transformed_name):
    """Map a transformed feature name back to the raw input column it came
    from. Categorical one-hots are named 'cat__<col>_<value>', passthrough
    columns 'remainder__<col>'."""
    bare = transformed_name.split("__", 1)[1]
    for col in CATEGORICAL_COLS:
        if bare.startswith(col + "_"):
            return col
    return bare


def explain_prediction(customer_id):
    """Returns None if customer_id isn't found. Otherwise:
    {customer_id, churn_probability, base_log_odds, total_log_odds,
     top_features: [{feature, value, log_odds, direction}, ...]} where
    top_features is the TOP_K source columns with the largest |log_odds|,
    positive log_odds meaning the feature pushed churn risk up."""
    features, _ = get_customer_features(customer_id)
    if features is None:
        return None

    pipeline = get_pipeline()
    preprocessor = pipeline.named_steps["preprocess"]
    X = prepare_model_input(features)
    X_transformed = preprocessor.transform(X)
    transformed_names = preprocessor.get_feature_names_out()

    booster = pipeline.named_steps["model"].get_booster()
    contributions = booster.predict(xgb.DMatrix(X_transformed), pred_contribs=True)[0]
    feature_contributions, base_log_odds = contributions[:-1], float(contributions[-1])

    grouped = {}
    for name, value in zip(transformed_names, feature_contributions):
        col = _source_column(name)
        grouped[col] = grouped.get(col, 0.0) + float(value)

    total_log_odds = base_log_odds + float(feature_contributions.sum())
    churn_probability = 1.0 / (1.0 + np.exp(-total_log_odds))

    ranked = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_K]
    top_features = [
        {
            "feature": col,
            "value": features[col],
            "log_odds": round(contribution, 4),
            "direction": "toward churn" if contribution > 0 else "away from churn",
        }
        for col, contribution in ranked
    ]

    return {
        "customer_id": customer_id,
        "churn_probability": round(float(churn_probability), 4),
        "base_log_odds": round(base_log_odds, 4),
        "total_log_odds": round(total_log_odds, 4),
        "top_features": top_features,
    }
