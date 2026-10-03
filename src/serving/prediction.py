"""
The core "predict this customer's churn risk" logic: look up their
features (feature_cache's cache-aside lookup), transform to the shape
the trained pipeline expects, and run inference. Shared by POST /predict
(api.py) and predict_churn_tool (src/agent/tools.py) - the agent tool
calls this directly, in-process, instead of making an HTTP request back
to its own server. An agent tool calling its own server over HTTP was
architecturally wrong regardless of auth (it needed a whole separate
internal-credential story just to get past JWT auth meant for end
users), and it would have meant a second, fully independent request
going through get_current_user, log_audit, etc. for what is really just
a function call the process can make on itself.

Imports src.common.production_model (not src.serving.api directly) for
the loaded pipeline - see that module's docstring for why: api.py
imports src.agent.agent, which imports src.agent.tools, which needs this
function, so importing straight from api.py here would be circular.
"""

from src.common.production_model import get_pipeline
from src.models.inference import prepare_model_input
from src.serving.feature_cache import get_customer_features


def predict_churn(customer_id: str) -> dict | None:
    """Returns None if customer_id isn't found in customer_features, else
    {customer_id, churn_probability, prediction, cache_hit}."""
    features, cache_hit = get_customer_features(customer_id)
    if features is None:
        return None

    X = prepare_model_input(features)
    churn_probability = float(get_pipeline().predict_proba(X)[0, 1])
    prediction = "Yes" if churn_probability >= 0.5 else "No"

    return {
        "customer_id": customer_id,
        "churn_probability": round(churn_probability, 4),
        "prediction": prediction,
        "cache_hit": cache_hit,
    }


# Account-level fields a what-if can change. Event-derived behavioral features
# (logins, tickets, usage) aren't here: they describe what a customer did, not
# something a plan change can set. Numeric bounds are the observed range in
# customer_features, so a what-if doesn't extrapolate past the training data.
_CATEGORICAL_OVERRIDES = {
    "contract": {"Month-to-month", "One year", "Two year"},
    "payment_method": {"Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"},
    "internet_service": {"DSL", "Fiber optic", "No"},
    "partner": {"Yes", "No"},
    "dependents": {"Yes", "No"},
}
_NUMERIC_OVERRIDES = {
    "senior_citizen": (0, 1, int),
    "tenure": (0, 72, int),
    "monthly_charges": (18.25, 118.75, float),
    "num_addons_active": (0, 6, int),
}


def _validate_overrides(overrides):
    if not overrides:
        raise ValueError("No overrides given - pass at least one field to change.")
    clean = {}
    for field, value in overrides.items():
        if field in _CATEGORICAL_OVERRIDES:
            allowed = _CATEGORICAL_OVERRIDES[field]
            if value not in allowed:
                raise ValueError(f"'{value}' is not a valid {field}. Allowed: {sorted(allowed)}.")
            clean[field] = value
        elif field in _NUMERIC_OVERRIDES:
            low, high, kind = _NUMERIC_OVERRIDES[field]
            if isinstance(value, bool):
                raise ValueError(f"{field} must be a number, got {value!r}.")
            try:
                number = kind(float(value))
            except (TypeError, ValueError):
                raise ValueError(f"{field} must be a number, got {value!r}.") from None
            if kind is int and float(value) != number:
                raise ValueError(f"{field} must be a whole number, got {value!r}.")
            if not (low <= number <= high):
                raise ValueError(f"{field}={number} is outside the valid range {low} to {high}.")
            clean[field] = number
        else:
            allowed_fields = sorted(list(_CATEGORICAL_OVERRIDES) + list(_NUMERIC_OVERRIDES))
            raise ValueError(f"'{field}' cannot be changed in a what-if. Changeable fields: {allowed_fields}.")
    return clean


def _score(features):
    return float(get_pipeline().predict_proba(prepare_model_input(features))[0, 1])


def simulate_prediction(customer_id: str, overrides: dict) -> dict | None:
    """Re-scores a customer with some account fields changed, for what-if
    questions. Raises ValueError for an invalid override (before any database
    lookup). Returns None if customer_id isn't found, else the original and
    modified churn probabilities and their difference.

    This is the model's counterfactual, not a causal forecast: it changes the
    input features and reports what the trained model would output."""
    clean = _validate_overrides(overrides)

    features, _ = get_customer_features(customer_id)
    if features is None:
        return None

    original = _score(features)
    modified = _score({**features, **clean})
    delta = modified - original

    return {
        "customer_id": customer_id,
        "overrides": clean,
        "original_probability": round(original, 4),
        "modified_probability": round(modified, 4),
        "delta": round(delta, 4),
        "direction": "decreased" if delta < 0 else "increased" if delta > 0 else "unchanged",
    }
