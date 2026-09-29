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
