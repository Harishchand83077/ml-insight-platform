"""
Holds the one shared trained XGBoost pipeline instance - same rationale
as src/common/embedding_model.py: a plain leaf module (no dependency on
src.serving.api or src.agent.agent) so both POST /predict (api.py) and
predict_churn_tool (src/agent/tools.py, via src/serving/prediction.py)
can share the SAME loaded pipeline object. Without this, tools.py
importing the pipeline from api.py directly would be circular
(api.py -> agent.py -> tools.py -> api.py), and loading a second copy of
the model just to avoid that would double its memory cost for no reason.

src/serving/api.py's lifespan handler calls set_pipeline() exactly once,
at startup, right alongside model_state["pipeline"] = ... and
set_embedder(...) - before the app accepts any requests.
"""

_pipeline = None


def set_pipeline(pipeline):
    global _pipeline
    _pipeline = pipeline


def get_pipeline():
    if _pipeline is None:
        raise RuntimeError(
            "Production model not loaded yet - src.serving.api's lifespan "
            "handler must call set_pipeline() before any request runs."
        )
    return _pipeline
