"""
Churn-prediction API: GET /health and POST /predict. The XGBoost model
(the full preprocessing + classifier pipeline, from its MLflow run
artifact) is loaded once at startup, not per-request.
"""

import logging
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.train_baseline import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, prepare_model_input
from src.serving.feature_cache import get_customer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api")

model_state = {}


def load_latest_xgboost_pipeline():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT_NAME}' not found")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'xgboost'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No 'xgboost' run found in MLflow experiment '{EXPERIMENT_NAME}'")

    run_id = runs[0].info.run_id
    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    logger.info("Loaded xgboost model from run %s", run_id)
    return pipeline, run_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["pipeline"], model_state["run_id"] = load_latest_xgboost_pipeline()
    yield
    model_state.clear()


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    customer_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    features, cache_hit = get_customer_features(req.customer_id)
    if features is None:
        raise HTTPException(status_code=404, detail=f"customer_id '{req.customer_id}' not found")

    X = prepare_model_input(features)
    churn_probability = float(model_state["pipeline"].predict_proba(X)[0, 1])
    prediction = "Yes" if churn_probability >= 0.5 else "No"

    return {
        "customer_id": req.customer_id,
        "churn_probability": round(churn_probability, 4),
        "prediction": prediction,
        "cache_hit": cache_hit,
    }
