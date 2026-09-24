"""
Exports the current best/latest XGBoost run's model artifact from the local
MLflow tracking store (sqlite:///mlruns.db, mlruns/) to a plain,
self-contained model directory: models/production_model/.

This decouples the deployed app from the tracking store entirely.
mlruns.db and mlruns/ are dev-time experiment history and stay local-only;
models/production_model/ is a deployment artifact, committed to git, and
is what src/serving/api.py loads at startup and what the Dockerfile COPYs
into the image - neither needs the tracking store present at runtime.

Uses the same run-lookup logic as api.py's load_latest_xgboost_pipeline():
the most recently started run named "xgboost" in the churn-baseline
experiment.

Run whenever a new xgboost run should become the deployed model:
    python scripts/export_model_for_deployment.py
"""

import shutil
import sys
from pathlib import Path

# scripts/ lives outside the src package, so add the project root to
# sys.path - running this file directly only puts scripts/ itself on the
# path, and `from src...` would otherwise fail.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402

from src.models.train_baseline import EXPERIMENT_NAME, MLFLOW_TRACKING_URI  # noqa: E402

OUTPUT_DIR = "models/production_model"


def find_latest_xgboost_run(client):
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
    return runs[0]


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    run = find_latest_xgboost_run(client)
    run_id = run.info.run_id
    print(f"Exporting xgboost run {run_id} -> {OUTPUT_DIR}")

    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        shutil.rmtree(output_path)  # save_model refuses to write into an existing dir

    # skops_trusted_types matches train_baseline.py's log_model call -
    # required for skops (mlflow's sklearn serializer) to load a pipeline
    # containing an XGBoost step.
    mlflow.sklearn.save_model(
        pipeline,
        path=str(output_path),
        skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
    )

    print(f"Saved model files to {output_path.resolve()}")


if __name__ == "__main__":
    main()
