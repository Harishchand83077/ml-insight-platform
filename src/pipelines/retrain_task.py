"""
Celery app for async model retraining, using Redis as both broker and
result backend. Start a worker with:
celery -A src.pipelines.retrain_task worker --loglevel=info
"""

from celery import Celery

app = Celery("retrain_task", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")


@app.task(name="retrain_model")
def retrain_model():
    # imported lazily so the worker doesn't pull in matplotlib/sklearn/
    # xgboost/mlflow just to register the task at startup
    from src.models.train_baseline import main as train_main

    train_main()
    return "retrain_model finished"
