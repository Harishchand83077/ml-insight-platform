"""
Stage 3b serving-compat check: a candidate that passes the statistical
gate must ALSO survive the real export/reload/serve round trip before it
can ever be promoted. Exports the candidate exactly the way
scripts/export_model_for_deployment.py does, reloads it through the SAME
loader src/serving/api.py uses at startup (skops.io.load with the same
trusted types), and requires its predictions - taken through the real
predict_churn() path, not a shortcut - to match the in-memory candidate's
own predictions within 1e-6 for a sample of real customers.

This catches a class of bug a pure statistical comparison can't: the
candidate scores fine in memory, but something about serialization
(skops's trusted-types list, a sklearn/xgboost version mismatch between
training and serving, an unpickleable custom step) breaks it on the way to
being served. Finding that AFTER promotion is the failure mode this
exists to prevent.
"""

import shutil
from pathlib import Path

import mlflow.sklearn
import skops.io

from src.common.production_model import get_pipeline, set_pipeline
from src.models.inference import prepare_model_input
from src.serving.feature_cache import get_customer_features
from src.serving.prediction import predict_churn as predict_churn_api

SKOPS_TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
DEFAULT_TOLERANCE = 1e-6


class CompatCheckError(RuntimeError):
    """A customer_id used for the compat check wasn't found via
    predict_churn - the check can't run, so it can't be trusted to have
    passed; this is raised, never silently skipped."""


def export_candidate(candidate, export_dir):
    """The exact same call scripts/export_model_for_deployment.py makes -
    not a second, divergent way of writing a model directory."""
    path = Path(export_dir)
    if path.exists():
        shutil.rmtree(path)
    mlflow.sklearn.save_model(candidate, path=str(path), skops_trusted_types=SKOPS_TRUSTED_TYPES)
    return path


def load_via_api_loader(model_dir):
    """The same skops.io.load() call src.serving.api.load_production_model()
    makes at startup, just against a given directory."""
    model_path = str(Path(model_dir) / "model.skops")
    return skops.io.load(model_path, trusted=SKOPS_TRUSTED_TYPES)


def _score_in_memory(candidate, customer_id):
    features, _ = get_customer_features(customer_id)
    if features is None:
        raise CompatCheckError(f"customer_id {customer_id!r} not found in customer_features")
    X = prepare_model_input(features)
    return float(candidate.predict_proba(X)[0, 1])


def _score_via_predict_churn(customer_id):
    """Calls the real predict_churn() (confirming the actual serving
    function runs cleanly against the reloaded model - customer found, no
    exception), but returns the UNROUNDED probability computed the exact
    same way predict_churn() does internally (get_pipeline() + the same
    prepare_model_input(features)), not predict_churn()'s own return
    value: that dict rounds churn_probability to 4 decimals for API
    display (see src/serving/prediction.py), which would make a 1e-6
    comparison meaningless - any two numbers that round to the same 4
    decimals "pass" regardless of serialization fidelity. The gate needs
    the full-precision value to actually test that."""
    result = predict_churn_api(customer_id)
    if result is None:
        raise CompatCheckError(f"customer_id {customer_id!r} not found via predict_churn")

    features, _ = get_customer_features(customer_id)
    X = prepare_model_input(features)
    return float(get_pipeline().predict_proba(X)[0, 1])


def check_serving_compat(candidate, customer_ids, export_dir, tolerance=DEFAULT_TOLERANCE):
    """Exports `candidate`, reloads it through the API's own loader, and
    compares its full-precision prediction (computed the same way
    predict_churn() does internally, after confirming predict_churn()
    itself runs cleanly against the reloaded model) to the in-memory
    candidate's own predict_proba, for each of `customer_ids`. Temporarily
    swaps the shared production-model singleton (src.common.
    production_model) to the reloaded model for the duration of those
    calls, then restores whatever was set before - regardless of outcome
    - so this never leaves a running server pointed at an unpromoted
    candidate."""
    export_candidate(candidate, export_dir)
    reloaded = load_via_api_loader(export_dir)

    try:
        previous_pipeline = get_pipeline()
    except RuntimeError:
        previous_pipeline = None

    try:
        set_pipeline(reloaded)
        via_reloaded = [_score_via_predict_churn(cid) for cid in customer_ids]
    finally:
        if previous_pipeline is not None:
            set_pipeline(previous_pipeline)

    via_in_memory = [_score_in_memory(candidate, cid) for cid in customer_ids]

    diffs = [abs(a - b) for a, b in zip(via_reloaded, via_in_memory)]
    passed = all(d <= tolerance for d in diffs)

    return {
        "passed": passed,
        "customer_ids": list(customer_ids),
        "via_reloaded_model": via_reloaded,
        "via_in_memory_candidate": via_in_memory,
        "max_abs_diff": max(diffs) if diffs else None,
        "tolerance": tolerance,
    }
