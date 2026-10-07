"""
Unit tests for src/live/drift.py's get_drift_report() response shape, and
that it refuses to run before any reference has been cached (i.e. before
POST /pipeline/reset has run once in this process). Evidently's report and
the database read are both faked - what's under test is the shape this
project's code produces from Evidently's output, not Evidently itself.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.live import drift


def _fake_snapshot_dict(column_results):
    """column_results: list of (column, method, threshold, value)."""
    metrics = []
    drifted_count = 0
    for column, method, threshold, value in column_results:
        metrics.append(
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": column,
                    "method": method,
                    "threshold": threshold,
                },
                "value": value,
            }
        )
        if drift.is_drifted(method, value, threshold):
            drifted_count += 1
    share = drifted_count / len(column_results) if column_results else 0.0
    metrics.append(
        {
            "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
            "value": {"count": drifted_count, "share": share},
        }
    )
    return {"metrics": metrics}


@pytest.fixture(autouse=True)
def _clear_reference():
    drift._reference = None
    yield
    drift._reference = None


class TestNoReferenceCachedYet:
    def test_raises_before_pipeline_reset_has_ever_run(self):
        with pytest.raises(drift.NoReferenceError):
            drift.get_drift_report()


class TestResponseShape:
    def test_shape_once_a_reference_is_cached(self):
        drift._reference = pd.DataFrame({"tenure": [1, 2, 3]})
        fake_snapshot = MagicMock()
        fake_snapshot.dict.return_value = _fake_snapshot_dict(
            [
                ("tenure", "wasserstein", 0.1, 0.05),  # distance < threshold: not drifted
                ("monthly_charges", "wasserstein", 0.1, 0.30),  # distance > threshold: drifted
                ("avg_usage_count", "wasserstein", 0.1, 0.02),  # distance < threshold: not drifted
            ]
        )

        with patch.object(drift, "_load_current", return_value=pd.DataFrame({"tenure": [1]})), \
             patch.object(drift, "psycopg2") as fake_psycopg2, \
             patch.object(drift, "Report") as fake_report_cls:
            fake_psycopg2.connect.return_value = MagicMock()
            fake_report_cls.return_value.run.return_value = fake_snapshot

            result = drift.get_drift_report()

        assert set(result) == {"columns", "drifted_columns", "total_columns", "share", "dataset_drift"}
        assert result["drifted_columns"] == 1
        assert result["share"] == pytest.approx(1 / 3)
        assert result["dataset_drift"] is False  # share < 0.5
        assert len(result["columns"]) == 3
        for col in result["columns"]:
            assert set(col) == {"column", "method", "distance", "threshold", "drifted"}

        drifted_names = {c["column"] for c in result["columns"] if c["drifted"]}
        assert drifted_names == {"monthly_charges"}

    def test_dataset_drift_true_when_share_at_or_above_half(self):
        drift._reference = pd.DataFrame({"tenure": [1, 2, 3]})
        fake_snapshot = MagicMock()
        fake_snapshot.dict.return_value = _fake_snapshot_dict(
            [
                ("tenure", "wasserstein", 0.1, 0.30),  # drifted
                ("monthly_charges", "wasserstein", 0.1, 0.05),  # not drifted
            ]
        )

        with patch.object(drift, "_load_current", return_value=pd.DataFrame({"tenure": [1]})), \
             patch.object(drift, "psycopg2") as fake_psycopg2, \
             patch.object(drift, "Report") as fake_report_cls:
            fake_psycopg2.connect.return_value = MagicMock()
            fake_report_cls.return_value.run.return_value = fake_snapshot

            result = drift.get_drift_report()

        assert result["share"] == 0.5
        assert result["dataset_drift"] is True  # share >= 0.5
