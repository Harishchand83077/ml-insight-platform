"""
Unit tests for src/live/gate.py, using the Stage 3a measured numbers as
fixtures: Scenario A's 3 strengths and Scenario B's 3 strengths, each as
(point diff, [ci_lower, ci_upper]) pairs, checked against decide()'s
actual promote/reject outcome - this is a regression test against the
real measurements, not synthetic data chosen to make the gate look good.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.live import gate

# --- Stage 3a's measured fixtures --------------------------------------

SCENARIO_A = [
    # (strength, point_diff, ci_lower, ci_upper)
    (1.0, 0.0006, -0.0249, 0.0237),
    (0.6, -0.0069, -0.0336, 0.0202),
    (0.3, 0.0104, -0.0156, 0.0347),
]

SCENARIO_B = [
    (0.3, -0.1340, -0.171, -0.097),
    (0.6, -0.0376, -0.072, -0.0004),
    (1.0, 0.1049, 0.065, 0.142),
]


class TestDecideAgainstMeasuredFixtures:
    @pytest.mark.parametrize("strength,point_diff,ci_lower,ci_upper", SCENARIO_A)
    def test_scenario_a_is_always_rejected(self, strength, point_diff, ci_lower, ci_upper):
        result = gate.decide(point_diff, ci_lower, ci_upper)
        assert result["promoted"] is False

    @pytest.mark.parametrize("strength,point_diff,ci_lower,ci_upper", SCENARIO_B[:2])
    def test_scenario_b_low_and_mid_strength_rejected(self, strength, point_diff, ci_lower, ci_upper):
        result = gate.decide(point_diff, ci_lower, ci_upper)
        assert result["promoted"] is False

    def test_scenario_b_full_strength_promoted(self):
        strength, point_diff, ci_lower, ci_upper = SCENARIO_B[2]
        result = gate.decide(point_diff, ci_lower, ci_upper)
        assert result["promoted"] is True
        assert result["reasons"]  # non-empty, explains why


class TestDecideEdgeCases:
    def test_interval_above_zero_but_point_diff_below_margin_is_rejected(self):
        # interval [0.001, 0.02] is entirely positive, but 0.015 < 0.02 margin
        result = gate.decide(0.015, 0.001, 0.02)
        assert result["promoted"] is False
        assert result["margin_shortfall"] == pytest.approx(0.005)
        assert result["interval_shortfall"] == 0.0

    def test_point_diff_above_margin_but_interval_includes_zero_is_rejected(self):
        result = gate.decide(0.03, -0.01, 0.05)
        assert result["promoted"] is False
        assert result["margin_shortfall"] == 0.0
        assert result["interval_shortfall"] == pytest.approx(0.01)

    def test_both_conditions_met_is_promoted(self):
        result = gate.decide(0.03, 0.005, 0.05)
        assert result["promoted"] is True
        assert result["margin_shortfall"] == 0.0
        assert result["interval_shortfall"] == 0.0

    def test_holdout_metadata_is_carried_through(self):
        metadata = {"rows": 1409, "label_table": "customers"}
        result = gate.decide(0.03, 0.01, 0.05, holdout_metadata=metadata)
        assert result["holdout_metadata"] == metadata

    def test_custom_margin_is_respected(self):
        result = gate.decide(0.03, 0.01, 0.05, margin=0.05)
        assert result["promoted"] is False  # 0.03 < custom margin 0.05


class TestEvaluateHoldoutMismatch:
    def test_mismatched_x_test_and_y_test_raises(self):
        X_test = pd.DataFrame({"tenure": [1, 2, 3]}, index=["c1", "c2", "c3"])
        y_test = pd.Series([0, 1, 0], index=["c1", "c2", "c4"])  # c3 vs c4 mismatch
        holdout = {"X_test": X_test, "y_test": y_test}

        with pytest.raises(gate.HoldoutMismatchError):
            gate.evaluate(MagicMock(), MagicMock(), holdout)

    def test_matched_x_test_and_y_test_does_not_raise_on_alignment(self):
        ids = ["c1", "c2", "c3"]
        X_test = pd.DataFrame({col: [1, 2, 3] for col in gate.FEATURE_COLUMNS}, index=ids)
        y_test = pd.Series([0, 1, 0], index=ids)
        holdout = {"X_test": X_test, "y_test": y_test, "metadata": {}}

        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]])

        result = gate.evaluate(model, model, holdout)
        assert "pr_auc_diff_point" in result
        assert result["pr_auc_diff_point"] == pytest.approx(0.0)  # same model both sides


class TestBootstrapDeterminism:
    def test_same_seed_gives_identical_results(self):
        rng = np.random.default_rng(0)
        y_test = rng.integers(0, 2, size=200)
        proba_a = rng.random(200)
        proba_b = rng.random(200)

        first = gate.paired_bootstrap_pr_auc_diff(y_test, proba_a, proba_b, n_resamples=200, seed=42)
        second = gate.paired_bootstrap_pr_auc_diff(y_test, proba_a, proba_b, n_resamples=200, seed=42)

        assert first == second

    def test_different_seeds_can_give_different_results(self):
        rng = np.random.default_rng(0)
        y_test = rng.integers(0, 2, size=200)
        proba_a = rng.random(200)
        proba_b = rng.random(200)

        first = gate.paired_bootstrap_pr_auc_diff(y_test, proba_a, proba_b, n_resamples=200, seed=1)
        second = gate.paired_bootstrap_pr_auc_diff(y_test, proba_a, proba_b, n_resamples=200, seed=2)

        assert first != second


class TestFreezeHoldout:
    def _fake_df(self, n=100, seed=0):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(
            {
                "customer_id": [f"c{i}" for i in range(n)],
                "label": rng.integers(0, 2, size=n),
            }
        )
        for col in gate.FEATURE_COLUMNS:
            df[col] = "Yes" if col in ("partner", "dependents") else rng.random(n)
        return df

    def test_metadata_reports_rows_positives_and_label_table(self):
        df = self._fake_df(n=200)

        with patch.object(gate, "_load_labeled_table", return_value=df):
            holdout = gate.freeze_holdout(label_table="customers")

        meta = holdout["metadata"]
        assert meta["label_table"] == "customers"
        assert meta["rows"] == len(holdout["X_test"])
        assert meta["positives"] == int(holdout["y_test"].sum())
        assert meta["frozen_at"]  # non-empty timestamp
        assert len(holdout["X_test"]) + len(holdout["X_train"]) == 200

    def test_split_is_reproducible_with_the_same_random_state(self):
        df = self._fake_df(n=200)

        with patch.object(gate, "_load_labeled_table", return_value=df):
            holdout_1 = gate.freeze_holdout(label_table="customers")
            holdout_2 = gate.freeze_holdout(label_table="customers")

        assert list(holdout_1["X_test"].index) == list(holdout_2["X_test"].index)


class TestPromoteRefusals:
    def test_refuses_after_a_failed_gate(self, tmp_path):
        from src.live import promote as promote_module

        gate_result = {"promoted": False, "reasons": ["interval includes zero"]}
        compat_result = {"passed": True}

        with pytest.raises(promote_module.PromotionRefused):
            promote_module.promote(str(tmp_path), gate_result, compat_result, target_dir=str(tmp_path / "target"))

    def test_refuses_after_a_failed_compat_check(self, tmp_path):
        from src.live import promote as promote_module

        gate_result = {"promoted": True, "reasons": ["clears margin"], "holdout_metadata": {}}
        compat_result = {"passed": False, "max_abs_diff": 0.5}

        with pytest.raises(promote_module.PromotionRefused):
            promote_module.promote(str(tmp_path), gate_result, compat_result, target_dir=str(tmp_path / "target"))

    def test_refuses_with_no_gate_result_at_all(self, tmp_path):
        from src.live import promote as promote_module

        with pytest.raises(promote_module.PromotionRefused):
            promote_module.promote(str(tmp_path), None, {"passed": True}, target_dir=str(tmp_path / "target"))
