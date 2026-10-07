"""
Failure-path tests that were missing: duplicate signup (409), what-if override
validation, retention rule selection, and duplicate feedback. The duplicate
feedback test documents what happens today; it doesn't assert a desired rule.
Database, model and explanation calls are mocked.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.serving import api, feedback, prediction
from src.serving.auth import EmailAlreadyExistsError


# --- duplicate signup -------------------------------------------------------

@pytest.fixture
def signup_client():
    return TestClient(api.app)  # no `with`: no lifespan, no model loading


class TestDuplicateSignup:
    def test_existing_email_is_rejected_by_the_precheck_with_409(self, signup_client):
        with patch.object(api, "get_user_by_email", return_value={"id": "u1", "email": "a@example.com"}), \
             patch.object(api, "create_user") as create:
            resp = signup_client.post("/auth/signup", json={"email": "a@example.com", "password": "TestPass123!"})

        assert resp.status_code == 409
        assert resp.json()["detail"] == "email already registered"
        create.assert_not_called()

    def test_race_on_insert_is_also_409(self, signup_client):
        # The precheck passed, but another request inserted the same email first.
        with patch.object(api, "get_user_by_email", return_value=None), \
             patch.object(api, "create_user", side_effect=EmailAlreadyExistsError("a@example.com")):
            resp = signup_client.post("/auth/signup", json={"email": "a@example.com", "password": "TestPass123!"})

        assert resp.status_code == 409
        assert resp.json()["detail"] == "email already registered"


# --- what-if override validation -------------------------------------------

class TestValidateOverrides:
    def test_valid_overrides_are_returned_with_numbers_coerced(self):
        clean = prediction._validate_overrides({"contract": "Two year", "tenure": 12.0})
        assert clean == {"contract": "Two year", "tenure": 12}
        assert isinstance(clean["tenure"], int)

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"contract": "Lifetime"}, "not a valid contract"),
            ({"payment_method": "Bitcoin"}, "not a valid payment_method"),
            ({"internet_service": "Satellite"}, "not a valid internet_service"),
            ({"partner": "Maybe"}, "not a valid partner"),
        ],
        ids=["bad-contract", "bad-payment", "bad-internet", "bad-partner"],
    )
    def test_bad_category_is_rejected(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            prediction._validate_overrides(overrides)

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"tenure": 100}, "outside the valid range"),
            ({"tenure": -1}, "outside the valid range"),
            ({"monthly_charges": 500.0}, "outside the valid range"),
            ({"senior_citizen": 2}, "outside the valid range"),
            ({"num_addons_active": 7}, "outside the valid range"),
        ],
        ids=["tenure-high", "tenure-low", "charges-high", "senior-out", "addons-high"],
    )
    def test_out_of_range_number_is_rejected(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            prediction._validate_overrides(overrides)

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"tenure": "abc"}, "must be a number"),
            ({"tenure": True}, "must be a number"),
            ({"monthly_charges": None}, "must be a number"),
            ({"tenure": [12]}, "must be a number"),
            ({"tenure": 3.5}, "must be a whole number"),
        ],
        ids=["string", "bool", "none", "list", "fraction-for-int"],
    )
    def test_wrong_type_is_rejected(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            prediction._validate_overrides(overrides)

    def test_empty_override_set_is_rejected(self):
        with pytest.raises(ValueError, match="No overrides given"):
            prediction._validate_overrides({})

    def test_field_that_is_not_changeable_is_rejected(self):
        # total_logins_90d is behavioural, not an account field a plan change can set.
        with pytest.raises(ValueError, match="cannot be changed in a what-if"):
            prediction._validate_overrides({"total_logins_90d": 3})


# --- retention rule selection ----------------------------------------------

def _explanation(probability, top_features):
    return {"customer_id": "7590-VHVEG", "churn_probability": probability, "top_features": top_features}


def _factor(feature, log_odds=0.8, value=1):
    return {"feature": feature, "value": value, "log_odds": log_odds}


class TestRetentionRuleSelection:
    def test_low_risk_gives_no_action_even_with_a_risk_factor(self):
        explanation = _explanation(0.1, [_factor("contract", 0.9, "Month-to-month")])
        with patch.object(prediction, "explain_prediction", return_value=explanation):
            result = prediction.recommend_retention_action("7590-VHVEG")

        assert result["priority"] == "low"
        assert result["action"] is None
        assert "No retention action is needed" in result["note"]

    @pytest.mark.parametrize("feature", sorted(prediction._RETENTION_RULES))
    def test_each_rule_maps_to_its_driver(self, feature):
        rule = prediction._RETENTION_RULES[feature]
        explanation = _explanation(0.6, [_factor(feature, 0.8)])
        simulated = {
            "overrides": rule["simulate"],
            "original_probability": 0.6,
            "modified_probability": 0.4,
            "delta": -0.2,
        }
        with patch.object(prediction, "explain_prediction", return_value=explanation), \
             patch.object(prediction, "simulate_prediction", return_value=simulated):
            result = prediction.recommend_retention_action("7590-VHVEG")

        assert result["priority"] == "high"
        assert result["triggering_factor"]["feature"] == feature
        assert result["action"] == rule["action"]
        assert (result["projected"] is not None) == (rule["simulate"] is not None)

    def test_medium_priority_still_gets_an_action(self):
        explanation = _explanation(0.3, [_factor("tenure", 0.5)])
        with patch.object(prediction, "explain_prediction", return_value=explanation):
            result = prediction.recommend_retention_action("7590-VHVEG")

        assert result["priority"] == "medium"
        assert result["action"] == prediction._RETENTION_RULES["tenure"]["action"]

    def test_no_factor_pushing_risk_up_gives_no_rule(self):
        explanation = _explanation(0.6, [_factor("tenure", -0.4)])
        with patch.object(prediction, "explain_prediction", return_value=explanation):
            result = prediction.recommend_retention_action("7590-VHVEG")

        assert result["action"] is None
        assert "No factor is pushing" in result["note"]

    def test_unknown_top_factor_says_no_rule_covers_it(self):
        explanation = _explanation(0.6, [_factor("some_unmapped_feature")])
        with patch.object(prediction, "explain_prediction", return_value=explanation):
            result = prediction.recommend_retention_action("7590-VHVEG")

        assert result["action"] is None
        assert "No rule covers" in result["note"]

    def test_unknown_customer_returns_none(self):
        with patch.object(prediction, "explain_prediction", return_value=None):
            assert prediction.recommend_retention_action("0000-ZZZZZ") is None


# --- duplicate feedback ----------------------------------------------------

class TestDuplicateFeedbackCurrentBehaviour:
    def test_same_feedback_submitted_twice_is_stored_twice_today(self):
        """Documents today's behaviour. The feedback table has only a primary key
        (no unique constraint on user, session and message), so a repeated
        submission inserts a second row. This test does not assert that this is
        the desired rule."""
        executed = []

        class _Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                executed.append(sql)

        class _Conn:
            def cursor(self):
                return _Cursor()

            def commit(self):
                executed.append("COMMIT")

        class _Pool:
            def getconn(self):
                return _Conn()

            def putconn(self, conn, close=False):
                pass

        with patch.object(feedback, "_get_pool", return_value=_Pool()):
            feedback.store_feedback("user-1", "session-1", "helpful answer", "up")
            feedback.store_feedback("user-1", "session-1", "helpful answer", "up")

        inserts = [s for s in executed if s.startswith("INSERT INTO feedback")]
        assert len(inserts) == 2
