"""
Redis is best-effort: a Redis error or timeout must never fail /predict or
/chat. Each test makes Redis raise, then checks that the request still
succeeds through its fallback. Redis, Postgres, the model, and the agent are
all mocked.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import redis
from fastapi.testclient import TestClient

from src.agent import semantic_cache
from src.serving import api, feature_cache, prediction

ROW = {
    "customer_id": "7590-VHVEG",
    "tenure": 12,
    "monthly_charges": 55.0,
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "num_addons_active": 2,
    "total_logins_90d": 20,
    "recent_30d_vs_older_60d_ratio": 1.0,
    "avg_session_duration_recent_30d": 300.0,
    "num_tickets": 1,
    "pct_unresolved": 0.0,
    "avg_resolution_time_hours": 5.0,
    "avg_usage_count": 10.0,
}

REDIS_ERRORS = [redis.ConnectionError("down"), redis.TimeoutError("timed out"), TimeoutError("socket timed out")]


class _Cursor:
    def __init__(self, row):
        self._row = row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return self._row


class _Conn:
    def __init__(self, row):
        self._row = row

    def cursor(self, cursor_factory=None):
        return _Cursor(self._row)


class _Pool:
    """Stands in for the shared pool: every checkout returns a connection whose
    single lookup yields `row`."""

    def __init__(self, row):
        self._row = row

    def getconn(self):
        return _Conn(self._row)

    def putconn(self, conn, close=False):
        pass


def _broken_redis(error):
    client = MagicMock()
    client.get.side_effect = error
    client.setex.side_effect = error
    client.lrange.side_effect = error
    client.lpush.side_effect = error
    client.ltrim.side_effect = error
    return client


class TestFeatureCacheFailsOpen:
    @pytest.mark.parametrize("error", REDIS_ERRORS, ids=lambda e: type(e).__name__)
    def test_get_and_setex_failures_fall_back_to_postgres(self, error):
        with patch.object(feature_cache, "get_redis_client", return_value=_broken_redis(error)), \
             patch.object(feature_cache, "_get_pool", return_value=_Pool(ROW)):
            features, cache_hit = feature_cache.get_customer_features("7590-VHVEG")

        assert cache_hit is False
        assert features["customer_id"] == "7590-VHVEG"

    def test_predict_returns_the_postgres_value_with_cache_hit_false(self):
        fake_pipeline = MagicMock()
        fake_pipeline.predict_proba.return_value = np.array([[0.3, 0.7]])
        with patch.object(feature_cache, "get_redis_client", return_value=_broken_redis(redis.ConnectionError("down"))), \
             patch.object(feature_cache, "_get_pool", return_value=_Pool(ROW)), \
             patch.object(prediction, "get_pipeline", return_value=fake_pipeline):
            result = prediction.predict_churn("7590-VHVEG")

        assert result is not None
        assert result["cache_hit"] is False
        assert result["churn_probability"] == pytest.approx(0.7)

    def test_missing_customer_still_returns_none_when_redis_is_down(self):
        with patch.object(feature_cache, "get_redis_client", return_value=_broken_redis(redis.ConnectionError("down"))), \
             patch.object(feature_cache, "_get_pool", return_value=_Pool(None)):
            features, cache_hit = feature_cache.get_customer_features("0000-ZZZZZ")

        assert features is None
        assert cache_hit is False


class TestSemanticCacheFailsOpen:
    @pytest.mark.parametrize("error", REDIS_ERRORS, ids=lambda e: type(e).__name__)
    def test_check_treats_a_redis_error_as_a_miss(self, error):
        import asyncio

        with patch.object(semantic_cache, "_get_redis_client", return_value=_broken_redis(error)):
            result = asyncio.run(semantic_cache.check_semantic_cache("What is a churn model?"))

        assert result is None

    @pytest.mark.parametrize("error", REDIS_ERRORS, ids=lambda e: type(e).__name__)
    def test_store_skips_silently_on_a_redis_error(self, error):
        import asyncio

        fake_embeddings = MagicMock()
        fake_embeddings.embed_query.return_value = [0.1, 0.2]
        with patch.object(semantic_cache, "_get_redis_client", return_value=_broken_redis(error)), \
             patch.object(semantic_cache, "get_embedder", return_value=fake_embeddings):
            asyncio.run(semantic_cache.store_in_semantic_cache(
                "What is a churn model?", "an answer", [{"tool": "query_project_docs_tool", "args": {}}]
            ))  # must not raise


@pytest.fixture
def chat_client():
    from src.serving.auth import get_current_user

    api.app.dependency_overrides[get_current_user] = lambda: {"id": "user-1", "email": "a@example.com"}
    try:
        yield TestClient(api.app)  # no `with`: skips the lifespan, so no model loading
    finally:
        api.app.dependency_overrides.pop(get_current_user, None)


class TestChatProceedsWithRedisDown:
    def test_chat_reaches_the_agent_when_lrange_raises(self, chat_client):
        answer = MagicMock(content="The early termination fee is covered in the contract docs.", tool_calls=[])

        def fake_invoke(payload, config=None):
            return {"messages": payload["messages"] + [answer]}

        fake_agent = MagicMock()
        fake_agent.invoke.side_effect = fake_invoke
        fake_embeddings = MagicMock()
        fake_embeddings.embed_query.return_value = [0.1, 0.2]

        with patch.dict(api.model_state, {"ready": True}), \
             patch.object(api, "get_agent", return_value=fake_agent), \
             patch.object(api, "log_audit"), \
             patch.object(semantic_cache, "_get_redis_client", return_value=_broken_redis(redis.ConnectionError("down"))), \
             patch.object(semantic_cache, "get_embedder", return_value=fake_embeddings):
            resp = chat_client.post(
                "/chat",
                json={"session_id": "fail-open-test", "message": "What is a churn model?"},
            )

        assert resp.status_code == 200
        assert resp.json()["response"] == answer.content
        fake_agent.invoke.assert_called_once()

    @pytest.mark.parametrize("error", REDIS_ERRORS, ids=lambda e: type(e).__name__)
    def test_chat_succeeds_on_any_redis_timeout_type(self, chat_client, error):
        answer = MagicMock(content="ok", tool_calls=[])
        fake_agent = MagicMock()
        fake_agent.invoke.side_effect = lambda payload, config=None: {"messages": payload["messages"] + [answer]}
        fake_embeddings = MagicMock()
        fake_embeddings.embed_query.return_value = [0.1, 0.2]

        with patch.dict(api.model_state, {"ready": True}), \
             patch.object(api, "get_agent", return_value=fake_agent), \
             patch.object(api, "log_audit"), \
             patch.object(semantic_cache, "_get_redis_client", return_value=_broken_redis(error)), \
             patch.object(semantic_cache, "get_embedder", return_value=fake_embeddings):
            resp = chat_client.post(
                "/chat",
                json={"session_id": f"timeout-{type(error).__name__}", "message": "What is a churn model?"},
            )

        assert resp.status_code == 200


class TestSetexSkippedAfterGetFailure:
    def test_failed_get_skips_the_setex_for_that_request(self):
        fake_redis = _broken_redis(redis.TimeoutError("timed out"))
        with patch.object(feature_cache, "get_redis_client", return_value=fake_redis), \
             patch.object(feature_cache, "_get_pool", return_value=_Pool(ROW)):
            feature_cache.get_customer_features("7590-VHVEG")

        fake_redis.get.assert_called_once()
        fake_redis.setex.assert_not_called()

    def test_successful_get_then_failed_setex_still_returns_the_row(self):
        fake_redis = MagicMock()
        fake_redis.get.return_value = None  # miss, so SETEX is attempted
        fake_redis.setex.side_effect = redis.ConnectionError("down")
        with patch.object(feature_cache, "get_redis_client", return_value=fake_redis), \
             patch.object(feature_cache, "_get_pool", return_value=_Pool(ROW)):
            features, cache_hit = feature_cache.get_customer_features("7590-VHVEG")

        fake_redis.setex.assert_called_once()
        assert features["customer_id"] == "7590-VHVEG"
        assert cache_hit is False
