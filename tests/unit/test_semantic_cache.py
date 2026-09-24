"""
Unit tests for src/agent/semantic_cache.py's pure logic (cosine similarity,
threshold behavior). Redis and the embedding model are both mocked - no
live Redis, no model download needed.
"""

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from src.agent import semantic_cache


class TestCosineSimilarity:
    def test_identical_vectors_similarity_one(self):
        v = [1.0, 2.0, 3.0]
        assert semantic_cache._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_similarity_zero(self):
        assert semantic_cache._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-9)

    def test_opposite_vectors_similarity_negative_one(self):
        assert semantic_cache._cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_scale_invariant(self):
        # cosine similarity ignores magnitude, only direction matters
        a = [1.0, 0.0]
        b = [5.0, 0.0]
        assert semantic_cache._cosine_similarity(a, b) == pytest.approx(1.0)


def _vector_at_similarity(similarity):
    """A 2D unit vector whose cosine similarity with [1, 0] is exactly `similarity`."""
    return [similarity, math.sqrt(1 - similarity**2)]


def _cached_entry_json(question, similarity_to_query, response="cached response", tool_calls=None):
    return json.dumps(
        {
            "question": question,
            "embedding": _vector_at_similarity(similarity_to_query),
            "response": response,
            "tool_calls": tool_calls or [{"tool": "predict_churn_tool", "args": {"customer_id": "1234"}}],
        }
    )


def _patch_redis_and_embeddings(raw_entries):
    fake_redis = MagicMock()
    fake_redis.lrange.return_value = raw_entries
    fake_embeddings = MagicMock()
    fake_embeddings.embed_query.return_value = [1.0, 0.0]  # the query vector
    return (
        patch.object(semantic_cache, "_get_redis_client", return_value=fake_redis),
        patch.object(semantic_cache, "get_embedder", return_value=fake_embeddings),
        fake_redis,
        fake_embeddings,
    )


class TestThresholdLogic:
    def test_below_threshold_is_a_miss(self):
        entry = _cached_entry_json("some cached question", similarity_to_query=0.89)
        redis_patch, emb_patch, _, _ = _patch_redis_and_embeddings([entry])

        with redis_patch, emb_patch:
            result = semantic_cache.check_semantic_cache("new question")

        assert result is None

    def test_above_threshold_is_a_hit(self):
        entry = _cached_entry_json(
            "some cached question",
            similarity_to_query=0.91,
            response="the cached answer",
            tool_calls=[{"tool": "get_customer_count", "args": {"filters": {}}}],
        )
        redis_patch, emb_patch, _, _ = _patch_redis_and_embeddings([entry])

        with redis_patch, emb_patch:
            result = semantic_cache.check_semantic_cache("new question, worded differently")

        assert result == {
            "response": "the cached answer",
            "tool_calls": [{"tool": "get_customer_count", "args": {"filters": {}}}],
        }

    def test_picks_the_best_match_among_multiple_entries(self):
        entries = [
            _cached_entry_json("low similarity question", 0.50, response="wrong match"),
            _cached_entry_json("high similarity question", 0.95, response="right match"),
            _cached_entry_json("mid similarity question", 0.70, response="also wrong"),
        ]
        redis_patch, emb_patch, _, _ = _patch_redis_and_embeddings(entries)

        with redis_patch, emb_patch:
            result = semantic_cache.check_semantic_cache("a question")

        assert result["response"] == "right match"

    def test_empty_cache_is_a_miss_without_embedding_the_question(self):
        redis_patch, emb_patch, _, fake_embeddings = _patch_redis_and_embeddings([])

        with redis_patch, emb_patch:
            result = semantic_cache.check_semantic_cache("anything")

        assert result is None
        fake_embeddings.embed_query.assert_not_called()


class TestStoreInSemanticCache:
    def test_stores_entry_and_trims_to_max_size(self):
        fake_redis = MagicMock()
        fake_embeddings = MagicMock()
        fake_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        with patch.object(semantic_cache, "_get_redis_client", return_value=fake_redis), \
             patch.object(semantic_cache, "get_embedder", return_value=fake_embeddings):
            semantic_cache.store_in_semantic_cache(
                "a new question", "a response", [{"tool": "predict_churn_tool", "args": {}}]
            )

        fake_redis.lpush.assert_called_once()
        key, payload = fake_redis.lpush.call_args[0]
        assert key == semantic_cache.CACHE_KEY
        stored = json.loads(payload)
        assert stored["question"] == "a new question"
        assert stored["response"] == "a response"
        assert stored["embedding"] == [0.1, 0.2, 0.3]

        fake_redis.ltrim.assert_called_once_with(
            semantic_cache.CACHE_KEY, 0, semantic_cache.MAX_ENTRIES - 1
        )
