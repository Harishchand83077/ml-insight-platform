"""
Unit tests for src/agent/semantic_cache.py's pure logic (cosine similarity,
threshold behavior). Redis and the embedding model are both mocked - no
live Redis, no model download needed. check_semantic_cache/
store_in_semantic_cache are async (they await asyncio.to_thread
internally - see semantic_cache.py's module docstring), so calls to them
here go through asyncio.run() rather than a full async test framework -
not worth a pytest-asyncio dependency for two call sites.
"""

import asyncio
import json
import math
import time
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


def _cached_entry_json(question, similarity_to_query, response="cached response", tool_calls=None, created_at=None):
    return json.dumps(
        {
            "question": question,
            "embedding": _vector_at_similarity(similarity_to_query),
            "response": response,
            "tool_calls": tool_calls or [{"tool": "predict_churn_tool", "args": {"customer_id": "1234"}}],
            "created_at": time.time() if created_at is None else created_at,
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
            result = asyncio.run(semantic_cache.check_semantic_cache("new question"))

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
            result = asyncio.run(semantic_cache.check_semantic_cache("new question, worded differently"))

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
            result = asyncio.run(semantic_cache.check_semantic_cache("a question"))

        assert result["response"] == "right match"

    def test_empty_cache_is_a_miss_without_embedding_the_question(self):
        redis_patch, emb_patch, _, fake_embeddings = _patch_redis_and_embeddings([])

        with redis_patch, emb_patch:
            result = asyncio.run(semantic_cache.check_semantic_cache("anything"))

        assert result is None
        fake_embeddings.embed_query.assert_not_called()


class TestStoreInSemanticCache:
    def test_stores_entry_and_trims_to_max_size(self):
        fake_redis = MagicMock()
        fake_embeddings = MagicMock()
        fake_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        with patch.object(semantic_cache, "_get_redis_client", return_value=fake_redis), \
             patch.object(semantic_cache, "get_embedder", return_value=fake_embeddings):
            asyncio.run(semantic_cache.store_in_semantic_cache(
                "a new question", "a response", [{"tool": "predict_churn_tool", "args": {}}]
            ))

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
        assert stored["created_at"] == pytest.approx(time.time(), abs=5)


class TestCustomerIdExclusion:
    @pytest.mark.parametrize(
        "question",
        [
            "What's the churn risk for customer 7590-VHVEG?",
            "What if customer 2691-NZETQ switched to a 2-year contract?",
            "2691-NZETQ",
        ],
    )
    def test_id_detected(self, question):
        assert semantic_cache.contains_customer_id(question)

    @pytest.mark.parametrize(
        "question",
        [
            "What if they switched to a 2-year contract?",
            "What percentage of month-to-month customers churn?",
            "Which contract type churns most in 2024?",
            "Which contract type churns most, 2-year or 1-year?",
        ],
    )
    def test_no_id_detected(self, question):
        assert not semantic_cache.contains_customer_id(question)

    def test_id_question_never_reads_cache(self):
        # Cache holds an identical-meaning entry that would otherwise match exactly.
        entry = _cached_entry_json("What's the churn risk for customer 7590-VHVEG?", similarity_to_query=1.0)
        redis_patch, emb_patch, fake_redis, fake_embeddings = _patch_redis_and_embeddings([entry])

        with redis_patch, emb_patch:
            result = asyncio.run(semantic_cache.check_semantic_cache(
                "What's the churn risk for customer 2691-NZETQ?"
            ))

        assert result is None
        fake_redis.lrange.assert_not_called()
        fake_embeddings.embed_query.assert_not_called()

    def test_id_question_never_writes_cache(self):
        fake_redis = MagicMock()
        fake_embeddings = MagicMock()

        with patch.object(semantic_cache, "_get_redis_client", return_value=fake_redis), \
             patch.object(semantic_cache, "get_embedder", return_value=fake_embeddings):
            asyncio.run(semantic_cache.store_in_semantic_cache(
                "What if customer 2691-NZETQ switched to a 2-year contract?",
                "a response",
                [],
            ))

        fake_redis.lpush.assert_not_called()
        fake_redis.ltrim.assert_not_called()
        fake_embeddings.embed_query.assert_not_called()


class TestRewordingStillHits:
    def test_reworded_non_id_question_hits(self):
        entry = _cached_entry_json(
            "What percentage of month-to-month customers churn?",
            similarity_to_query=0.92,
            response="the churn rate answer",
        )
        redis_patch, emb_patch, _, _ = _patch_redis_and_embeddings([entry])

        with redis_patch, emb_patch:
            result = asyncio.run(semantic_cache.check_semantic_cache(
                "What share of customers on month-to-month contracts churn?"
            ))

        assert result is not None
        assert result["response"] == "the churn rate answer"


class TestTtl:
    def test_expired_entry_is_ignored(self):
        old = time.time() - semantic_cache.TTL_SECONDS - 60
        entry = _cached_entry_json("an old question", similarity_to_query=1.0, created_at=old)
        redis_patch, emb_patch, _, _ = _patch_redis_and_embeddings([entry])

        with redis_patch, emb_patch:
            result = asyncio.run(semantic_cache.check_semantic_cache("an old question"))

        assert result is None

    def test_entry_without_timestamp_is_ignored(self):
        legacy = json.dumps({
            "question": "a pre-TTL question",
            "embedding": _vector_at_similarity(1.0),
            "response": "legacy answer",
            "tool_calls": [],
        })
        redis_patch, emb_patch, _, _ = _patch_redis_and_embeddings([legacy])

        with redis_patch, emb_patch:
            result = asyncio.run(semantic_cache.check_semantic_cache("a pre-TTL question"))

        assert result is None

    def test_fresh_match_wins_over_expired_better_match(self):
        old = time.time() - semantic_cache.TTL_SECONDS - 60
        entries = [
            _cached_entry_json("expired exact", similarity_to_query=1.0, response="stale", created_at=old),
            _cached_entry_json("fresh near match", similarity_to_query=0.95, response="fresh"),
        ]
        redis_patch, emb_patch, _, _ = _patch_redis_and_embeddings(entries)

        with redis_patch, emb_patch:
            result = asyncio.run(semantic_cache.check_semantic_cache("a question"))

        assert result["response"] == "fresh"
