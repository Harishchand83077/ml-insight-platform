"""
Integration test for the real /chat conversation flow: session memory
across two turns, and DELETE /chat/{session_id} actually clearing it.

This makes real Groq API calls (2 short exchanges, kept deliberately
minimal to avoid burning free-tier quota) and needs Redis running (the
semantic-cache check on the first message hits Redis unconditionally).
Skips with a clear message if GROQ_API_KEY or Redis isn't available,
rather than silently passing.
"""

import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client(redis_up, groq_key_present):
    if not redis_up:
        pytest.skip("Redis not reachable on localhost:6379 - required by /chat's semantic cache check")
    if not groq_key_present:
        pytest.skip("GROQ_API_KEY not set (.env) - required to call the agent's LLM")

    try:
        from src.serving.api import app
    except Exception as e:
        pytest.skip(f"Could not import src.serving.api (missing mlruns.db or a logged xgboost run?): {e}")

    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


def test_session_memory_and_clear(client):
    session_id = f"pytest-{uuid.uuid4()}"

    # turn 1: give the agent a fact to remember, a question with no tool needed
    first = client.post(
        "/chat",
        json={"session_id": session_id, "message": "Remember this codeword: PINEAPPLE42. Just say OK."},
    )
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["session_id"] == session_id
    assert isinstance(first_data["response"], str) and first_data["response"]

    # turn 2: recall it - only possible if session history was actually passed back in
    second = client.post(
        "/chat",
        json={"session_id": session_id, "message": "What was the codeword I just gave you?"},
    )
    assert second.status_code == 200
    second_data = second.json()
    assert "pineapple42" in second_data["response"].lower()

    # clearing the session works and is idempotent-safe (reports whether it existed)
    delete_first = client.delete(f"/chat/{session_id}")
    assert delete_first.status_code == 200
    assert delete_first.json() == {"session_id": session_id, "cleared": True}

    delete_second = client.delete(f"/chat/{session_id}")
    assert delete_second.json() == {"session_id": session_id, "cleared": False}

    # after clearing, a fresh message in the same session_id has no memory of the codeword
    third = client.post(
        "/chat",
        json={"session_id": session_id, "message": "What was the codeword I gave you earlier?"},
    )
    assert third.status_code == 200
    assert "pineapple42" not in third.json()["response"].lower()
