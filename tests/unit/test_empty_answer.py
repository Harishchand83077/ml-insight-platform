"""
Empty final answers: /chat retries once, falls back to a fixed message if
the retry is also empty, and never stores an empty turn in session history.
The model is a fake controlled per test; Redis, Postgres, and audit logging
are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage
from prometheus_client import REGISTRY

from src.serving import api


@pytest.fixture
def chat_client():
    from src.serving.auth import get_current_user

    api.app.dependency_overrides[get_current_user] = lambda: {"id": "user-1", "email": "a@example.com"}
    try:
        yield TestClient(api.app)  # no `with`: skips the lifespan, so no model loading
    finally:
        api.app.dependency_overrides.pop(get_current_user, None)


def _post_chat(client, agent, session):
    with patch.dict(api.model_state, {"ready": True}), \
         patch.object(api, "get_agent", return_value=agent), \
         patch.object(api, "log_audit"), \
         patch.object(api, "check_semantic_cache", return_value=None), \
         patch.object(api, "store_in_semantic_cache") as store:
        resp = client.post("/chat", json={"session_id": session, "message": "What is a churn model?"})
    return resp, store


def _empty_message():
    return AIMessage(content="", tool_calls=[], response_metadata={"finish_reason": "stop", "token_usage": {"completion_tokens": 101}})


def _counter_value(outcome):
    return REGISTRY.get_sample_value("agent_empty_answers_total", {"outcome": outcome}) or 0.0


class TestRetrySucceeds:
    def test_empty_then_nonempty_returns_the_retried_answer(self, chat_client):
        calls = {"n": 0}

        def invoke(payload, config=None):
            calls["n"] += 1
            msg = _empty_message() if calls["n"] == 1 else AIMessage(content="Here is your answer.")
            return {"messages": payload["messages"] + [msg]}

        agent = MagicMock()
        agent.invoke.side_effect = invoke
        before = _counter_value("retried_ok")

        resp, store = _post_chat(chat_client, agent, "retry-ok")

        assert resp.status_code == 200
        assert resp.json()["response"] == "Here is your answer."
        assert calls["n"] == 2
        assert _counter_value("retried_ok") == before + 1
        store.assert_called_once()  # a real answer is still eligible for the semantic cache

        history = api.chat_sessions[("user-1", "retry-ok")]
        assert history[-1].content == "Here is your answer."
        assert all(
            (m.content or "").strip()
            for m in history
            if isinstance(m, BaseMessage) and not getattr(m, "tool_calls", None)
        )


class TestRetryAlsoFails:
    def test_empty_twice_returns_the_fallback(self, chat_client):
        agent = MagicMock()
        agent.invoke.side_effect = lambda payload, config=None: {"messages": payload["messages"] + [_empty_message()]}
        before = _counter_value("fallback")

        resp, store = _post_chat(chat_client, agent, "retry-fail")

        assert resp.status_code == 200
        assert resp.json()["response"] == api.EMPTY_ANSWER_FALLBACK
        assert agent.invoke.call_count == 2
        assert _counter_value("fallback") == before + 1
        store.assert_not_called()  # the fallback message is never cached as a real answer

        history = api.chat_sessions[("user-1", "retry-fail")]
        assert history[-1].content == api.EMPTY_ANSWER_FALLBACK
        assert history[-1].content.strip()  # never an empty turn


class TestNormalAnswersUntouched:
    def test_a_non_empty_first_answer_is_not_retried(self, chat_client):
        agent = MagicMock()
        agent.invoke.side_effect = lambda payload, config=None: {
            "messages": payload["messages"] + [AIMessage(content="The churn rate is 26.5%.")]
        }

        resp, store = _post_chat(chat_client, agent, "normal")

        assert resp.status_code == 200
        assert resp.json()["response"] == "The churn rate is 26.5%."
        assert agent.invoke.call_count == 1
        store.assert_called_once()

    def test_a_final_message_with_tool_calls_is_not_treated_as_empty(self, chat_client):
        # Shouldn't happen in practice (the graph loops while tool_calls are
        # present), but empty content plus a pending tool call is not the
        # "model gave up" case this handles.
        agent = MagicMock()
        agent.invoke.side_effect = lambda payload, config=None: {
            "messages": payload["messages"]
            + [AIMessage(content="", tool_calls=[{"name": "get_customer_count", "args": {}, "id": "c1"}])]
        }

        resp, _ = _post_chat(chat_client, agent, "tool-call-only")

        assert resp.status_code == 200
        assert agent.invoke.call_count == 1
        assert resp.json()["response"] == ""
