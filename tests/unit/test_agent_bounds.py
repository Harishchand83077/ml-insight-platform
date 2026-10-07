"""
The agent is bounded and its failures become clean responses:
- a tool loop hits the recursion limit and returns a friendly message (200),
- a Groq timeout or connection error returns 503 with a retry hint,
- a database error inside a SQL tool becomes an error string for the model,
  not an exception that fails /chat.
The model, Groq, and the database are mocked; the LangGraph loop is real.
"""

from unittest.mock import MagicMock, patch

import groq
import httpx
import psycopg2
import psycopg2.errors
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.tools import get_customer_count, get_churn_rate_by_column
from src.serving import api

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


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
         patch.object(api, "store_in_semantic_cache"):
        return client.post("/chat", json={"session_id": session, "message": "How many customers are on contracts?"})


def _looping_agent():
    """A real LangGraph loop: the model node always requests get_customer_count,
    the ToolNode runs the real tool, and the edge goes back to the model. The
    only way out is the recursion limit."""

    def model(state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_customer_count", "args": {"filters": {}}, "id": "call-loop"}],
                )
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("model", model)
    graph.add_node("tools", ToolNode([get_customer_count]))
    graph.add_edge(START, "model")
    graph.add_edge("model", "tools")
    graph.add_edge("tools", "model")
    return graph.compile()


class TestRecursionLimit:
    def test_invoke_is_called_with_the_recursion_limit(self, chat_client):
        agent = MagicMock()
        agent.invoke.side_effect = lambda payload, config=None: {"messages": payload["messages"] + [AIMessage(content="ok")]}

        _post_chat(chat_client, agent, "limit-config")

        _, kwargs = agent.invoke.call_args
        assert kwargs["config"] == {"recursion_limit": api.AGENT_RECURSION_LIMIT}
        assert api.AGENT_RECURSION_LIMIT == 12

    def test_forced_tool_loop_returns_the_friendly_message_not_a_500(self, chat_client):
        with patch("src.agent.tools.get_shared_pool", return_value="fake-pool-not-a-real-connection"), \
             patch("src.agent.tools.run_read", return_value=[(7043,)]) as fake_read:
            resp = _post_chat(chat_client, _looping_agent(), "limit-loop")

        assert resp.status_code == 200
        assert resp.json()["response"] == "I couldn't finish that request. Try a more specific question."
        assert resp.json()["tool_calls"] == []
        # the loop really ran the tool several times before the limit stopped it
        assert fake_read.call_count >= 2

    def test_a_two_tool_answer_completes_under_the_limit(self, chat_client):
        # model -> tools -> model -> tools -> model: five steps, well under 12
        calls = {"n": 0}

        def model(state):
            calls["n"] += 1
            if calls["n"] <= 2:
                name = "get_customer_count"
                return {"messages": [AIMessage(content="", tool_calls=[{"name": name, "args": {"filters": {}}, "id": f"c{calls['n']}"}])]}
            return {"messages": [AIMessage(content="There are 7043 customers.")]}

        graph = StateGraph(MessagesState)
        graph.add_node("model", model)
        graph.add_node("tools", ToolNode([get_customer_count]))
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", lambda s: "tools" if s["messages"][-1].tool_calls else END)
        graph.add_edge("tools", "model")

        with patch("src.agent.tools.get_shared_pool", return_value="fake-pool-not-a-real-connection"), \
             patch("src.agent.tools.run_read", return_value=[(7043,)]):
            resp = _post_chat(chat_client, graph.compile(), "two-tools")

        assert resp.status_code == 200
        assert resp.json()["response"] == "There are 7043 customers."
        assert len(resp.json()["tool_calls"]) == 2


class TestGroqTimeoutAndConnectionErrors:
    @pytest.mark.parametrize(
        "error",
        [
            groq.APITimeoutError(request=REQUEST),
            groq.APIConnectionError(request=REQUEST),
        ],
        ids=["APITimeoutError", "APIConnectionError"],
    )
    def test_returns_503_with_a_retry_hint(self, chat_client, error):
        agent = MagicMock()
        agent.invoke.side_effect = error

        resp = _post_chat(chat_client, agent, f"groq-{type(error).__name__}")

        assert resp.status_code == 503
        assert "retry" in resp.json()["error"].lower()

    def test_rate_limit_still_returns_503_with_its_own_message(self, chat_client):
        agent = MagicMock()
        agent.invoke.side_effect = groq.APIStatusError(
            "rate limited", response=httpx.Response(429, request=REQUEST), body=None
        )

        resp = _post_chat(chat_client, agent, "groq-429")

        assert resp.status_code == 503
        assert "rate-limited" in resp.json()["error"]


class TestSqlToolErrorsBecomeToolMessages:
    def test_count_tool_returns_an_error_string_on_db_failure(self):
        with patch("src.agent.tools.get_shared_pool", return_value="fake-pool-not-a-real-connection"), \
             patch("src.agent.tools.run_read", side_effect=psycopg2.OperationalError("database is down")):
            result = get_customer_count.invoke({"filters": {}})

        assert result.startswith("Error counting customers")
        assert "database is down" in result

    def test_churn_rate_tool_returns_an_error_string_on_db_failure(self):
        with patch("src.agent.tools.get_shared_pool", return_value="fake-pool-not-a-real-connection"), \
             patch("src.agent.tools.run_read", side_effect=psycopg2.errors.QueryCanceled("statement timeout")):
            result = get_churn_rate_by_column.invoke({"column_name": "contract"})

        assert result.startswith("Error computing the churn rate by contract")
        assert "statement timeout" in result

    def test_sql_tools_set_a_statement_timeout_on_the_query(self):
        from src.agent import tools

        executed = []

        class _Cur:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                executed.append(sql)

            def fetchall(self):
                return [(7043,)]

        class _Conn:
            def cursor(self):
                return _Cur()

        def fake_run_read(pool, query):
            return query(_Conn())

        with patch.object(tools, "get_shared_pool", return_value="fake-pool-not-a-real-connection"), \
             patch.object(tools, "run_read", side_effect=fake_run_read):
            tools.get_customer_count.invoke({"filters": {}})

        assert executed[0] == "SET LOCAL statement_timeout = '5s'"
