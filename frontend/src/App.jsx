import { useEffect, useRef, useState } from "react";
import Message from "./components/Message.jsx";
import ChatInput from "./components/ChatInput.jsx";
import AuthScreen from "./components/AuthScreen.jsx";
import { sendChatMessage, clearChatSession, setAuthToken, setOnUnauthorized } from "./api.js";
import "./App.css";

function newSessionId() {
  return crypto.randomUUID();
}

export default function App() {
  // Deliberately React state, not localStorage: an XSS payload that can run
  // in this page can read anything localStorage holds, indefinitely, but
  // can only read in-memory state for as long as this page stays open. The
  // trade-off is a page refresh always logs the user out - acceptable for
  // this project's scope. A production app would use httpOnly cookies
  // instead, which survive a refresh without ever being readable by
  // JavaScript in the first place (closing the risk rather than just
  // shortening its window).
  const [token, setToken] = useState(null);
  const [sessionId, setSessionId] = useState(newSessionId);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [slowLoading, setSlowLoading] = useState(false);
  const [error, setError] = useState(null);
  const scrollRef = useRef(null);
  const slowLoadingTimerRef = useRef(null);

  // Mirrors token into api.js's axios interceptor, which can't read React
  // state directly.
  useEffect(() => {
    setAuthToken(token);
  }, [token]);

  // Registered once: any 401 from the API (expired/invalid/missing token)
  // logs the user out and drops them back on the auth screen, rather than
  // leaving them stuck on a chat UI that would fail on every request.
  useEffect(() => {
    setOnUnauthorized(() => {
      setToken(null);
      setMessages([]);
      setSessionId(newSessionId());
    });
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  function handleAuthenticated(newToken) {
    setError(null);
    setMessages([]);
    setSessionId(newSessionId());
    setToken(newToken);
  }

  function handleLogout() {
    setToken(null);
    setMessages([]);
    setSessionId(newSessionId());
    setError(null);
  }

  async function handleSend(text) {
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);

    // Most requests finish well under this - it only fires (and swaps the
    // spinner's text) when something's actually taking a while, e.g. a
    // cold Render free-tier instance spinning up.
    slowLoadingTimerRef.current = setTimeout(() => setSlowLoading(true), 5000);

    try {
      const data = await sendChatMessage(sessionId, text);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response, toolCalls: data.tool_calls },
      ]);
    } catch (err) {
      // A 401 already sends the user to the auth screen via
      // setOnUnauthorized (api.js) - no need for a duplicate error banner
      // on a component that's about to be replaced.
      if (err.response?.status !== 401) {
        setError(
          err.code === "ERR_NETWORK" || err.message === "Network Error"
            ? "Couldn't reach the API. Is the FastAPI server running on localhost:8000?"
            : `Request failed: ${err.response?.data?.detail || err.message}`
        );
      }
    } finally {
      clearTimeout(slowLoadingTimerRef.current);
      setSlowLoading(false);
      setLoading(false);
    }
  }

  async function handleNewConversation() {
    setError(null);
    try {
      await clearChatSession(sessionId);
    } catch {
      // best-effort - still start a fresh session locally even if the
      // delete call failed (e.g. server already restarted)
    }
    setSessionId(newSessionId());
    setMessages([]);
  }

  if (!token) {
    return <AuthScreen onAuthenticated={handleAuthenticated} />;
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Churn Analytics Assistant</h1>
          <span className="session-id">session: {sessionId.slice(0, 8)}</span>
        </div>
        <div className="header-actions">
          <button className="new-conversation" onClick={handleNewConversation}>
            New conversation
          </button>
          <button className="log-out" onClick={handleLogout}>
            Log out
          </button>
        </div>
      </header>

      <main className="chat-area" ref={scrollRef}>
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            Ask about a specific customer's churn risk, aggregate churn rates by segment, or
            why a metric/decision was made in this project.
          </div>
        )}

        {messages.map((m, i) => (
          <Message key={i} role={m.role} content={m.content} toolCalls={m.toolCalls} />
        ))}

        {loading && (
          <div className="message message-assistant message-pending">
            <div className="message-role">Assistant</div>
            {slowLoading && (
              <div className="slow-loading-hint">
                Waking up the server (this can take up to a minute on first request)...
              </div>
            )}
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}

        {error && <div className="error-banner">{error}</div>}
      </main>

      <ChatInput onSend={handleSend} disabled={loading} />
    </div>
  );
}
