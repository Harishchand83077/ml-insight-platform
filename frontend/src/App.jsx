import { useEffect, useRef, useState } from "react";
import Message from "./components/Message.jsx";
import ChatInput from "./components/ChatInput.jsx";
import AuthScreen from "./components/AuthScreen.jsx";
import {
  sendChatMessage,
  clearChatSession,
  setAuthToken,
  setOnUnauthorized,
  checkHealth,
  classifyChatError,
} from "./api.js";
import "./App.css";

const HEALTH_POLL_INTERVAL_MS = 3000;
const HEALTH_POLL_MAX_WAIT_MS = 120000;

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
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [slowLoading, setSlowLoading] = useState(false);
  const [error, setError] = useState(null);
  // True from mount until GET /health reports models_ready, or ~2 minutes
  // pass without that happening (see the effect below) - covers both a
  // cold Render instance that hasn't started responding at all yet, and
  // one that's up but still loading the XGBoost/embedding models.
  const [serverStarting, setServerStarting] = useState(true);
  const scrollRef = useRef(null);
  const slowLoadingTimerRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    const startedAt = Date.now();

    async function poll() {
      if (cancelled) return;
      try {
        const data = await checkHealth();
        if (data.status === "ok" && data.models_ready) {
          if (!cancelled) setServerStarting(false);
          return;
        }
      } catch {
        // 502/503/504, a timeout, or a network error here all mean the
        // same thing for this purpose: still starting, not a failure -
        // keep polling rather than giving up.
      }
      if (cancelled) return;
      if (Date.now() - startedAt >= HEALTH_POLL_MAX_WAIT_MS) {
        // Stop waiting rather than blocking Send forever if this check is
        // ever wrong - an actual send attempt still surfaces the same
        // "waking up" message via handleSend/classifyChatError, so giving
        // up here just hands the user back the normal retry path instead
        // of leaving them stuck.
        setServerStarting(false);
        return;
      }
      setTimeout(poll, HEALTH_POLL_INTERVAL_MS);
    }

    poll();
    return () => {
      cancelled = true;
    };
  }, []);

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
    setInputValue("");
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
      // setOnUnauthorized (api.js) - classifyChatError returns no message
      // for it, so no duplicate banner shows on a screen about to be
      // replaced.
      const { message, retryableInput } = classifyChatError(err);
      if (message) {
        setError(message);
      }
      if (retryableInput) {
        // The message never actually went through - drop the optimistic
        // bubble added above (so resending doesn't leave a duplicate in
        // the transcript) and hand the text back to the input.
        setMessages((prev) => prev.slice(0, -1));
        setInputValue(text);
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

  const serverStatusBanner = serverStarting && (
    <div className="server-status-banner">Server is starting up...</div>
  );

  if (!token) {
    return (
      <>
        {serverStatusBanner}
        <AuthScreen onAuthenticated={handleAuthenticated} />
      </>
    );
  }

  return (
    <>
      {serverStatusBanner}
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
            <Message
              key={i}
              role={m.role}
              content={m.content}
              toolCalls={m.toolCalls}
              sessionId={sessionId}
            />
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

        <ChatInput
          value={inputValue}
          onChange={setInputValue}
          onSend={handleSend}
          disabled={loading}
          sendDisabled={serverStarting}
        />
      </div>
    </>
  );
}
