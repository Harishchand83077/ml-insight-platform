import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { sendFeedback } from "../api.js";

function ToolCalls({ toolCalls }) {
  if (!toolCalls || toolCalls.length === 0) return null;

  return (
    <details className="tool-calls">
      <summary>
        <span aria-hidden="true">🔧</span> Tools used ({toolCalls.length})
      </summary>
      <div className="tool-calls-list">
        {toolCalls.map((call, i) => (
          <div className="tool-call" key={i}>
            <span className="tool-call-name">{call.tool}</span>
            <pre className="tool-call-args">{JSON.stringify(call.args, null, 2)}</pre>
          </div>
        ))}
      </div>
    </details>
  );
}

function FeedbackButtons({ sessionId, content }) {
  const [rating, setRating] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function vote(value) {
    if (rating !== null || submitting) return;
    setSubmitting(true);
    try {
      await sendFeedback(sessionId, content, value);
      setRating(value);
    } catch {
      // best-effort - leave both buttons enabled so the user can retry
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="feedback-row">
      <button
        type="button"
        className={`feedback-btn${rating === "up" ? " feedback-btn-up-selected" : ""}`}
        onClick={() => vote("up")}
        disabled={rating !== null || submitting}
        aria-label="Good response"
        aria-pressed={rating === "up"}
      >
        👍
      </button>
      <button
        type="button"
        className={`feedback-btn${rating === "down" ? " feedback-btn-down-selected" : ""}`}
        onClick={() => vote("down")}
        disabled={rating !== null || submitting}
        aria-label="Bad response"
        aria-pressed={rating === "down"}
      >
        👎
      </button>
    </div>
  );
}

export default function Message({ role, content, toolCalls, isError, sessionId }) {
  const roleLabel = role === "user" ? "You" : "Assistant";

  return (
    <div className={`message message-${role}${isError ? " message-error" : ""}`}>
      <div className="message-role">{roleLabel}</div>
      {role === "assistant" && <ToolCalls toolCalls={toolCalls} />}
      <div className="message-content">
        {role === "assistant" ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        ) : (
          content
        )}
      </div>
      {role === "assistant" && <FeedbackButtons sessionId={sessionId} content={content} />}
    </div>
  );
}
