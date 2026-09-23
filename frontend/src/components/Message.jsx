import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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

export default function Message({ role, content, toolCalls, isError }) {
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
    </div>
  );
}
