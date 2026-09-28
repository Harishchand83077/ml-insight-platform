// Controlled by App.jsx (value/onChange) rather than owning its own text
// state - a failed send needs to be able to restore the user's text after
// the fact (see App.jsx's handleSend), which isn't possible if this
// component clears itself the moment onSend is called.
export default function ChatInput({ value, onChange, onSend, disabled, sendDisabled }) {
  const blocked = disabled || sendDisabled;

  function handleSubmit(e) {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || blocked) return;
    onSend(trimmed);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  }

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about a customer, a churn rate, or how a metric was chosen..."
        disabled={disabled}
        rows={1}
      />
      <button type="submit" disabled={blocked || !value.trim()}>
        {disabled ? "Sending..." : "Send"}
      </button>
    </form>
  );
}
