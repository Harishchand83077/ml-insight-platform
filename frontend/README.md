# Churn Analytics Assistant - frontend

A minimal chat UI for the LangChain churn-analytics agent, built with
React + Vite and plain CSS (no UI framework). Talks to the FastAPI
`/chat` and `/chat/{session_id}` endpoints at `http://localhost:8000`
(hardcoded in `src/api.js`).

## Running locally

```bash
npm install
npm run dev
```

Opens at `http://localhost:5173`.

**Prerequisite**: the backend API must already be running
(`uvicorn src.serving.api:app --reload --port 8000` from the project
root) - the chat interface calls it directly and will show an error
message in the UI if it can't be reached.

## What's here

- `src/App.jsx` - top-level state: message list, session_id
  (`crypto.randomUUID()`, generated on load and reset by "New
  conversation"), loading/error state
- `src/components/Message.jsx` - renders one message; assistant messages
  render as Markdown (tables, bold, lists - the agent's responses use
  these), with a collapsible "🔧 Tools used" section when `tool_calls` is
  non-empty
- `src/components/ChatInput.jsx` - the input box + send button, disabled
  while a request is in flight
- `src/api.js` - the two API calls (`POST /chat`, `DELETE
  /chat/{session_id}`), axios instance pointed at `localhost:8000`
