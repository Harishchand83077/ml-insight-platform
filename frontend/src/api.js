import axios from "axios";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  timeout: 60000,
});

export async function sendChatMessage(sessionId, message) {
  // Longer than the client default: a cold Render free-tier instance can
  // take the better part of a minute just to spin up, on top of the
  // LLM + tool-call round trip itself.
  const { data } = await client.post(
    "/chat",
    { session_id: sessionId, message },
    { timeout: 120000 }
  );
  return data; // { session_id, response, tool_calls }
}

export async function clearChatSession(sessionId) {
  const { data } = await client.delete(`/chat/${sessionId}`);
  return data; // { session_id, cleared }
}
