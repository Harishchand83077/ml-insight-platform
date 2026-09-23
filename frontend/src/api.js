import axios from "axios";

const client = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 60000,
});

export async function sendChatMessage(sessionId, message) {
  const { data } = await client.post("/chat", {
    session_id: sessionId,
    message,
  });
  return data; // { session_id, response, tool_calls }
}

export async function clearChatSession(sessionId) {
  const { data } = await client.delete(`/chat/${sessionId}`);
  return data; // { session_id, cleared }
}
