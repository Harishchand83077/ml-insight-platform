import axios from "axios";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  timeout: 60000,
});

// The JWT lives in React state (App.jsx), deliberately never localStorage -
// see App.jsx for why. This module-level variable is just a mirror the
// request interceptor below can read without every call site having to
// pass the token through explicitly; setAuthToken is the only way it's
// ever written, always called from App.jsx whenever its own token state
// changes.
let authToken = null;

export function setAuthToken(token) {
  authToken = token;
}

client.interceptors.request.use((config) => {
  if (authToken) {
    config.headers.Authorization = `Bearer ${authToken}`;
  }
  return config;
});

// Registered once by App.jsx and called on any 401 (expired/invalid/missing
// token) - centralizes "log the user out" here instead of every protected
// call site needing its own 401 handling.
let onUnauthorized = null;

export function setOnUnauthorized(handler) {
  onUnauthorized = handler;
}

client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && onUnauthorized) {
      onUnauthorized();
    }
    return Promise.reject(error);
  }
);

export async function signup(email, password) {
  const { data } = await client.post("/auth/signup", { email, password });
  return data; // { access_token, token_type }
}

export async function login(email, password) {
  const { data } = await client.post("/auth/login", { email, password });
  return data; // { access_token, token_type }
}

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

export async function sendFeedback(sessionId, messageContent, rating) {
  const { data } = await client.post("/feedback", {
    session_id: sessionId,
    message_content: messageContent,
    rating,
  });
  return data; // { status: "recorded" }
}
