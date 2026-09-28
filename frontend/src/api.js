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

// Longer than the client default: a cold Render free-tier instance can
// take the better part of a minute just to spin up, and signup/login are
// often the very first request a session makes, before the on-load
// /health poll (below) has necessarily confirmed the server is awake.
const AUTH_TIMEOUT_MS = 90000;

export async function signup(email, password) {
  const { data } = await client.post("/auth/signup", { email, password }, { timeout: AUTH_TIMEOUT_MS });
  return data; // { access_token, token_type }
}

export async function login(email, password) {
  const { data } = await client.post("/auth/login", { email, password }, { timeout: AUTH_TIMEOUT_MS });
  return data; // { access_token, token_type }
}

// Short, deliberately not the client default: used by App.jsx's on-load
// readiness poll, which fires every few seconds - a single hung attempt
// shouldn't be allowed to stall that cadence for anywhere near as long as
// a real request would be allowed to run.
export async function checkHealth() {
  const { data } = await client.get("/health", { timeout: 4000 });
  return data; // { status, models_ready }
}

// Classifies an axios error from a chat/predict-style call into a
// user-facing message and whether the failure is one where the user's
// own input is worth preserving for a retry (see App.jsx's handleSend).
// 401 is deliberately not given a message here - that's handled globally
// by the onUnauthorized flow above, which drops the user back to the
// auth screen instead of showing a banner on a screen that's about to
// disappear.
const WAKING_UP_MESSAGE = "The server is waking up - free-tier hosting can take a minute. Please try again.";

export function classifyChatError(err) {
  const status = err.response?.status;

  if (status === 401) {
    return { message: null, retryableInput: false };
  }
  if (status === 409) {
    return { message: err.response?.data?.detail || "Conflict.", retryableInput: false };
  }
  if (status === 502 || status === 503 || status === 504) {
    // Input is only restored for 502/503, not 504: a gateway timeout can
    // mean the request actually reached the app and is just slow to
    // finish, where silently resending risks a duplicate; 502/503 mean it
    // was bounced before that could happen.
    return { message: WAKING_UP_MESSAGE, retryableInput: status !== 504 };
  }
  if (status >= 500) {
    return { message: `Server error (${status})`, retryableInput: false };
  }
  if (!status) {
    // No HTTP response reached us at all: a client-side timeout
    // (err.code === "ECONNABORTED") or the connection failing outright
    // (err.code === "ERR_NETWORK") - in practice indistinguishable from a
    // 502/503 from the user's point of view, so same message.
    return { message: WAKING_UP_MESSAGE, retryableInput: false };
  }
  return { message: err.response?.data?.detail || err.message, retryableInput: false };
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
