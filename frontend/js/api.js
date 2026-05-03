const API_BASE_STORAGE_KEY = "sevt_api_base_url";
const DEFAULT_CLOUD_API_BASE = "https://vehicle-surveillance-backend-production.up.railway.app";

function normalizeBaseUrl(value) {
  if (!value || typeof value !== "string") return "";
  return value.trim().replace(/\/+$/, "");
}

function resolveApiBaseUrl() {
  const params = new URLSearchParams(window.location.search || "");
  const queryApi = normalizeBaseUrl(params.get("api"));
  if (queryApi) {
    try {
      localStorage.setItem(API_BASE_STORAGE_KEY, queryApi);
    } catch (_err) {
      // Ignore storage failures and keep query value for current session.
    }
    return queryApi;
  }

  const runtimeApi = normalizeBaseUrl(window.BACKEND_URL);
  if (runtimeApi) return runtimeApi;

  const isHostedFrontend =
    window.location.hostname.endsWith(".vercel.app") ||
    window.location.hostname.includes("surveillance");
  if (isHostedFrontend) return DEFAULT_CLOUD_API_BASE;

  try {
    const storedApi = normalizeBaseUrl(localStorage.getItem(API_BASE_STORAGE_KEY));
    if (storedApi) return storedApi;
  } catch (_err) {
    // Ignore storage read errors.
  }

  const apiHost = window.location.hostname || "localhost";
  const apiProtocol = window.location.protocol === "https:" ? "https" : "http";
  return `${apiProtocol}://${apiHost}:5000`;
}

const API_BASE_URL = resolveApiBaseUrl();
window.API_BASE_URL = API_BASE_URL;

async function pingHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    const data = await res.json();
    return { ok: res.ok, data };
  } catch (error) {
    return { ok: false, data: { error: String(error) } };
  }
}

async function fetchLiveEvents() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/events/live`);
    const data = await res.json();
    return { ok: res.ok, data };
  } catch (error) {
    return { ok: false, data: { error: String(error) } };
  }
}

async function fetchRecentEvents() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/events/recent`);
    const data = await res.json();
    return { ok: res.ok, data };
  } catch (error) {
    return { ok: false, data: { error: String(error) } };
  }
}

async function fetchStatus() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/status`);
    const data = await res.json();
    return { ok: res.ok, data };
  } catch (error) {
    return { ok: false, data: { error: String(error) } };
  }
}

async function fetchDemoMode() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/demo_mode`);
    const data = await res.json();
    return { ok: res.ok, data };
  } catch (error) {
    return { ok: false, data: { error: String(error) } };
  }
}

async function setDemoMode(mode) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/demo_mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    const data = await res.json();
    return { ok: res.ok, data };
  } catch (error) {
    return { ok: false, data: { error: String(error) } };
  }
}

async function resetDemo() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/reset_demo`, {
      method: "POST",
    });
    const data = await res.json();
    return { ok: res.ok, data };
  } catch (error) {
    return { ok: false, data: { error: String(error) } };
  }
}

async function processFrame(imageDataUrl, source) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/process_frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageDataUrl, source }),
    });
    const data = await res.json();
    return { ok: res.ok, data };
  } catch (error) {
    return { ok: false, data: { error: String(error) } };
  }
}

window.apiClient = {
  pingHealth,
  fetchLiveEvents,
  fetchRecentEvents,
  fetchStatus,
  fetchDemoMode,
  setDemoMode,
  resetDemo,
  processFrame,
};

