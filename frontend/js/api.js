const API_BASE_URL =
  window.__API_BASE_URL__ ||
  localStorage.getItem("API_BASE_URL") ||
  "http://localhost:5000";
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

async function processFrame(imageDataUrl) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/process_frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageDataUrl }),
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
  processFrame,
};

