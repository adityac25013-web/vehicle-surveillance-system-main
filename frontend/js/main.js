const statusDot = document.getElementById("backend-status");
const healthResult = document.getElementById("health-result");
const pingButton = document.getElementById("ping-backend");
const resetDemoButton = document.getElementById("reset-demo");
const demoModeSelect = document.getElementById("demo-mode-select");
const chipBackend = document.getElementById("chip-backend");
const chipCamera = document.getElementById("chip-camera");
const chipModel = document.getElementById("chip-model");
const detectionsList = document.getElementById("detections-list");
const historyList = document.getElementById("history-list");
const recentCards = document.getElementById("recent-cards");
const fpsBadge = document.getElementById("fps-badge");
const inferenceBadge = document.getElementById("inference-badge");
const alertBanner = document.getElementById("alert-banner");
const videoStream = document.getElementById("video-stream");
const cameraInput = document.getElementById("camera-input");
const streamPlaceholder = document.getElementById("stream-placeholder");
const captureCanvas = document.createElement("canvas");

let cameraReady = false;
let mode = "normal";
let noDataStreak = 0;
const recentDetectionEvents = [];
const isLikelyMobileClient = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent || "");
const frameSource = isLikelyMobileClient ? "mobile-browser" : "laptop-browser";
let usingBackendFeed = false;
const frameLoopMs = isLikelyMobileClient ? 1100 : 1800;
const maxCaptureWidth = isLikelyMobileClient ? 960 : 640;
const jpegQuality = isLikelyMobileClient ? 0.85 : 0.65;

function showAlert(message) {
  alertBanner.textContent = message;
  alertBanner.hidden = false;
}

function hideAlert() {
  alertBanner.hidden = true;
  alertBanner.textContent = "";
}

function confidenceClass(confidence) {
  if (confidence >= 0.8) return "confidence-high";
  if (confidence >= 0.5) return "confidence-medium";
  return "confidence-low";
}

function setChip(el, label, state) {
  el.textContent = `${label}: ${state}`;
  el.classList.remove("chip-neutral", "chip-ok", "chip-warn", "chip-error");
  if (state === "ok" || state === "ready") {
    el.classList.add("chip-ok");
  } else if (state === "degraded") {
    el.classList.add("chip-warn");
  } else if (state === "error" || state === "unavailable") {
    el.classList.add("chip-error");
  } else {
    el.classList.add("chip-neutral");
  }
}

async function startCamera() {
  const tryConstraints = [
    { video: { facingMode: { ideal: "environment" } }, audio: false },
    { video: true, audio: false },
  ];

  for (const constraints of tryConstraints) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      cameraInput.srcObject = stream;
      await cameraInput.play();
      cameraInput.style.display = "block";
      videoStream.style.display = "none";
      cameraReady = true;
      usingBackendFeed = false;
      streamPlaceholder.hidden = true;
      hideAlert();
      return;
    } catch (_err) {
      // Try next constraint
    }
  }

  try {
    // Keep the original error behavior for logging context.
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    cameraInput.srcObject = stream;
    await cameraInput.play();
    cameraInput.style.display = "block";
    videoStream.style.display = "none";
    cameraReady = true;
    usingBackendFeed = false;
    streamPlaceholder.hidden = true;
    hideAlert();
  } catch (error) {
    cameraReady = false;
    const reason = (error && error.message) ? ` (${error.message})` : "";
    if (isLikelyMobileClient) {
      usingBackendFeed = false;
      videoStream.style.display = "none";
      cameraInput.style.display = "none";
      streamPlaceholder.hidden = false;
      streamPlaceholder.style.display = "flex";
      streamPlaceholder.textContent =
        "Phone camera blocked. Allow camera permission in browser settings and reload this page.";
      showAlert("Mobile camera unavailable." + reason);
      return;
    }
    startVideoFeed();
    showAlert("Browser camera unavailable, switched to backend stream." + reason);
  }
}

function startVideoFeed() {
  const base = window.API_BASE_URL || "http://localhost:5000";
  streamPlaceholder.style.display = "none";
  streamPlaceholder.hidden = true;
  videoStream.src = `${base}/video_feed?t=${Date.now()}`;
  usingBackendFeed = true;
}

videoStream.onload = function () {
  streamPlaceholder.hidden = true;
  videoStream.classList.remove("load-error");
  videoStream.classList.add("loaded");
};

videoStream.onerror = function () {
  streamPlaceholder.hidden = false;
  streamPlaceholder.style.display = "flex";
  streamPlaceholder.textContent =
    "Video stream failed. Ensure backend is running on localhost:5000 and click Ping Backend.";
  videoStream.classList.add("load-error");
  showAlert("Video stream failed. Click Ping Backend and retry.");
};

function pushRecentDetections(events) {
  const nowTs = Date.now();
  for (const e of events) {
    recentDetectionEvents.unshift({
      plate: (e.license_plate || "").trim(),
      label: e.label || "vehicle",
      confidence: Number(e.confidence || 0),
      timestamp: e.timestamp || new Date().toISOString(),
      id: `${e.track_id ?? "0"}-${e.timestamp ?? nowTs}`,
    });
  }
  const dedup = [];
  const seen = new Set();
  for (const item of recentDetectionEvents) {
    if (seen.has(item.id)) continue;
    seen.add(item.id);
    dedup.push(item);
    if (dedup.length >= 5) break;
  }
  recentDetectionEvents.length = 0;
  recentDetectionEvents.push(...dedup);
}

function renderRecentCards() {
  if (!recentDetectionEvents.length) {
    recentCards.textContent = "No recent detections yet.";
    return;
  }
  recentCards.innerHTML = recentDetectionEvents
    .map((e) => {
      const platePart = e.plate ? `<div class="plate">${e.plate}</div>` : "<div>-</div>";
      const confCls = confidenceClass(e.confidence);
      const confPct = `${Math.round((e.confidence || 0) * 100)}%`;
      const timeText = e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : "--";
      return `<div class="recent-card">
        <div>${e.label}</div>
        ${platePart}
        <div class="${confCls}">Confidence: ${confPct}</div>
        <div>${timeText}</div>
      </div>`;
    })
    .join("");
}

function renderEvents(events) {
  if (!events || events.length === 0) {
    detectionsList.textContent =
      "No vehicles detected. Point the camera at cars or a screen showing cars.";
    return;
  }
  const rows = events
    .map((e) => {
      const plate = (e.license_plate || "").trim();
      const plateStr = plate ? ` • Plate: ${plate}` : "";
      const confPct = `${Math.round((Number(e.confidence || 0) || 0) * 100)}%`;
      return `ID #${e.track_id ?? "-"} • ${e.label}${plateStr} • Confidence: ${confPct} • ${
        e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : ""
      }`;
    })
    .join("\n");
  detectionsList.textContent = rows;
}

function updateMetrics(stats, fallbackProcessingMs) {
  const fps = typeof stats?.fps === "number" ? stats.fps : 0;
  const detectionMs =
    typeof stats?.detection_ms === "number"
      ? stats.detection_ms
      : typeof fallbackProcessingMs === "number"
      ? fallbackProcessingMs
      : 0;
  fpsBadge.textContent = `FPS: ${fps > 0 ? fps.toFixed(1) : "--"}`;
  inferenceBadge.textContent = `Inference: ${detectionMs > 0 ? detectionMs.toFixed(1) : "--"} ms`;
}

async function handlePing() {
  const { ok, data } = await window.apiClient.pingHealth();
  statusDot.classList.remove("ok", "error");
  statusDot.classList.add(ok ? "ok" : "error");
  healthResult.textContent = JSON.stringify(data, null, 2);
  if (ok) {
    await startCamera();
    await refreshStatus();
  }
}

async function refreshStatus() {
  const { ok, data } = await window.apiClient.fetchStatus();
  if (!ok) {
    setChip(chipBackend, "Backend", "error");
    setChip(chipCamera, "Camera", "error");
    setChip(chipModel, "Model", "error");
    return;
  }
  setChip(chipBackend, "Backend", data.backend || "ok");
  setChip(chipCamera, "Camera", data.camera || "--");
  setChip(chipModel, "Model", data.model || "--");
  if (data.mode && data.mode !== mode) {
    mode = data.mode;
    demoModeSelect.value = mode;
  }
  updateMetrics(data.stats || {}, undefined);
}

async function refreshDetections() {
  // In browser-camera mode, render detections from process_frame only.
  // This avoids mixing backend webcam detections with this client stream.
  if (cameraReady) return;
  if (!usingBackendFeed) return;
  const { ok, data } = await window.apiClient.fetchLiveEvents();
  if (!ok) {
    detectionsList.textContent = "Error fetching detections.";
    return;
  }
  const events = data.events || [];
  renderEvents(events);
  if (events.length > 0) {
    pushRecentDetections(events);
    renderRecentCards();
    noDataStreak = 0;
    hideAlert();
  } else {
    noDataStreak += 1;
    if (noDataStreak >= 10) {
      showAlert("No detections for ~10 seconds. Auto-retrying stream...");
      startVideoFeed();
      noDataStreak = 0;
    }
  }
  updateMetrics(data.stats || {}, undefined);
}

async function processCurrentFrame() {
  if (!cameraReady || !cameraInput.videoWidth || !cameraInput.videoHeight) return;
  const scale = Math.min(1, maxCaptureWidth / cameraInput.videoWidth);
  captureCanvas.width = Math.max(320, Math.floor(cameraInput.videoWidth * scale));
  captureCanvas.height = Math.max(180, Math.floor(cameraInput.videoHeight * scale));
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(cameraInput, 0, 0, captureCanvas.width, captureCanvas.height);
  const imageDataUrl = captureCanvas.toDataURL("image/jpeg", jpegQuality);
  const { ok, data } = await window.apiClient.processFrame(imageDataUrl, frameSource);
  if (!ok) return;
  const events = data.events || [];
  if (events.length > 0) {
    pushRecentDetections(events);
    renderRecentCards();
  }
  renderEvents(events);
  updateMetrics({}, data?.stats?.processing_ms);
}

async function refreshHistory() {
  const { ok, data } = await window.apiClient.fetchRecentEvents();
  if (!ok) {
    historyList.textContent = "Error fetching history.";
    return;
  }

  const events = data.events || [];
  const seen = new Map();

  for (const e of events) {
    const plate = (e.license_plate || "").trim();
    if (!plate) continue;

    const key = plate;
    const existing = seen.get(key);
    const ts = e.timestamp ? new Date(e.timestamp) : null;
    if (!existing) {
      seen.set(key, {
        plate,
        firstSeen: ts,
        count: 1,
      });
    } else {
      existing.count += 1;
      if (ts && existing.firstSeen && ts < existing.firstSeen) {
        existing.firstSeen = ts;
      }
    }
  }

  const items = Array.from(seen.values());
  if (items.length === 0) {
    historyList.textContent = "No plates captured yet.";
    return;
  }

  const lines = items
    .sort((a, b) => {
      if (a.firstSeen && b.firstSeen) return a.firstSeen - b.firstSeen;
      return 0;
    })
    .map((item) => {
      const timeStr = item.firstSeen ? item.firstSeen.toLocaleTimeString() : "";
      const countStr = item.count > 1 ? ` • Seen ${item.count} times` : "";
      return `Plate: ${item.plate} • First seen: ${timeStr}${countStr}`;
    })
    .join("\n");

  historyList.textContent = lines;
}

async function syncDemoMode() {
  const { ok, data } = await window.apiClient.fetchDemoMode();
  if (!ok) return;
  mode = data.mode || "normal";
  if (isLikelyMobileClient && mode === "normal") {
    const switched = await window.apiClient.setDemoMode("accurate");
    if (switched.ok) mode = "accurate";
  }
  demoModeSelect.value = mode;
}

async function onDemoModeChange() {
  const nextMode = demoModeSelect.value;
  const { ok } = await window.apiClient.setDemoMode(nextMode);
  if (!ok) {
    showAlert("Unable to change demo mode. Please retry.");
    demoModeSelect.value = mode;
    return;
  }
  mode = nextMode;
  showAlert(`Switched to ${mode} mode.`);
  setTimeout(() => hideAlert(), 1800);
  await refreshStatus();
}

async function onResetDemo() {
  const { ok } = await window.apiClient.resetDemo();
  if (!ok) {
    showAlert("Reset failed. Please retry.");
    return;
  }
  recentDetectionEvents.length = 0;
  renderRecentCards();
  detectionsList.textContent = "Waiting for fresh detections...";
  showAlert("Demo state reset successfully.");
  setTimeout(() => hideAlert(), 1500);
}

pingButton.addEventListener("click", handlePing);
resetDemoButton.addEventListener("click", onResetDemo);
demoModeSelect.addEventListener("change", onDemoModeChange);

async function boot() {
  await handlePing();
  await syncDemoMode();
  await refreshStatus();
  refreshDetections();
  refreshHistory();
  showAlert("Startup self-check complete. System ready for demo.");
  setTimeout(() => hideAlert(), 1800);
}

boot();
setInterval(processCurrentFrame, frameLoopMs);
setInterval(refreshDetections, 1000);
setInterval(refreshHistory, 3000);
setInterval(refreshStatus, 2500);

