const statusDot = document.getElementById("backend-status");
const healthResult = document.getElementById("health-result");
const pingButton = document.getElementById("ping-backend");
const detectionsList = document.getElementById("detections-list");
const historyList = document.getElementById("history-list");
const videoStream = document.getElementById("video-stream");
const streamPlaceholder = document.getElementById("stream-placeholder");
const captureCanvas = document.createElement("canvas");
let cameraReady = false;

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    videoStream.srcObject = stream;
    await videoStream.play();
    cameraReady = true;
    streamPlaceholder.hidden = true;
    videoStream.classList.remove("load-error");
    videoStream.classList.add("loaded");
  } catch (error) {
    cameraReady = false;
    videoStream.classList.add("load-error");
    streamPlaceholder.hidden = false;
    streamPlaceholder.textContent =
      "Camera access denied/unavailable. Allow camera permission and refresh.";
  }
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
      return `ID #${e.track_id ?? "-"} • ${e.label}${plateStr} • ${
        e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : ""
      }`;
    })
    .join("\n");
  detectionsList.textContent = rows;
}

async function handlePing() {
  const { ok, data } = await window.apiClient.pingHealth();
  statusDot.classList.remove("ok", "error");
  statusDot.classList.add(ok ? "ok" : "error");
  healthResult.textContent = JSON.stringify(data, null, 2);
  if (ok) await startCamera();
}

async function processCurrentFrame() {
  if (!cameraReady || !videoStream.videoWidth || !videoStream.videoHeight) return;
  captureCanvas.width = videoStream.videoWidth;
  captureCanvas.height = videoStream.videoHeight;
  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(videoStream, 0, 0, captureCanvas.width, captureCanvas.height);
  const imageDataUrl = captureCanvas.toDataURL("image/jpeg", 0.9);
  const { ok, data } = await window.apiClient.processFrame(imageDataUrl);
  if (!ok) {
    detectionsList.textContent = "Error processing frame on backend.";
    return;
  }
  renderEvents(data.events || []);
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

pingButton.addEventListener("click", handlePing);

handlePing();
setInterval(processCurrentFrame, 1000);
setInterval(refreshHistory, 3000);

