"use strict";

const state = {
  missionActive: false,
  missionConfigured: false,
  cameraAndTelemetryAllowed: false,
  missionRequestRunning: false,
  telemetryTimer: null,
  cameraTimer: null,
  statusTimer: null,
  telemetryRequestRunning: false,
  cameraRequestRunning: false,
};

const elements = {
  connection: document.querySelector("#connection-state"),
  missionState: document.querySelector("#mission-state"),
  missionDetail: document.querySelector("#mission-detail"),
  start: document.querySelector("#start-mission"),
  refresh: document.querySelector("#refresh-mission"),
  end: document.querySelector("#end-mission"),
  camera: document.querySelector("#front-camera"),
  cameraPlaceholder: document.querySelector("#camera-placeholder"),
  cameraState: document.querySelector("#camera-state"),
  cameraMeta: document.querySelector("#camera-meta"),
  telemetryTime: document.querySelector("#telemetry-time"),
  checkpointSummary: document.querySelector("#checkpoint-summary"),
  checkpointList: document.querySelector("#checkpoint-list"),
  responseLog: document.querySelector("#response-log"),
};

function nowLabel() {
  return new Date().toLocaleTimeString();
}

function appendLog(label, payload, isError = false) {
  const entry = document.createElement("div");
  entry.className = `log-entry ${isError ? "log-error" : "log-success"}`;
  const body = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
  entry.textContent = `[${nowLabel()}] ${label}\n${body}`;
  elements.responseLog.prepend(entry);
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    ...options,
  });
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = { detail: `HTTP ${response.status} returned non-JSON data` };
  }
  if (!response.ok) {
    const error = new Error(payload.detail || `HTTP ${response.status}`);
    error.payload = payload;
    throw error;
  }
  return payload;
}

function setConnection(online) {
  elements.connection.textContent = online ? "SDK online" : "SDK unavailable";
  elements.connection.className = `status ${online ? "status-online" : "status-offline"}`;
}

function setMissionControls(active, configured = state.missionConfigured) {
  state.missionActive = active;
  state.missionConfigured = configured;
  elements.start.disabled = !configured || active || state.missionRequestRunning;
  elements.end.disabled = !configured || !active || state.missionRequestRunning;
  elements.refresh.disabled = state.missionRequestRunning;
  elements.missionState.textContent = configured
    ? (active ? "ACTIVE" : "INACTIVE")
    : "DIRECT BOT MODE";
  elements.missionState.style.color = active || !configured ? "#8cdda9" : "#ffaaa3";
  if (!state.cameraAndTelemetryAllowed) {
    stopMissionPolling();
    elements.camera.style.display = "none";
    elements.cameraPlaceholder.style.display = "grid";
    elements.cameraState.textContent = "Idle";
    elements.cameraState.className = "status status-idle";
  } else {
    startMissionPolling();
  }
}

function renderMissionStatus(status) {
  setConnection(true);
  state.cameraAndTelemetryAllowed = Boolean(status.camera_and_telemetry_allowed);
  setMissionControls(
    Boolean(status.mission_active),
    Boolean(status.mission_configured),
  );
  elements.missionDetail.textContent = status.mission_configured
    ? `${status.checkpoint_count} checkpoints loaded`
    : "No mission tracking; direct camera and telemetry test mode";
  elements.checkpointSummary.textContent = status.mission_configured
    ? `count ${status.checkpoint_count} | latest ${status.latest_scanned_checkpoint ?? "-"}`
    : "Not used in direct bot mode";
}

async function refreshStatus(logResult = false) {
  try {
    const status = await requestJson("/mission-status");
    renderMissionStatus(status);
    if (logResult) {
      appendLog("GET /mission-status", status);
    }
    return status;
  } catch (error) {
    setConnection(false);
    elements.missionDetail.textContent = String(error.message);
    if (logResult) {
      appendLog("GET /mission-status failed", error.payload || error.message, true);
    }
    return null;
  }
}

function normalizeCheckpoints(payload) {
  const value = payload.checkpoints_list ?? payload.checkpoints ?? payload.data ?? [];
  if (Array.isArray(value)) {
    return value;
  }
  return value.checkpoints_list ?? value.checkpoints ?? [];
}

function renderCheckpoints(payload) {
  const checkpoints = normalizeCheckpoints(payload);
  elements.checkpointList.replaceChildren();
  if (!checkpoints.length) {
    const item = document.createElement("li");
    item.textContent = "No checkpoint data.";
    elements.checkpointList.append(item);
    return;
  }
  checkpoints.forEach((checkpoint, index) => {
    const item = document.createElement("li");
    const sequence = checkpoint.sequence ?? checkpoint.order ?? index + 1;
    const latitude = checkpoint.latitude ?? checkpoint.lat ?? "-";
    const longitude = checkpoint.longitude ?? checkpoint.lon ?? "-";
    item.textContent = `#${sequence}  ${latitude}, ${longitude}`;
    elements.checkpointList.append(item);
  });
}

async function refreshMission(logResult = true) {
  const status = await refreshStatus(logResult);
  if (!status?.mission_configured || !status?.mission_active) {
    return;
  }
  try {
    const checkpoints = await requestJson("/checkpoints-list");
    renderCheckpoints(checkpoints);
    if (logResult) {
      appendLog("GET /checkpoints-list", checkpoints);
    }
  } catch (error) {
    appendLog("GET /checkpoints-list failed", error.payload || error.message, true);
  }
}

async function runMissionAction(label, path) {
  if (state.missionRequestRunning) {
    return;
  }
  state.missionRequestRunning = true;
  setMissionControls(state.missionActive, state.missionConfigured);
  try {
    const result = await requestJson(path, { method: "POST" });
    appendLog(`${label} ${path}`, result);
  } catch (error) {
    appendLog(`${label} ${path} failed`, error.payload || error.message, true);
  } finally {
    state.missionRequestRunning = false;
    await refreshMission(false);
  }
}

function valueOrDash(value) {
  return value === null || value === undefined || value === "" ? "-" : String(value);
}

async function pollTelemetry() {
  if (!state.cameraAndTelemetryAllowed || state.telemetryRequestRunning) {
    return;
  }
  state.telemetryRequestRunning = true;
  try {
    const payload = await requestJson("/data");
    const data = payload.data && typeof payload.data === "object" ? payload.data : payload;
    const gps = data.gps && typeof data.gps === "object" ? data.gps : {};
    document.querySelector("#latitude").textContent = valueOrDash(data.latitude ?? gps.latitude ?? gps.lat);
    document.querySelector("#longitude").textContent = valueOrDash(data.longitude ?? gps.longitude ?? gps.lon);
    document.querySelector("#orientation").textContent = valueOrDash(data.orientation ?? data.heading);
    document.querySelector("#speed").textContent = valueOrDash(data.speed);
    document.querySelector("#battery").textContent = valueOrDash(data.battery);
    document.querySelector("#signal").textContent = valueOrDash(data.signal_level ?? data.signal);
    document.querySelector("#gps-signal").textContent = valueOrDash(data.gps_signal ?? gps.signal);
    document.querySelector("#rpms").textContent = valueOrDash(data.rpms ?? data.rpm);
    elements.telemetryTime.textContent = nowLabel();
  } catch (error) {
    elements.telemetryTime.textContent = `Error: ${error.message}`;
  } finally {
    state.telemetryRequestRunning = false;
  }
}

async function pollCamera() {
  if (!state.cameraAndTelemetryAllowed || state.cameraRequestRunning) {
    return;
  }
  state.cameraRequestRunning = true;
  const started = performance.now();
  try {
    const payload = await requestJson("/v2/front");
    const encoded = payload.front_frame ?? payload.image ?? payload.frame;
    if (!encoded) {
      throw new Error("front frame is missing");
    }
    elements.camera.src = `data:image/jpeg;base64,${encoded}`;
    elements.camera.style.display = "block";
    elements.cameraPlaceholder.style.display = "none";
    elements.cameraState.textContent = "Live";
    elements.cameraState.className = "status status-online";
    elements.cameraMeta.textContent =
      `sdk ${valueOrDash(payload.timestamp)} | request ${(performance.now() - started).toFixed(0)} ms`;
  } catch (error) {
    elements.cameraState.textContent = "Frame error";
    elements.cameraState.className = "status status-offline";
    elements.cameraMeta.textContent = error.message;
  } finally {
    state.cameraRequestRunning = false;
  }
}

function startMissionPolling() {
  if (state.telemetryTimer === null) {
    pollTelemetry();
    state.telemetryTimer = window.setInterval(pollTelemetry, 1000);
  }
  if (state.cameraTimer === null) {
    pollCamera();
    state.cameraTimer = window.setInterval(pollCamera, 250);
  }
}

function stopMissionPolling() {
  if (state.telemetryTimer !== null) {
    clearInterval(state.telemetryTimer);
    state.telemetryTimer = null;
  }
  if (state.cameraTimer !== null) {
    clearInterval(state.cameraTimer);
    state.cameraTimer = null;
  }
}

elements.start.addEventListener("click", () => runMissionAction("Start Mission", "/start-mission"));
elements.refresh.addEventListener("click", () => refreshMission(true));
elements.end.addEventListener("click", () => {
  if (window.confirm("End the active mission? This does not send a movement command.")) {
    runMissionAction("End Mission", "/end-mission");
  }
});
document.querySelector("#clear-log").addEventListener("click", () => {
  elements.responseLog.replaceChildren();
});

refreshStatus(true);
state.statusTimer = window.setInterval(() => refreshStatus(false), 2000);
