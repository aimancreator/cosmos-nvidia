const form = document.querySelector("#generate-form");
const input = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const preview = document.querySelector("#image-preview");
const processingImage = document.querySelector("#processing-image");
const prompt = document.querySelector("#prompt");
const promptCount = document.querySelector("#prompt-count");
const errorBox = document.querySelector("#form-error");
const generateButton = document.querySelector("#generate-button");
const emptyState = document.querySelector("#empty-state");
const processingState = document.querySelector("#processing-state");
const resultState = document.querySelector("#result-state");
const resultVideo = document.querySelector("#result-video");
const elapsedNode = document.querySelector("#elapsed");
const processingTitle = document.querySelector("#processing-title");
const processingMessage = document.querySelector("#processing-message");
const activityLog = document.querySelector("#activity-log");
const runStatus = document.querySelector("#run-status");

let imageUrl = null;
let videoUrl = null;
let currentJobId = null;
let pollTimer = null;
let elapsedTimer = null;
let workerLogStream = null;
let startedAt = 0;
let processingLogged = false;
let workerLogDisconnected = false;
const seenWorkerLogs = new Set();
const MAX_GENERATION_MS = 15 * 60 * 1000;

function addLog(type, title, detail = "") {
  const entry = document.createElement("div");
  entry.className = `log-entry ${type}`;

  const dot = document.createElement("span");
  dot.className = "log-dot";
  const copy = document.createElement("div");
  const heading = document.createElement("strong");
  heading.textContent = title;
  const description = document.createElement("p");
  description.textContent = detail;
  const timestamp = document.createElement("time");
  timestamp.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  copy.append(heading, description);
  entry.append(dot, copy, timestamp);
  activityLog.append(entry);
  while (activityLog.children.length > 30) activityLog.firstElementChild.remove();
  activityLog.scrollTop = activityLog.scrollHeight;
}

function setRunStatus(status, label) {
  runStatus.className = `run-status ${status}`;
  runStatus.innerHTML = "<i></i>";
  runStatus.append(document.createTextNode(` ${label}`));
}

function stopWorkerLogStream() {
  if (!workerLogStream) return;
  workerLogStream.close();
  workerLogStream = null;
}

function startWorkerLogStream(jobId) {
  stopWorkerLogStream();
  seenWorkerLogs.clear();
  workerLogDisconnected = false;

  const stream = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
  workerLogStream = stream;

  stream.onopen = () => {
    if (workerLogStream !== stream) return;
    addLog("success", "Live worker log connected", `Streaming generation ${jobId.slice(-8)}.`);
  };

  stream.onmessage = (event) => {
    if (workerLogStream !== stream) return;
    try {
      const payload = JSON.parse(event.data);
      const message = String(payload.message || "").trim();
      if (!message || seenWorkerLogs.has(message)) return;
      seenWorkerLogs.add(message);
      if (seenWorkerLogs.size > 250) seenWorkerLogs.delete(seenWorkerLogs.values().next().value);
      addLog(payload.level || "info", "Remote worker", message);
    } catch {
      // Ignore malformed log lines; job result polling remains authoritative.
    }
  };

  stream.onerror = () => {
    if (workerLogStream !== stream) return;
    stopWorkerLogStream();
    if (!workerLogDisconnected && currentJobId === jobId) {
      workerLogDisconnected = true;
      addLog("warning", "Live worker log disconnected", "Generation status is still being monitored.");
    }
  };
}

function reportError(message) {
  showError(message);
  setRunStatus("error", "Error");
  addLog("error", "Generation failed", message);
}

function elapsedLabel() {
  const seconds = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.toggle("visible", Boolean(message));
}

function useImage(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) return reportError("Please choose an image file.");
  if (file.size > 20 * 1024 * 1024) return reportError("Image must be 20 MB or smaller.");
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = URL.createObjectURL(file);
  preview.src = imageUrl;
  processingImage.src = imageUrl;
  dropZone.classList.add("has-image");
  showError("");
  setRunStatus("ready", "Ready");
  addLog("info", "Image selected", `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`);
}

input.addEventListener("change", () => useImage(input.files[0]));
document.querySelector("#change-image").addEventListener("click", (event) => {
  event.preventDefault();
  input.click();
});
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  useImage(file);
});

prompt.addEventListener("input", () => {
  promptCount.textContent = `${prompt.value.length} / 4000`;
});

function setView(name) {
  emptyState.classList.toggle("hidden", name !== "empty");
  processingState.classList.toggle("hidden", name !== "processing");
  resultState.classList.toggle("hidden", name !== "result");
}

function startClock(startTime = Date.now()) {
  startedAt = startTime;
  clearInterval(elapsedTimer);
  elapsedTimer = setInterval(() => {
    const seconds = Math.floor((Date.now() - startedAt) / 1000);
    const minutes = String(Math.floor(seconds / 60)).padStart(2, "0");
    const remainder = String(seconds % 60).padStart(2, "0");
    elapsedNode.textContent = `${minutes}:${remainder}`;
    if (seconds > 180) {
      processingTitle.textContent = "Generating the video frames";
      processingMessage.textContent = "Cosmos is denoising the sequence. Long step times are expected for this 64B model.";
    }
  }, 1000);
}

async function readError(response) {
  try {
    const data = await response.json();
    return data.detail || data.message || "Something went wrong.";
  } catch {
    return `Request failed with status ${response.status}.`;
  }
}

async function stopActiveJob(message, cancelled = false) {
  const jobId = currentJobId;
  clearTimeout(pollTimer);
  clearInterval(elapsedTimer);
  stopWorkerLogStream();
  currentJobId = null;
  localStorage.removeItem("cosmosJobId");
  localStorage.removeItem("cosmosJobStartedAt");
  generateButton.disabled = false;
  setView("empty");

  let terminationError = null;
  if (jobId) {
    try {
      const response = await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
      if (!response.ok) throw new Error(await readError(response));
    } catch (error) {
      terminationError = error;
      addLog("error", "Container stop was not confirmed", error.message);
    }
  }

  if (cancelled && !terminationError) {
    setRunStatus("cancelled", "Cancelled");
    addLog("warning", "Generation cancelled", `Job ${jobId.slice(-8)} and its GPU container were stopped.`);
    return;
  }

  const detail = terminationError
    ? `${message} The GPU stop request was not confirmed: ${terminationError.message}`
    : message;
  reportError(detail);
}

async function pollJob() {
  if (!currentJobId) return;
  if (Date.now() - startedAt >= MAX_GENERATION_MS) {
    await stopActiveJob("Generation exceeded the 15-minute safety limit and was stopped.");
    return;
  }
  try {
    const response = await fetch(`/api/jobs/${currentJobId}`, { cache: "no-store" });
    if (response.status === 202) {
      if (!processingLogged) {
        processingLogged = true;
        addLog("info", "Generation in progress", "The motion engine is preparing and rendering video frames.");
      }
      pollTimer = setTimeout(pollJob, 5000);
      return;
    }
    if (!response.ok) throw new Error(await readError(response));

    const videoBlob = await response.blob();
    const completedJobId = currentJobId;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    videoUrl = URL.createObjectURL(videoBlob);
    resultVideo.src = videoUrl;
    localStorage.removeItem("cosmosJobId");
    localStorage.removeItem("cosmosJobStartedAt");
    currentJobId = null;
    clearInterval(elapsedTimer);
    stopWorkerLogStream();
    setView("result");
    generateButton.disabled = false;
    setRunStatus("success", "Success");
    addLog(
      "success",
      "Video generated successfully",
      `Job ${completedJobId.slice(-8)} · ${(videoBlob.size / 1024 / 1024).toFixed(2)} MB · ${elapsedLabel()}`,
    );
  } catch (error) {
    await stopActiveJob(error.message);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");
  if (!input.files[0]) return reportError("Choose a starting image first.");
  if (prompt.value.trim().length < 8) return reportError("Describe the motion in a little more detail.");

  generateButton.disabled = true;
  processingLogged = false;
  setView("processing");
  setRunStatus("running", "Running");
  addLog("info", "Submitting generation", "Preparing the starting frame and motion direction.");
  processingTitle.textContent = "Initializing the motion engine";
  processingMessage.textContent = "The first render may take longer while the model prepares your scene.";
  startClock();

  try {
    const response = await fetch("/api/generate", { method: "POST", body: new FormData(form) });
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json();
    currentJobId = data.job_id;
    localStorage.setItem("cosmosJobId", currentJobId);
    localStorage.setItem("cosmosJobStartedAt", String(startedAt));
    addLog("success", "Render accepted", `Generation ${currentJobId.slice(-8)} is queued.`);
    startWorkerLogStream(currentJobId);
    pollJob();
  } catch (error) {
    clearInterval(elapsedTimer);
    generateButton.disabled = false;
    setView("empty");
    reportError(error.message);
  }
});

document.querySelector("#cancel-button").addEventListener("click", async () => {
  if (!currentJobId) return;
  await stopActiveJob("Generation cancellation failed.", true);
});

document.querySelector("#download-button").addEventListener("click", () => {
  if (!videoUrl) return;
  const link = document.createElement("a");
  link.href = videoUrl;
  link.download = `cosmos3-${new Date().toISOString().slice(0, 10)}.mp4`;
  link.click();
  addLog("success", "Download started", link.download);
});

document.querySelector("#new-video-button").addEventListener("click", () => {
  generateButton.disabled = false;
  resultVideo.pause();
  setView("empty");
  setRunStatus("ready", "Ready");
  addLog("info", "Ready for another video", "Choose an image and submit a new motion prompt.");
  form.scrollIntoView({ behavior: "smooth", block: "start" });
});

const savedJob = localStorage.getItem("cosmosJobId");
if (savedJob) {
  currentJobId = savedJob;
  processingImage.src = "";
  generateButton.disabled = true;
  processingLogged = false;
  setView("processing");
  setRunStatus("running", "Running");
  addLog("info", "Reconnected to render", `Generation ${currentJobId.slice(-8)} is still active.`);
  processingTitle.textContent = "Reconnecting to your generation";
  processingMessage.textContent = "Your render continued while this page was away.";
  const savedStartedAt = Number(localStorage.getItem("cosmosJobStartedAt"));
  startClock(Number.isFinite(savedStartedAt) && savedStartedAt > 0 ? savedStartedAt : Date.now());
  startWorkerLogStream(currentJobId);
  pollJob();
}
