const form = document.getElementById("separation-form");
const statusBox = document.getElementById("status");
const submitButton = document.getElementById("submit-button");
const resultBox = document.getElementById("result");
const contentId = document.getElementById("content-id");
const targetDetection = document.getElementById("target-detection");
const originalAudio = document.getElementById("original-audio");
const targetAudio = document.getElementById("target-audio");
const residualAudio = document.getElementById("residual-audio");
const originalDownload = document.getElementById("original-download");
const targetDownload = document.getElementById("target-download");
const residualDownload = document.getElementById("residual-download");
const micDescription = document.getElementById("mic-description");
const chunkSeconds = document.getElementById("chunk-seconds");
const startMicButton = document.getElementById("start-mic-button");
const stopMicButton = document.getElementById("stop-mic-button");
const micStatus = document.getElementById("mic-status");
const chunkResults = document.getElementById("chunk-results");

let mediaRecorder = null;
let micStream = null;
let chunkIndex = 0;
let uploadQueue = Promise.resolve();

function setStatus(message) {
  statusBox.textContent = message;
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Separating..." : "Separate Audio";
}

function getDetectionText(detection) {
  if (!detection) {
    return "Detection unavailable";
  }
  return `${detection.label} · RMS ${detection.rms} · Peak ${detection.peak}`;
}

function setDetectionBadge(element, detection) {
  element.textContent = getDetectionText(detection);
  element.classList.toggle("detected", Boolean(detection && detection.detected));
  element.classList.toggle("not-detected", Boolean(detection && !detection.detected));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const audio = formData.get("audio");
  const description = formData.get("description");

  if (!audio || !description) {
    setStatus("Audio file and description are required.");
    return;
  }

  setLoading(true);
  setStatus("Uploading audio and running SAM-Audio...");
  resultBox.classList.add("hidden");

  try {
    const response = await fetch("api/separate", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Failed to separate audio.");
    }

    const cacheKey = `?t=${Date.now()}`;
    originalAudio.src = data.original_url + cacheKey;
    targetAudio.src = data.target_url + cacheKey;
    residualAudio.src = data.residual_url + cacheKey;
    originalDownload.href = data.original_url;
    targetDownload.href = data.target_url;
    residualDownload.href = data.residual_url;
    contentId.textContent = `Content ID: ${data.content_id}`;
    setDetectionBadge(targetDetection, data.detection);

    resultBox.classList.remove("hidden");
    setStatus("Done.");
  } catch (error) {
    setStatus(error.message);
  } finally {
    setLoading(false);
  }
});

function setMicStatus(message) {
  micStatus.textContent = message;
}

function appendChunkResult(index, data) {
  const cacheKey = `?t=${Date.now()}`;
  const card = document.createElement("div");
  card.className = "chunk-card";
  const detectionClass = data.detection && data.detection.detected ? "detected" : "not-detected";
  card.innerHTML = `
    <h3>Chunk ${index}</h3>
    <p>Content ID: ${data.content_id}</p>
    <div class="detection-badge ${detectionClass}">${getDetectionText(data.detection)}</div>
    <div class="player">
      <h3>Original</h3>
      <audio controls src="${data.original_url + cacheKey}"></audio>
      <a href="${data.original_url}" download>Download original</a>
    </div>
    <div class="player">
      <h3>Target</h3>
      <audio controls src="${data.target_url + cacheKey}"></audio>
      <a href="${data.target_url}" download>Download target.wav</a>
    </div>
    <div class="player">
      <h3>Residual</h3>
      <audio controls src="${data.residual_url + cacheKey}"></audio>
      <a href="${data.residual_url}" download>Download residual.wav</a>
    </div>
  `;
  chunkResults.prepend(card);
}

async function uploadMicrophoneChunk(blob, index) {
  const description = micDescription.value.trim();
  if (!description) {
    setMicStatus("Description is required.");
    return;
  }

  const formData = new FormData();
  formData.append("audio", blob, `mic-chunk-${index}.webm`);
  formData.append("description", description);

  setMicStatus(`Processing chunk ${index}...`);
  const response = await fetch("api/separate", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Failed to process chunk ${index}.`);
  }

  appendChunkResult(index, data);
  setMicStatus(`Chunk ${index} done.`);
}

startMicButton.addEventListener("click", async () => {
  const description = micDescription.value.trim();
  if (!description) {
    setMicStatus("Description is required.");
    return;
  }

  const seconds = Number(chunkSeconds.value || 5);
  if (seconds < 3) {
    setMicStatus("Chunk seconds must be at least 3.");
    return;
  }

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(micStream, {
      mimeType: "audio/webm",
    });
    chunkIndex = 0;

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (!event.data || event.data.size === 0) {
        return;
      }

      chunkIndex += 1;
      const currentIndex = chunkIndex;
      uploadQueue = uploadQueue
        .then(() => uploadMicrophoneChunk(event.data, currentIndex))
        .catch((error) => {
          setMicStatus(error.message);
        });
    });

    mediaRecorder.start(seconds * 1000);
    startMicButton.disabled = true;
    stopMicButton.disabled = false;
    setMicStatus(`Recording microphone in ${seconds}s chunks...`);
  } catch (error) {
    setMicStatus(error.message);
  }
});

stopMicButton.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
  }
  startMicButton.disabled = false;
  stopMicButton.disabled = true;
  setMicStatus("Microphone is stopped.");
});
