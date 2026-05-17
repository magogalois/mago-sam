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
const realtimeTargetBadge = document.getElementById("realtime-target-badge");
const targetChunkList = document.getElementById("target-chunk-list");
const targetChunkEmpty = document.getElementById("target-chunk-empty");

const TARGET_SAMPLE_RATE = 16000;

let audioContext = null;
let micSource = null;
let micProcessor = null;
let micSilence = null;
let micStream = null;
let micChunkTimer = null;
let micBuffers = [];
let chunkIndex = 0;
let uploadQueue = Promise.resolve();
let detectedTargetChunks = [];

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
  return `${detection.label} · RMS ${toDbText(detection.rms)} · Peak ${toDbText(detection.peak)}`;
}

function toDbText(value) {
  const amplitude = Math.max(Number(value) || 0, 0.000001);
  return `${Math.round(20 * Math.log10(amplitude))} dB`;
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

function mergeAudioBuffers(buffers) {
  const sampleCount = buffers.reduce((total, buffer) => total + buffer.length, 0);
  const samples = new Float32Array(sampleCount);
  let offset = 0;

  buffers.forEach((buffer) => {
    samples.set(buffer, offset);
    offset += buffer.length;
  });

  return samples;
}

function resampleMonoAudio(samples, inputSampleRate, outputSampleRate) {
  if (inputSampleRate === outputSampleRate) {
    return samples;
  }

  const ratio = inputSampleRate / outputSampleRate;
  const outputLength = Math.max(1, Math.round(samples.length / ratio));
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i += 1) {
    const sourceIndex = i * ratio;
    const leftIndex = Math.floor(sourceIndex);
    const rightIndex = Math.min(leftIndex + 1, samples.length - 1);
    const weight = sourceIndex - leftIndex;
    output[i] = samples[leftIndex] * (1 - weight) + samples[rightIndex] * weight;
  }

  return output;
}

function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const channelCount = 1;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  function writeString(offset, value) {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channelCount * bytesPerSample, true);
  view.setUint16(32, channelCount * bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += bytesPerSample;
  }

  return new Blob([view], { type: "audio/wav" });
}

function createWavChunkBlob(buffers, inputSampleRate) {
  const mergedSamples = mergeAudioBuffers(buffers);
  const resampledSamples = resampleMonoAudio(
    mergedSamples,
    inputSampleRate,
    TARGET_SAMPLE_RATE,
  );
  return encodeWav(resampledSamples, TARGET_SAMPLE_RATE);
}

function resetRealtimeTargetPanel() {
  detectedTargetChunks = [];
  realtimeTargetBadge.textContent = "Waiting for chunks";
  realtimeTargetBadge.className = "target-state pending";
  targetChunkList.innerHTML = "";
  targetChunkList.appendChild(targetChunkEmpty);
  targetChunkEmpty.classList.remove("hidden");
}

function updateRealtimeTargetPanel(index, detection) {
  if (!detection || !detection.detected) {
    if (detectedTargetChunks.length > 0) {
      realtimeTargetBadge.textContent = `Target confirmed (${detectedTargetChunks.length} chunks)`;
      realtimeTargetBadge.className = "target-state detected";
      return;
    }

    realtimeTargetBadge.textContent = `Chunk ${index}: no target yet`;
    realtimeTargetBadge.className = "target-state not-detected";
    return;
  }

  detectedTargetChunks.push(index);
  realtimeTargetBadge.textContent = `Chunk ${index}: target confirmed`;
  realtimeTargetBadge.className = "target-state detected";
  targetChunkEmpty.classList.add("hidden");

  const chunkBadge = document.createElement("span");
  chunkBadge.className = "target-chunk-badge";
  chunkBadge.textContent = `Chunk ${index}`;
  targetChunkList.appendChild(chunkBadge);
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
    <details class="chunk-audio-details">
      <summary>Show audio playback</summary>
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
    </details>
  `;
  chunkResults.appendChild(card);
}

async function uploadMicrophoneChunk(blob, index) {
  const description = micDescription.value.trim();
  if (!description) {
    setMicStatus("Description is required.");
    return;
  }

  const formData = new FormData();
  formData.append("audio", blob, `mic-chunk-${index}.wav`);
  formData.append("description", description);
  formData.append("already_wav", "true");

  setMicStatus(`Processing chunk ${index}...`);
  const response = await fetch("api/separate", {
    method: "POST",
    body: formData,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `Failed to process chunk ${index}.`);
  }

  updateRealtimeTargetPanel(index, data.detection);
  appendChunkResult(index, data);
  setMicStatus(`Chunk ${index} done.`);
}

function flushMicrophoneChunk() {
  if (!audioContext || micBuffers.length === 0) {
    return;
  }

  const buffers = micBuffers;
  micBuffers = [];
  chunkIndex += 1;
  const currentIndex = chunkIndex;
  const wavBlob = createWavChunkBlob(buffers, audioContext.sampleRate);

  uploadQueue = uploadQueue
    .then(() => uploadMicrophoneChunk(wavBlob, currentIndex))
    .catch((error) => {
      setMicStatus(error.message);
    });
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
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        autoGainControl: false,
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        sampleRate: TARGET_SAMPLE_RATE,
      },
    });
    audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    micSource = audioContext.createMediaStreamSource(micStream);
    micProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    micSilence = audioContext.createGain();
    micSilence.gain.value = 0;
    chunkIndex = 0;
    micBuffers = [];
    uploadQueue = Promise.resolve();
    chunkResults.innerHTML = "";
    resetRealtimeTargetPanel();

    micProcessor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      micBuffers.push(new Float32Array(input));

      const output = event.outputBuffer.getChannelData(0);
      output.fill(0);
    };

    micSource.connect(micProcessor);
    micProcessor.connect(micSilence);
    micSilence.connect(audioContext.destination);
    micChunkTimer = window.setInterval(flushMicrophoneChunk, seconds * 1000);

    startMicButton.disabled = true;
    stopMicButton.disabled = false;
    setMicStatus(`Recording microphone in ${seconds}s WAV chunks...`);
  } catch (error) {
    setMicStatus(error.message);
  }
});

stopMicButton.addEventListener("click", async () => {
  if (micChunkTimer) {
    window.clearInterval(micChunkTimer);
    micChunkTimer = null;
  }
  flushMicrophoneChunk();

  if (micProcessor) {
    micProcessor.disconnect();
    micProcessor.onaudioprocess = null;
    micProcessor = null;
  }
  if (micSource) {
    micSource.disconnect();
    micSource = null;
  }
  if (micSilence) {
    micSilence.disconnect();
    micSilence = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
    micStream = null;
  }
  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }
  startMicButton.disabled = false;
  stopMicButton.disabled = true;
  setMicStatus("Microphone is stopped.");
});
