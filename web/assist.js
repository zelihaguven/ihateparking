const video = document.getElementById('source-video');
const playButton = document.getElementById('analyze-button');
const resetButton = document.getElementById('reset-video-button');
const speedSlider = document.getElementById('assist-speed-slider');
const speedValue = document.getElementById('assist-speed-value');
const voiceButton = document.getElementById('voice-button');
const boxesCanvas = document.getElementById('vision-boxes');
const boxesContext = boxesCanvas.getContext('2d');
const instructionCard = document.getElementById('instruction-card');
const instruction = document.getElementById('instruction');
const instructionReason = document.getElementById('instruction-reason');
let payload = null;
let voiceEnabled = false;
let lastSpokenInstruction = '';

function signalForTime(time) {
  if (!payload?.signals?.length) return null;
  let current = payload.signals[0];
  for (const signal of payload.signals) {
    if (signal.t <= time) current = signal;
    else break;
  }
  return current;
}
function renderSignal() {
  const signal = signalForTime(video.currentTime);
  if (!signal) return;
  const isStop = signal.instruction.startsWith('STOP');
  const isTurn = signal.instruction.includes('STEER');
  instruction.textContent = signal.instruction;
  instructionReason.textContent = signal.reason;
  instructionCard.className = `instruction-card ${isStop ? 'stop' : isTurn ? 'turn' : ''}`;
  drawBoundaryBoxes(signal);
  if (voiceEnabled && signal.instruction !== lastSpokenInstruction && 'speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(signal.instruction));
  }
  lastSpokenInstruction = signal.instruction;
  document.getElementById('frame-counter').textContent = `frame ${signal.frame}`;
  document.getElementById('boundary-readout').textContent = `${signal.boundary_count} boundary response${signal.boundary_count === 1 ? '' : 's'}`;
  document.getElementById('confidence-readout').textContent = `${Math.round(signal.confidence * 100)}% confidence`;
  document.getElementById('assist-time').textContent = `${video.currentTime.toFixed(1)} s`;
  document.getElementById('boundary-count').textContent = signal.boundary_count;
  document.getElementById('boundary-center').textContent = signal.boundary_centroid_x == null ? 'not available' : `${Math.round(signal.boundary_centroid_x * 100)}% of frame`;
}
function drawBoundaryBoxes(signal) {
  const width = video.clientWidth;
  const height = video.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  boxesCanvas.width = Math.max(1, Math.round(width * dpr));
  boxesCanvas.height = Math.max(1, Math.round(height * dpr));
  boxesContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  boxesContext.clearRect(0, 0, width, height);
  boxesContext.strokeStyle = '#edaa4e';
  boxesContext.fillStyle = '#edaa4e';
  boxesContext.lineWidth = 2;
  for (const box of signal.boundaries || []) {
    const x = box.x * width;
    const y = box.y * height;
    const w = box.w * width;
    const h = box.h * height;
    boxesContext.strokeRect(x, y, w, h);
    boxesContext.font = '800 10px Inter, system-ui, sans-serif';
    boxesContext.fillText('BOUNDARY', x + 3, Math.max(12, y - 4));
  }
}
function syncPlayback() { renderSignal(); if (!video.paused) requestAnimationFrame(syncPlayback); }
video.addEventListener('timeupdate', renderSignal);
video.addEventListener('play', () => { playButton.innerHTML = 'Ⅱ Pause assist'; requestAnimationFrame(syncPlayback); });
video.addEventListener('pause', () => { playButton.innerHTML = '▶ Play assist'; renderSignal(); });
video.addEventListener('ended', () => { playButton.innerHTML = '▶ Replay assist'; renderSignal(); });
playButton.addEventListener('click', () => { if (video.paused) video.play(); else video.pause(); });
resetButton.addEventListener('click', () => { video.pause(); video.currentTime = 0; playButton.innerHTML = '▶ Play assist'; renderSignal(); });
voiceButton.addEventListener('click', () => {
  voiceEnabled = !voiceEnabled;
  voiceButton.textContent = voiceEnabled ? 'Voice cues on' : 'Enable voice';
  if (!voiceEnabled && 'speechSynthesis' in window) window.speechSynthesis.cancel();
});
speedSlider.addEventListener('input', () => { video.playbackRate = Number(speedSlider.value); speedValue.textContent = `${speedSlider.value}×`; });
fetch('../artifacts/footage_assist.json').then(response => { if (!response.ok) throw new Error(`signal request failed: ${response.status}`); return response.json(); }).then(json => { payload = json; renderSignal(); }).catch(error => { instruction.textContent = 'SIGNAL UNAVAILABLE'; instructionReason.textContent = error.message; instructionCard.className = 'instruction-card stop'; });
