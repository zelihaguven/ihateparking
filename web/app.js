const canvas = document.getElementById('parking-canvas');
const ctx = canvas.getContext('2d');
const loading = document.getElementById('loading');
const playButton = document.getElementById('play-button');
const stepButton = document.getElementById('step-button');
const resetButton = document.getElementById('reset-button');
const speedSlider = document.getElementById('speed-slider');
const speedValue = document.getElementById('speed-value');
const runStatus = document.getElementById('run-status');
const missionBadge = document.getElementById('mission-badge');
const timelineFill = document.getElementById('timeline-fill');
const timelineThumb = document.getElementById('timeline-thumb');
const timelineTime = document.getElementById('timeline-time');
const timelineProgress = document.getElementById('timeline-progress');

let data = null;
let frame = 0;
let playing = false;
let lastTimestamp = 0;
let animationHandle = null;
let scale = 1;

const colors = { ink: '#13212b', muted: '#829198', grid: '#dbe7e3', road: '#eff4f2', blue: '#246b86', blueDeep: '#174e63', mint: '#56bd91', amber: '#edaa4e', red: '#df705e', obstacle: '#465461' };

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
window.addEventListener('resize', resizeCanvas);

function worldToCanvas(x, y) {
  const [xmin, xmax, ymin, ymax] = data.bounds;
  const padding = 42;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const usableW = width - padding * 2;
  const usableH = height - padding * 2;
  const worldScale = Math.min(usableW / (xmax - xmin), usableH / (ymax - ymin));
  const offsetX = (width - (xmax - xmin) * worldScale) / 2;
  const offsetY = (height - (ymax - ymin) * worldScale) / 2;
  scale = worldScale;
  return [offsetX + (x - xmin) * worldScale, height - offsetY - (y - ymin) * worldScale];
}

function polygonForBox(center, length, width, yaw) {
  const [cx, cy] = center;
  const local = [[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2], [-length / 2, width / 2]];
  return local.map(([x, y]) => [cx + x * Math.cos(yaw) - y * Math.sin(yaw), cy + x * Math.sin(yaw) + y * Math.cos(yaw)]);
}
function vehiclePolygon(pose) {
  const v = data.vehicle;
  const front = v.length - v.rear_overhang;
  const local = [[-v.rear_overhang, -v.width / 2], [front, -v.width / 2], [front, v.width / 2], [-v.rear_overhang, v.width / 2]];
  return local.map(([x, y]) => [pose.x + x * Math.cos(pose.yaw) - y * Math.sin(pose.yaw), pose.y + x * Math.sin(pose.yaw) + y * Math.cos(pose.yaw)]);
}
function drawPolygon(points, fill, stroke, lineWidth = 1, alpha = 1) {
  ctx.save(); ctx.globalAlpha = alpha; ctx.beginPath();
  points.forEach(([x, y], i) => { const p = worldToCanvas(x, y); if (i === 0) ctx.moveTo(...p); else ctx.lineTo(...p); });
  ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke(); ctx.restore();
}
function drawPolyline(points, stroke, width, dash = []) {
  if (!points.length) return;
  ctx.save(); ctx.beginPath(); ctx.setLineDash(dash); ctx.lineWidth = width; ctx.strokeStyle = stroke;
  points.forEach(([x, y], i) => { const p = worldToCanvas(x, y); if (i === 0) ctx.moveTo(...p); else ctx.lineTo(...p); });
  ctx.stroke(); ctx.restore();
}
function drawArrow(pose, color = colors.ink) {
  const start = worldToCanvas(pose.x, pose.y);
  const end = worldToCanvas(pose.x + 0.9 * Math.cos(pose.yaw), pose.y + 0.9 * Math.sin(pose.yaw));
  ctx.save(); ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(...start); ctx.lineTo(...end); ctx.stroke();
  const angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
  ctx.beginPath(); ctx.moveTo(...end); ctx.lineTo(end[0] - 8 * Math.cos(angle - .45), end[1] - 8 * Math.sin(angle - .45)); ctx.lineTo(end[0] - 8 * Math.cos(angle + .45), end[1] - 8 * Math.sin(angle + .45)); ctx.closePath(); ctx.fill(); ctx.restore();
}
function drawGrid() {
  const [xmin, xmax, ymin, ymax] = data.bounds;
  ctx.save(); ctx.strokeStyle = colors.grid; ctx.lineWidth = 1;
  for (let x = Math.ceil(xmin); x <= xmax; x += 2) { const a = worldToCanvas(x, ymin), b = worldToCanvas(x, ymax); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); }
  for (let y = Math.ceil(ymin); y <= ymax; y += 2) { const a = worldToCanvas(xmin, y), b = worldToCanvas(xmax, y); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); }
  ctx.restore();
}
function drawLabel(text, x, y, color = colors.ink, rotate = 0) {
  const p = worldToCanvas(x, y); ctx.save(); ctx.translate(p[0], p[1]); ctx.rotate(-rotate); ctx.fillStyle = color; ctx.font = '800 10px Inter, system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.fillText(text, 0, 0); ctx.restore();
}
function draw() {
  if (!data) return;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height); ctx.fillStyle = colors.road; ctx.fillRect(0, 0, width, height); drawGrid();
  const slot = polygonForBox(data.slot.center, data.slot.length, data.slot.width, data.slot.yaw);
  drawPolygon(slot, '#dff2e9', '#2f956b', 2, .86); drawLabel('TARGET SLOT', data.slot.center[0], data.slot.center[1], '#287653', data.slot.yaw);
  data.obstacles.forEach(o => { drawPolygon(polygonForBox(o.center, o.length, o.width, o.yaw), colors.obstacle, '#26333d', 1.2, 1); drawLabel(o.name.replace('_', ' '), o.center[0], o.center[1], '#fff', o.yaw); });
  drawPolyline(data.planned_path.map(p => [p.x, p.y]), colors.amber, 2, [7, 5]);
  const visible = data.trace.slice(0, Math.max(1, Math.floor(frame) + 1)); drawPolyline(visible.map(p => [p.x, p.y]), colors.blue, 3);
  const pose = data.trace[Math.min(data.trace.length - 1, Math.floor(frame))] || data.start;
  drawPolygon(vehiclePolygon(data.start), colors.amber, colors.ink, 1.2, .78); drawArrow(data.start);
  drawPolygon(vehiclePolygon(pose), data.metrics.controller_success ? colors.mint : colors.red, colors.ink, 1.4, .98); drawArrow(pose);
  updateReadout(pose);
}
function updateReadout(pose) {
  const current = data.trace[Math.min(data.trace.length - 1, Math.floor(frame))] || { t: 0, gear: 0, steering: 0 };
  const progress = data.trace.length <= 1 ? 100 : Math.round((frame / (data.trace.length - 1)) * 100);
  timelineFill.style.width = `${progress}%`; timelineThumb.style.left = `${progress}%`; timelineProgress.textContent = `${progress}%`; timelineTime.textContent = `${(current.t || 0).toFixed(1)} s`;
  document.getElementById('signal-position').textContent = `${pose.x.toFixed(2)}, ${pose.y.toFixed(2)} m`;
  document.getElementById('signal-heading').textContent = `${(pose.yaw * 180 / Math.PI).toFixed(1)}°`;
  document.getElementById('signal-steering').textContent = `${((current.steering || 0) * 180 / Math.PI).toFixed(1)}°`;
  document.getElementById('signal-time').textContent = `${(current.t || 0).toFixed(1)} s`;
  document.getElementById('metric-gear').textContent = `gear ${current.gear > 0 ? 'forward' : current.gear < 0 ? 'reverse' : '—'}`;
}
function setPlaying(next) {
  playing = next;
  playButton.innerHTML = playing ? '<span class="play-icon">Ⅱ</span> Duraklat' : '<span class="play-icon">▶</span> Başlat';
  runStatus.textContent = playing ? 'REPLAY RUNNING' : (frame >= data.trace.length - 1 ? 'RUN COMPLETE' : 'READY TO REPLAY');
  if (playing) { lastTimestamp = performance.now(); animationHandle = requestAnimationFrame(tick); }
  else if (animationHandle) cancelAnimationFrame(animationHandle);
}
function tick(timestamp) {
  if (!playing) return;
  const delta = timestamp - lastTimestamp; lastTimestamp = timestamp;
  const speed = Number(speedSlider.value);
  frame += (delta / 50) * speed;
  if (frame >= data.trace.length - 1) { frame = data.trace.length - 1; setPlaying(false); }
  draw();
  if (playing) animationHandle = requestAnimationFrame(tick);
}
playButton.addEventListener('click', () => setPlaying(!playing));
stepButton.addEventListener('click', () => { setPlaying(false); frame = Math.min(data.trace.length - 1, frame + 5); draw(); });
resetButton.addEventListener('click', () => { setPlaying(false); frame = 0; draw(); });
speedSlider.addEventListener('input', () => { speedValue.textContent = `${speedSlider.value}×`; });

fetch('../artifacts/trace.json').then(response => { if (!response.ok) throw new Error(`trace request failed: ${response.status}`); return response.json(); }).then(payload => {
  data = payload; loading.classList.add('hidden');
  document.getElementById('metric-expanded').textContent = `${payload.metrics.expanded} expansions`;
  document.getElementById('metric-path').textContent = `${payload.metrics.path_length_m.toFixed(2)} m`;
  document.getElementById('metric-collisions').textContent = payload.metrics.collisions;
  missionBadge.textContent = payload.metrics.controller_success ? 'SUCCESS' : 'FAILED';
  missionBadge.className = payload.metrics.controller_success ? 'badge good' : 'badge warn';
  resizeCanvas(); draw();
}).catch(error => { loading.textContent = `Trace yüklenemedi: ${error.message}`; runStatus.textContent = 'TRACE ERROR'; });
