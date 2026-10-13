const $ = (id) => document.getElementById(id);
let state = null;
let running = false;
let timer = null;

const canvas = $("world-canvas");
const ctx = canvas.getContext("2d");

async function request(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function setConnection(live, label = live ? "LIVE / FASTAPI" : "OFFLINE") {
  $("connection-dot").classList.toggle("live", live);
  $("connection-label").textContent = label;
}

function fmt(value, digits = 2, suffix = "") {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(digits)}${suffix}` : "—";
}

function posePoint(pose, bounds, width, height, margin = 32) {
  const [xmin, xmax, ymin, ymax] = bounds;
  return {
    x: margin + (pose.x - xmin) / (xmax - xmin) * (width - margin * 2),
    y: height - margin - (pose.y - ymin) / (ymax - ymin) * (height - margin * 2),
  };
}

function drawPolygon(points, fill, stroke, lineWidth = 1) {
  if (!points.length) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  points.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke(); }
}

function boxPoints(center, length, width, yaw, bounds, cw, ch) {
  const corners = [[-length / 2, -width / 2], [length / 2, -width / 2], [length / 2, width / 2], [-length / 2, width / 2]];
  return corners.map(([x, y]) => posePoint({ x: center[0] + x * Math.cos(yaw) - y * Math.sin(yaw), y: center[1] + x * Math.sin(yaw) + y * Math.cos(yaw) }, bounds, cw, ch));
}

function drawWorld(data) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(480, rect.width);
  const height = Math.max(380, rect.height);
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const { bounds } = data.scene;
  ctx.fillStyle = "#061013";
  ctx.fillRect(0, 0, width, height);

  // Grid and world frame.
  ctx.strokeStyle = "#173038";
  ctx.lineWidth = 1;
  for (let x = Math.ceil(bounds[0]); x <= bounds[1]; x += 1) {
    const p = posePoint({ x, y: bounds[2] }, bounds, width, height);
    ctx.beginPath(); ctx.moveTo(p.x, 20); ctx.lineTo(p.x, height - 25); ctx.stroke();
  }
  for (let y = Math.ceil(bounds[2]); y <= bounds[3]; y += 1) {
    const p = posePoint({ x: bounds[0], y }, bounds, width, height);
    ctx.beginPath(); ctx.moveTo(25, p.y); ctx.lineTo(width - 25, p.y); ctx.stroke();
  }

  // Planned path.
  if (data.plan?.poses?.length) {
    ctx.beginPath();
    data.plan.poses.forEach((pose, index) => { const p = posePoint(pose, bounds, width, height); if (index === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y); });
    ctx.strokeStyle = "#9ce6bf"; ctx.lineWidth = 2; ctx.setLineDash([7, 6]); ctx.stroke(); ctx.setLineDash([]);
  }

  data.scene.static_obstacles.forEach((obstacle) => {
    drawPolygon(boxPoints(obstacle.center, obstacle.length, obstacle.width, obstacle.yaw, bounds, width, height), "#34454b", "#64787d", 1);
  });

  const slot = data.scene.slot;
  drawPolygon(boxPoints(slot.center, slot.length, slot.width, slot.yaw, bounds, width, height), null, "#f3bf67", 2);
  const dynamicPoses = data.snapshot.dynamic_actor_poses || {};
  data.scene.dynamic_actors.forEach((actor) => {
    const pose = dynamicPoses[actor.actor_id] || actor.start_pose;
    const size = actor.kind === "vehicle" ? [4.5, 2.0] : [.7, .7];
    drawPolygon(boxPoints([pose.x, pose.y], size[0], size[1], pose.yaw, bounds, width, height), "#7f383b", "#ff786d", 2);
  });

  // Neural detections as rings around the estimated boxes.
  (data.snapshot.perception?.obstacles || []).forEach((obstacle) => {
    const color = obstacle.dynamic ? "#ff786d" : "#75a9ff";
    const points = boxPoints([obstacle.pose.x, obstacle.pose.y], obstacle.length_m, obstacle.width_m, obstacle.pose.yaw, bounds, width, height);
    drawPolygon(points, null, color, 1);
  });
  (data.snapshot.perception?.slots || []).forEach((slotEstimate) => {
    const p = posePoint(slotEstimate.pose, bounds, width, height);
    ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2); ctx.strokeStyle = "#f3bf67"; ctx.lineWidth = 2; ctx.stroke();
  });

  const vehicle = data.snapshot.state.pose;
  const ego = boxPoints([vehicle.x, vehicle.y], 4.4, 1.9, vehicle.yaw, bounds, width, height);
  drawPolygon(ego, "#22616b", "#72d8dc", 2);
  const direction = posePoint({ x: vehicle.x + Math.cos(vehicle.yaw) * 1.5, y: vehicle.y + Math.sin(vehicle.yaw) * 1.5 }, bounds, width, height);
  const origin = posePoint(vehicle, bounds, width, height);
  ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(direction.x, direction.y); ctx.strokeStyle = "#e9f1ed"; ctx.lineWidth = 2; ctx.stroke();
}

function renderEvents(events) {
  $("event-log").innerHTML = events.slice().reverse().map((event) => `<div class="event ${event.level}"><span class="event-time">t=${fmt(event.time_s, 2)}s</span><span class="event-level">${event.level}</span><span>${event.message}</span></div>`).join("");
}

function render(data) {
  state = data;
  const snapshot = data.snapshot;
  const safety = data.safety;
  const perception = snapshot.perception || {};
  const obstacles = perception.obstacles || [];
  const dynamicCount = obstacles.filter((item) => item.dynamic).length;
  $("frame-id").textContent = perception.frame_id || "—";
  $("frame-number").textContent = `frame ${snapshot.frame}`;
  $("model-name").textContent = perception.model_name || "—";
  $("slot-count").textContent = perception.slots?.length ?? 0;
  $("obstacle-count").textContent = obstacles.length;
  $("dynamic-count-readout").textContent = dynamicCount;
  $("ood-score").textContent = fmt(perception.out_of_distribution_score, 2);
  $("gear").textContent = String(snapshot.state.gear).toUpperCase();
  $("position").textContent = `${fmt(snapshot.state.pose.x)} / ${fmt(snapshot.state.pose.y)}`;
  $("speed").textContent = `${fmt(Math.abs(snapshot.state.speed_mps), 2)} m/s`;
  $("steering").textContent = `${fmt(snapshot.state.steering_angle_rad, 2)} rad`;
  $("collisions").textContent = snapshot.state.collisions;
  $("replans").textContent = data.replan_count;
  $("clearance").textContent = fmt(safety.minimum_clearance_m) + " m";
  $("braking-distance").textContent = fmt(safety.braking_distance_m) + " m";
  $("ttc").textContent = safety.time_to_collision_s == null ? "—" : `${fmt(safety.time_to_collision_s)} s`;
  $("safety-action").textContent = safety.action.toUpperCase();
  $("safety-reason").textContent = safety.reason;
  $("safety-icon").textContent = safety.action === "brake" ? "!" : safety.action === "replan" ? "↻" : "✓";
  $("mode-badge").textContent = data.autonomy_mode.toUpperCase();
  $("mode-badge").className = `badge ${safety.action === "brake" ? "brake" : safety.action === "replan" ? "replan" : "neutral"}`;
  document.querySelector(".safety-card").className = `panel safety-card ${safety.action}`;
  $("last-time").textContent = `t=${fmt(snapshot.timestamp_s)}s`;
  drawWorld(data);
  renderEvents(data.events || []);
}

async function refresh() {
  try { render(await request("/api/state")); setConnection(true); }
  catch (error) { setConnection(false, "API OFFLINE"); console.error(error); }
}

async function tick() {
  try { render(await request("/api/tick", { method: "POST", body: JSON.stringify({ dt: .10 }) })); setConnection(true); }
  catch (error) { setConnection(false, "API OFFLINE"); stop(); console.error(error); }
}

function start() {
  if (running) return;
  running = true;
  $("run-button").textContent = "Pause loop";
  timer = setInterval(tick, 260);
}
function stop() {
  running = false;
  $("run-button").textContent = "Run loop";
  if (timer) clearInterval(timer);
  timer = null;
}

$("run-button").addEventListener("click", () => running ? stop() : start());
$("step-button").addEventListener("click", tick);
$("reset-button").addEventListener("click", async () => {
  stop();
  const body = { seed: Number($("seed").value), kind: $("scenario-kind").value, dynamic_actor_count: Number($("dynamic-count").value) };
  try { render(await request("/api/reset", { method: "POST", body: JSON.stringify(body) })); setConnection(true); }
  catch (error) { setConnection(false, "API OFFLINE"); console.error(error); }
});
window.addEventListener("resize", () => { if (state) drawWorld(state); });
refresh();
