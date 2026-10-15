const byId = (id) => document.getElementById(id);
const formValues = {
  horizon_steps: () => Number(byId("horizon").value),
  max_iterations: () => Number(byId("iterations").value),
  dt: () => Number(byId("dt").value),
  extra_clearance_m: () => Number(byId("extra-clearance").value),
  warm_start: () => byId("warm-start").checked,
};
let lastResult = null;

function fmt(value, digits = 2, suffix = "") {
  const n = Number(value);
  return Number.isFinite(n) ? `${n.toFixed(digits)}${suffix}` : "—";
}

function setStatus(message, kind = "") {
  const node = byId("run-status");
  node.textContent = message;
  node.dataset.kind = kind;
}

function setCardStatus(id, success, label) {
  const node = byId(id);
  node.textContent = label;
  node.classList.toggle("fail", !success);
  node.classList.toggle("warn", false);
}

function setResultCard(prefix, result) {
  const blocked = Number(result.collisions || 0) > 0;
  setCardStatus(`${prefix}-status`, Boolean(result.success), result.success ? "GOAL REACHED" : blocked ? "SAFETY STOP" : "NOT COMPLETED");
  if (blocked && !result.success) {
    const status = byId(`${prefix}-status`);
    status.classList.remove("fail");
    status.classList.add("warn");
  }
  byId(`${prefix}-message`).textContent = blocked
    ? "Safety check rejected the proposed next pose before it was applied to the executed trace."
    : (result.message || "—");
  byId(`${prefix}-collisions`).previousElementSibling.textContent = "collision-guard stops";
  byId(`${prefix}-collisions`).textContent = String(result.collisions ?? "—");
  byId(`${prefix}-clearance`).textContent = fmt(result.minimum_raw_polygon_clearance_m, 3, " m");
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(300, rect.width);
  const height = Math.max(220, rect.height);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function drawTrajectory(data) {
  const canvas = byId("trajectory-canvas");
  const { ctx, width, height } = resizeCanvas(canvas);
  const [xmin, xmax, ymin, ymax] = data.scene.bounds;
  const margin = 28;
  const scale = Math.min((width - 2 * margin) / (xmax - xmin), (height - 2 * margin) / (ymax - ymin));
  const drawWidth = (xmax - xmin) * scale;
  const drawHeight = (ymax - ymin) * scale;
  const ox = (width - drawWidth) / 2;
  const oy = (height - drawHeight) / 2;
  const point = ([x, y]) => [ox + (x - xmin) * scale, height - oy - (y - ymin) * scale];

  ctx.fillStyle = "#071114";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#183139";
  ctx.lineWidth = 1;
  for (let x = Math.ceil(xmin); x <= xmax; x += 1) {
    const [px] = point([x, ymin]);
    ctx.beginPath(); ctx.moveTo(px, oy); ctx.lineTo(px, height - oy); ctx.stroke();
  }
  for (let y = Math.ceil(ymin); y <= ymax; y += 1) {
    const [, py] = point([xmin, y]);
    ctx.beginPath(); ctx.moveTo(ox, py); ctx.lineTo(width - ox, py); ctx.stroke();
  }

  const polygon = (vertices, fill, stroke, dash = []) => {
    if (!vertices?.length) return;
    ctx.beginPath();
    vertices.forEach(([x, y], index) => {
      const [px, py] = point([x, y]);
      if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.setLineDash(dash);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
    ctx.setLineDash([]);
  };
  const path = (poses, color, dash = []) => {
    if (!poses?.length) return;
    ctx.beginPath();
    poses.forEach((pose, index) => {
      const [px, py] = point([pose.x, pose.y]);
      if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.4;
    ctx.setLineDash(dash);
    ctx.stroke();
    ctx.setLineDash([]);
  };

  data.scene.obstacles.forEach((obstacle) => polygon(obstacle.polygon, "#40535a", "#788b90"));
  polygon(data.scene.slot_polygon, null, "#f3bf67", [7, 5]);
  path(data.trajectories.reference, "#91a39e", [4, 5]);
  path(data.trajectories.pure_pursuit, "#75a9ff");
  path(data.trajectories.constrained_mpc, "#ff786d");
  const start = point([data.scene.start.x, data.scene.start.y]);
  const goal = point([data.scene.goal.x, data.scene.goal.y]);
  ctx.fillStyle = "#e9f1ed"; ctx.beginPath(); ctx.arc(start[0], start[1], 4, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#9ce6bf"; ctx.beginPath(); ctx.arc(goal[0], goal[1], 5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#8ba29f"; ctx.font = "10px ui-monospace,monospace";
  ctx.fillText("start", start[0] + 7, start[1] - 7);
  ctx.fillText("goal", goal[0] + 7, goal[1] - 7);
  ctx.fillText("x / y [m]", 10, 16);
}

function drawLatency(data) {
  const canvas = byId("latency-canvas");
  const { ctx, width, height } = resizeCanvas(canvas);
  const values = data.latency.controller_call_ms || [];
  const budget = data.latency.simulation_step_budget_ms;
  const margin = { left: 46, right: 14, top: 24, bottom: 32 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const maxValue = Math.max(200, budget * 1.12, ...values) * 1.12;
  const y = (value) => margin.top + plotH * (1 - value / maxValue);

  ctx.fillStyle = "#071114"; ctx.fillRect(0, 0, width, height);
  ctx.font = "9px ui-monospace,monospace";
  ctx.strokeStyle = "#20363c"; ctx.fillStyle = "#8ba29f";
  for (let tick = 0; tick <= 4; tick += 1) {
    const v = maxValue * tick / 4;
    const py = y(v);
    ctx.beginPath(); ctx.moveTo(margin.left, py); ctx.lineTo(width - margin.right, py); ctx.stroke();
    ctx.fillText(`${Math.round(v)}`, 7, py + 3);
  }
  const thresholds = [
    { value: 20, color: "#9ce6bf", label: "50 Hz · 20 ms" },
    { value: 50, color: "#72d8dc", label: "20 Hz · 50 ms" },
    { value: budget, color: "#f3bf67", label: `sim step · ${Math.round(budget)} ms` },
  ];
  thresholds.forEach((item) => {
    if (item.value > maxValue) return;
    const py = y(item.value);
    ctx.strokeStyle = item.color; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(margin.left, py); ctx.lineTo(width - margin.right, py); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = item.color; ctx.fillText(item.label, margin.left + 5, py - 4);
  });
  if (values.length) {
    ctx.beginPath();
    values.forEach((value, index) => {
      const px = margin.left + (values.length < 2 ? 0 : index / (values.length - 1) * plotW);
      const py = y(value);
      if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = "#ff786d"; ctx.lineWidth = 2; ctx.stroke();
  }
  ctx.fillStyle = "#8ba29f"; ctx.fillText("controller call", margin.left, height - 9);
}

function render(data) {
  lastResult = data;
  byId("results").hidden = false;
  setResultCard("pp", data.pure_pursuit);
  setResultCard("mpc", data.constrained_mpc);

  const calls = data.latency.controller_call_ms || [];
  const sorted = calls.slice().sort((a, b) => a - b);
  const p95 = data.constrained_mpc.p95_controller_call_ms;
  const budget = data.latency.simulation_step_budget_ms;
  const misses = data.latency.misses_simulation_step_budget;
  const count = data.latency.controller_calls;
  const realPeriod = p95 <= 20 ? "p95 within 50 Hz period" : p95 <= 50 ? "p95 within 20 Hz period only" : "p95 misses 20–50 Hz periods";
  const budgetPass = Number.isFinite(p95) && p95 <= budget;
  setCardStatus("latency-status", budgetPass, budgetPass ? "WITHIN SIM STEP" : "MISSES SIM STEP");
  if (budgetPass && p95 > 50) byId("latency-status").classList.add("warn");
  byId("latency-message").textContent = `${realPeriod}; shifted warm start ${data.mpc_configuration.warm_start ? "on" : "off"}; ${misses}/${count} calls exceeded ${fmt(budget, 0, " ms")} simulation step.`;
  byId("latency-percentiles").textContent = `${fmt(data.constrained_mpc.mean_controller_call_ms, 1)} / ${fmt(p95, 1)} / ${fmt(data.constrained_mpc.p99_controller_call_ms, 1)} ms`;
  byId("deadline-misses").textContent = `${misses} / ${count}`;
  byId("optimizer-p95").textContent = fmt(data.constrained_mpc.p95_mpc_solve_time_ms, 1, " ms");
  byId("safety-note").textContent = `${data.safety_note} Collision-guard stops mean a proposed next pose was rejected before it was appended to the executed trace; this demo does not deliberately apply a colliding pose. Scene: ${data.scenario.reference_source}. ${data.constrained_mpc.success ? "The current MPC rollout reached the goal." : "The MPC rollout did not complete; treat it as a failed experiment, not a reason to fall back to Pure Pursuit."}`;
  byId("download-json").disabled = false;
  drawTrajectory(data);
  drawLatency(data);
  if (budgetPass) setStatus(`Run complete: ${count} MPC calls, no simulation-step deadline misses in this run. This is not a vehicle real-time guarantee.`, "success");
  else setStatus(`Run complete with ${misses} / ${count} simulation-step deadline misses. Inspect feasibility and latency before changing parameters.`, "error");
}

async function runComparison() {
  const button = byId("run-comparison");
  button.disabled = true;
  byId("download-json").disabled = true;
  setStatus("Running paired Python simulation… this can take several seconds.");
  const body = Object.fromEntries(Object.entries(formValues).map(([key, value]) => [key, value()]));
  try {
    const response = await fetch("/api/mpc/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `${response.status} ${response.statusText}`);
    render(payload);
  } catch (error) {
    setStatus(`Run failed: ${error.message}`, "error");
  } finally {
    button.disabled = false;
  }
}

byId("extra-clearance").addEventListener("input", (event) => {
  byId("clearance-value").value = `${Number(event.target.value).toFixed(2)} m`;
});
byId("fast-preset").addEventListener("click", () => {
  byId("horizon").value = "4"; byId("iterations").value = "12"; byId("dt").value = "0.18";
  byId("warm-start").checked = true;
  setStatus("Quick profile loaded. Run it to measure this session’s actual results.");
});
byId("reference-preset").addEventListener("click", () => {
  byId("horizon").value = "8"; byId("iterations").value = "40"; byId("dt").value = "0.18";
  byId("warm-start").checked = false;
  setStatus("Reference profile loaded. Expect higher SLSQP latency; run it to measure.");
});
byId("run-comparison").addEventListener("click", runComparison);
byId("download-json").addEventListener("click", () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url; link.download = "ihateparking-mpc-run.json"; link.click();
  URL.revokeObjectURL(url);
});
window.addEventListener("resize", () => {
  if (lastResult) { drawTrajectory(lastResult); drawLatency(lastResult); }
});
