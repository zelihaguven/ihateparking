(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("lot-canvas");
  const ctx = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const COLORS = {
    ground: "#061013", grid: "#0f2126", bay: "#2c4248", bayLabel: "#7f9692",
    car: "#34454b", carEdge: "#64787d", target: "#f3bf67", hover: "#72d8dc",
    plan: "#9ce6bf", ego: "#72d8dc", fail: "#ff786d", start: "#8ba29f",
  };

  let lot = null;
  let result = null;
  let hoverBay = null;
  let busy = false;
  let animation = { frame: 0, raf: 0, running: false };

  const setText = (id, value) => { $(id).textContent = value; };

  // ---- geometry -----------------------------------------------------------
  function view() {
    const rect = canvas.getBoundingClientRect();
    const [xmin, xmax, ymin, ymax] = lot.bounds;
    const pad = 10;
    const scale = Math.min((rect.width - 2 * pad) / (xmax - xmin), (rect.height - 2 * pad) / (ymax - ymin));
    return {
      width: rect.width, height: rect.height, scale,
      ox: (rect.width - (xmax - xmin) * scale) / 2 - xmin * scale,
      oy: (rect.height + (ymax - ymin) * scale) / 2 + ymin * scale,
    };
  }
  const toCanvas = (v, x, y) => [v.ox + x * v.scale, v.oy - y * v.scale];
  const toWorld = (v, px, py) => [(px - v.ox) / v.scale, (v.oy - py) / v.scale];

  function vehiclePolygon(pose) {
    const { length, width, rear_overhang: rear } = lot.vehicle;
    const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
    return [[-rear, -width / 2], [length - rear, -width / 2], [length - rear, width / 2], [-rear, width / 2]]
      .map(([x, y]) => [pose.x + x * c - y * s, pose.y + x * s + y * c]);
  }

  function inside(point, polygon) {
    let hit = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const [xi, yi] = polygon[i], [xj, yj] = polygon[j];
      if ((yi > point[1]) !== (yj > point[1]) && point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
  }

  // ---- drawing ------------------------------------------------------------
  function polygonPath(v, points) {
    ctx.beginPath();
    points.forEach(([x, y], index) => {
      const [px, py] = toCanvas(v, x, y);
      if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
  }

  function polyline(v, poses, color, width, dash = []) {
    if (poses.length < 2) return;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    ctx.beginPath();
    poses.forEach((pose, index) => {
      const [px, py] = toCanvas(v, pose.x, pose.y);
      if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.restore();
  }

  function drawVehicle(v, pose, stroke, fill, dash = []) {
    ctx.save();
    polygonPath(v, vehiclePolygon(pose));
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.setLineDash(dash); ctx.stroke();
    const [ax, ay] = toCanvas(v, pose.x, pose.y);
    const [nx, ny] = toCanvas(v, pose.x + 2.6 * Math.cos(pose.yaw), pose.y + 2.6 * Math.sin(pose.yaw));
    ctx.setLineDash([]); ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(nx, ny); ctx.stroke();
    ctx.restore();
  }

  function finalAttempt() {
    return result && result.attempts.length ? result.attempts[result.attempts.length - 1] : null;
  }

  function draw() {
    if (!lot) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const v = view();
    ctx.fillStyle = COLORS.ground; ctx.fillRect(0, 0, v.width, v.height);

    ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1;
    const [xmin, xmax, ymin, ymax] = lot.bounds;
    for (let x = xmin; x <= xmax; x += 5) { const [a, b] = toCanvas(v, x, ymin), [c, d] = toCanvas(v, x, ymax); ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke(); }
    for (let y = ymin; y <= ymax; y += 5) { const [a, b] = toCanvas(v, xmin, y), [c, d] = toCanvas(v, xmax, y); ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke(); }

    const attempt = finalAttempt();
    const target = $("bay-select").value;
    for (const bay of lot.bays) {
      const isTarget = bay.id === target;
      const isHover = hoverBay && bay.id === hoverBay.id && !bay.occupied;
      polygonPath(v, bay.polygon);
      if (isHover && !isTarget) { ctx.fillStyle = "rgba(114,216,220,.12)"; ctx.fill(); }
      ctx.save();
      ctx.strokeStyle = isTarget ? COLORS.target : isHover ? COLORS.hover : COLORS.bay;
      ctx.lineWidth = isTarget ? 2.6 : 1.2;
      ctx.setLineDash(isTarget ? [7, 5] : []);
      ctx.stroke();
      ctx.restore();
    }
    for (const obstacle of lot.obstacles) {
      polygonPath(v, obstacle.polygon);
      ctx.fillStyle = COLORS.car; ctx.fill();
      ctx.strokeStyle = COLORS.carEdge; ctx.lineWidth = 1; ctx.stroke();
    }
    ctx.font = "600 11px 'IBM Plex Mono', ui-monospace, monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const bay of lot.bays) {
      const [px, py] = toCanvas(v, bay.center[0], bay.center[1]);
      ctx.fillStyle = bay.id === target ? COLORS.target : COLORS.bayLabel;
      ctx.fillText(bay.id, px, py);
    }

    drawVehicle(v, lot.start, COLORS.start, null, [5, 4]);
    if (attempt) {
      polyline(v, attempt.plan, COLORS.plan, 1.5, [3, 4]);
      const trace = attempt.trace;
      if (trace.length) {
        const index = Math.min(animation.frame, trace.length - 1);
        polyline(v, trace.slice(0, index + 1), COLORS.ego, 2.2);
        const failed = index === trace.length - 1 && attempt.decision !== "PARKED";
        const colour = failed ? COLORS.fail : COLORS.ego;
        drawVehicle(v, trace[index], colour, failed ? "rgba(255,120,109,.25)" : "rgba(114,216,220,.28)");
        const gear = trace[index].gear < 0 ? "R" : "D";
        ctx.fillStyle = "#e9f1ed"; ctx.textAlign = "left";
        ctx.fillText(`gear ${gear}  ·  t = ${(index * (attempt.dt_s || 0)).toFixed(1)} s`, 14, 16);
      }
    }
  }

  // ---- panels -------------------------------------------------------------
  function colorFor(decision) {
    if (decision === "PARKED") return "#9ce6bf";
    if (decision === "REFUSED") return "#f3bf67";
    return "#ff786d";
  }

  function populateBays(selected) {
    const select = $("bay-select");
    select.replaceChildren();
    for (const bay of lot.bays) {
      const option = document.createElement("option");
      option.value = bay.id;
      option.textContent = `${bay.id} · ${bay.kind}${bay.occupied ? " · taken" : ""}`;
      select.append(option);
    }
    const firstFree = lot.bays.find((bay) => !bay.occupied);
    select.value = selected && lot.bays.some((bay) => bay.id === selected) ? selected : (firstFree ? firstFree.id : lot.bays[0].id);
  }

  function renderOptions(attempt) {
    const body = $("options-body");
    body.replaceChildren();
    if (!attempt || !attempt.options.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 4; cell.textContent = attempt ? "not evaluated — bay refused" : "—";
      row.append(cell); body.append(row);
      return;
    }
    for (const option of attempt.options) {
      const row = document.createElement("tr");
      if (option.manoeuvre === attempt.chosen_manoeuvre && attempt.decision === "PARKED") row.className = "chosen";
      else if (!option.planned) row.className = "unplanned";
      const cells = [
        option.manoeuvre,
        option.planned ? option.cost.toFixed(1) : (option.fits ? "no route" : "doesn't fit"),
        option.planned ? (option.exact_goal ? "yes" : "tolerance") : "—",
        `${option.plan_time_s.toFixed(1)} s`,
      ];
      for (const text of cells) { const cell = document.createElement("td"); cell.textContent = text; row.append(cell); }
      body.append(row);
    }
  }

  function renderResult() {
    const attempt = finalAttempt();
    const decision = result.decision;
    const badge = $("decision-badge");
    badge.textContent = decision;
    badge.style.color = colorFor(decision);
    const panel = $("decision-panel");
    panel.dataset.verdict = decision === "PARKED" ? "park" : decision === "REFUSED" ? "safe_stop" : "no_go";
    $("decision-title").textContent = attempt ? `${decision} · ${attempt.bay}` : "NO FREE BAY";
    $("decision-title").style.color = colorFor(decision);
    let message = attempt ? attempt.reason : "Every bay in this lot is taken.";
    if (!result.requested_bay && result.attempts.length) {
      const tried = result.attempts.map((item) => item.bay).join(" → ");
      message = `No bay requested; tried nearest-first: ${tried}. ${message}`;
    }
    setText("decision-message", message);
    renderOptions(attempt);
    setText("chosen", attempt && attempt.chosen_manoeuvre ? attempt.chosen_manoeuvre : "—");
    setText("gear-changes", attempt && attempt.gear_changes !== null ? String(attempt.gear_changes) : "—");
    setText("manoeuvre-time", attempt && attempt.manoeuvre_time_s !== null ? `${attempt.manoeuvre_time_s.toFixed(1)} s` : "—");
    setText("final-error", attempt && attempt.final_error_m !== null ? `${(attempt.final_error_m * 100).toFixed(1)} cm` : "—");
    setText("final-yaw", attempt && attempt.final_yaw_error_deg !== null ? `${attempt.final_yaw_error_deg.toFixed(1)}°` : "—");
    setText("collisions", attempt ? String(attempt.collisions) : "—");
    $("replay-button").disabled = !(attempt && attempt.trace.length);
    if (attempt) $("bay-select").value = attempt.bay;
  }

  // ---- animation ----------------------------------------------------------
  function play() {
    cancelAnimationFrame(animation.raf); animation.running = false;
    const attempt = finalAttempt();
    if (!attempt || !attempt.trace.length) { animation.frame = 0; draw(); return; }
    if (reduceMotion) { animation.frame = attempt.trace.length - 1; draw(); return; }
    const speed = Number($("speed-select").value);
    const started = performance.now();
    const step = (now) => {
      const simulated = ((now - started) / 1000) * speed;
      animation.frame = Math.min(attempt.trace.length - 1, Math.floor(simulated / attempt.dt_s));
      draw();
      if (animation.frame < attempt.trace.length - 1) animation.raf = requestAnimationFrame(step);
      else animation.running = false;
    };
    animation.running = true;
    animation.raf = requestAnimationFrame(step);
  }

  // ---- requests -----------------------------------------------------------
  function setBusy(value, label) {
    busy = value;
    for (const id of ["park-button", "nearest-button", "seed-input", "occupancy-select", "bay-select"]) $(id).disabled = value;
    canvas.classList.toggle("busy", value);
    if (label) setText("run-status", label);
  }

  async function loadLot(keepBay) {
    cancelAnimationFrame(animation.raf); animation.running = false;
    result = null; animation.frame = 0;
    setText("run-status", "LOADING LOT…");
    const params = new URLSearchParams({ seed: $("seed-input").value || "0", occupancy: $("occupancy-select").value });
    const response = await fetch(`/api/autopark/lot?${params}`);
    if (!response.ok) { setText("run-status", `LOT ERROR ${response.status}`); return; }
    lot = await response.json();
    populateBays(keepBay);
    const free = lot.bays.filter((bay) => !bay.occupied).length;
    setText("lot-title", `Lot seed ${lot.seed} · ${free} of ${lot.bays.length} bays free`);
    setText("run-status", "READY — CLICK A BAY");
    $("decision-badge").textContent = "READY";
    $("decision-badge").style.color = "";
    $("replay-button").disabled = true;
    renderOptions(null);
    draw();
  }

  async function park(bayId) {
    if (busy) return;
    cancelAnimationFrame(animation.raf); animation.running = false;
    result = null; animation.frame = 0;
    if (bayId) $("bay-select").value = bayId;
    draw();
    const started = performance.now();
    const label = bayId ? `PLANNING ${bayId}…` : "SEARCHING NEAREST FREE BAY…";
    setBusy(true, label);
    const timer = setInterval(() => setText("run-status", `${label} ${((performance.now() - started) / 1000).toFixed(0)} s`), 500);
    try {
      const response = await fetch("/api/autopark/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed: Number($("seed-input").value || 0), occupancy: Number($("occupancy-select").value), bay: bayId }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(response.status === 429 ? "Another simulation is running; try again shortly." : (detail.detail ? JSON.stringify(detail.detail) : `HTTP ${response.status}`));
      }
      result = await response.json();
      lot = result.lot;
      renderResult();
      setText("run-status", `${result.decision} · ${((performance.now() - started) / 1000).toFixed(1)} s`);
      play();
    } catch (error) {
      setText("run-status", "REQUEST FAILED");
      setText("decision-message", error.message);
      $("decision-title").textContent = "ERROR";
    } finally {
      clearInterval(timer);
      setBusy(false);
    }
  }

  // ---- events -------------------------------------------------------------
  function bayAt(event) {
    if (!lot) return null;
    const rect = canvas.getBoundingClientRect();
    const point = toWorld(view(), event.clientX - rect.left, event.clientY - rect.top);
    return lot.bays.find((bay) => inside(point, bay.polygon)) || null;
  }

  canvas.addEventListener("mousemove", (event) => {
    const bay = bayAt(event);
    if ((bay && bay.id) !== (hoverBay && hoverBay.id)) { hoverBay = bay; if (!animation.running) draw(); }
    canvas.title = bay ? `${bay.id} · ${bay.kind}${bay.occupied ? " · taken (will be refused)" : ""}` : "";
  });
  canvas.addEventListener("mouseleave", () => { hoverBay = null; draw(); });
  canvas.addEventListener("click", (event) => { const bay = bayAt(event); if (bay) park(bay.id); });
  $("park-button").addEventListener("click", () => park($("bay-select").value));
  $("nearest-button").addEventListener("click", () => park(null));
  $("replay-button").addEventListener("click", play);
  $("bay-select").addEventListener("change", draw);
  $("seed-input").addEventListener("change", () => loadLot($("bay-select").value));
  $("occupancy-select").addEventListener("change", () => loadLot($("bay-select").value));
  window.addEventListener("resize", draw);

  loadLot();
})();
