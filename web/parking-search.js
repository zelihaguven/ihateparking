(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("scene-canvas");
  const ctx = canvas.getContext("2d");
  const runButton = $("assess-button");
  let lastResult = null;

  const setText = (id, value) => { $(id).textContent = value; };
  const meters = (value) => Number.isFinite(value) ? `${value.toFixed(2)} m` : "—";

  function colorForDecision(verdict) {
    if (verdict === "PARK") return "#9ce6bf";
    if (verdict === "SAFE_STOP") return "#f3bf67";
    if (verdict === "NO_GO") return "#ff786d";
    return "#72d8dc";
  }

  function renderStateSequence(states) {
    const list = $("state-sequence");
    list.replaceChildren();
    for (const state of states || []) {
      const item = document.createElement("li");
      item.textContent = state.replaceAll("_", " ");
      list.append(item);
    }
  }

  function updateMetrics(result) {
    const assessment = result.assessment || {};
    const candidate = result.scene?.candidate;
    const estimate = candidate?.estimate || {};
    setText("gap-width", meters(assessment.estimated_width_m));
    setText("required-width", meters(assessment.required_width_m));
    setText("gap-length", meters(assessment.estimated_length_m));
    setText("required-length", meters(assessment.required_length_m));
    setText("confidence", `${Number(assessment.confidence || 0).toFixed(2)} / ${Number(assessment.confidence_floor || 0).toFixed(2)}`);
    const checks = assessment.checks || [];
    const passed = checks.filter((check) => check.passed).length;
    setText("fit-checks", checks.length ? `${passed}/${checks.length} passed` : "stopped early");
    setText("planner-status", result.planner ? (result.planner.success ? "PATH VERIFIED" : "NO PATH FOUND") : "NOT RUN");
    setText("planner-expansions", result.planner ? String(result.planner.expanded) : "0");
    setText("motion-steps", String(result.safety?.motion_steps ?? 0));
    setText("applied-contacts", String(result.safety?.applied_collision_count ?? 0));
    setText("guard-rejections", String(result.safety?.rejected_unsafe_step_proposals ?? 0));
    const badge = $("case-badge");
    badge.textContent = candidate?.decision || result.status || "NOT RUN";
    badge.style.color = colorForDecision(result.decision);
    const panel = $("decision-panel");
    panel.dataset.verdict = result.decision.toLowerCase();
    $("decision-title").textContent = result.status.replaceAll("_", " ");
    $("decision-title").style.color = colorForDecision(result.decision);
    setText("decision-message", result.message);
    setText("run-status", `${result.decision} · ${result.case_label}`.toUpperCase());
    renderStateSequence(result.state_sequence);
  }

  function worldToCanvas(point, view) {
    return {
      x: view.offsetX + (point[0] - view.xMin) * view.scale,
      y: view.height - view.offsetY - (point[1] - view.yMin) * view.scale,
    };
  }

  function drawRotatedRect(center, length, width, yaw, view, fill, stroke, lineWidth = 1.5, dash = []) {
    const localCorners = [
      [-length / 2, -width / 2], [length / 2, -width / 2],
      [length / 2, width / 2], [-length / 2, width / 2],
    ];
    const corners = localCorners.map(([x, y]) => [
      center[0] + x * Math.cos(yaw) - y * Math.sin(yaw),
      center[1] + x * Math.sin(yaw) + y * Math.cos(yaw),
    ]).map((p) => worldToCanvas(p, view));
    ctx.beginPath();
    ctx.moveTo(corners[0].x, corners[0].y);
    for (const corner of corners.slice(1)) ctx.lineTo(corner.x, corner.y);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.setLineDash(dash);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawPath(points, view, color, dash = []) {
    if (!Array.isArray(points) || points.length < 2) return;
    ctx.beginPath();
    const first = worldToCanvas([points[0].x, points[0].y], view);
    ctx.moveTo(first.x, first.y);
    for (const pose of points.slice(1)) {
      const point = worldToCanvas([pose.x, pose.y], view);
      ctx.lineTo(point.x, point.y);
    }
    ctx.setLineDash(dash);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawArrow(pose, view, color) {
    if (!pose) return;
    const start = worldToCanvas([pose.x, pose.y], view);
    const end = worldToCanvas([pose.x + 0.9 * Math.cos(pose.yaw), pose.y + 0.9 * Math.sin(pose.yaw)], view);
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(start.x, start.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  function drawScene(result) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(320, rect.width);
    const height = Math.max(300, rect.height);
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.fillStyle = "#061013";
    ctx.fillRect(0, 0, width, height);

    const scene = result.scene;
    const [xMin, xMax, yMin, yMax] = scene.bounds;
    const pad = 26;
    const scale = Math.min((width - 2 * pad) / (xMax - xMin), (height - 2 * pad) / (yMax - yMin));
    const viewWidth = (xMax - xMin) * scale;
    const viewHeight = (yMax - yMin) * scale;
    const view = { xMin, xMax, yMin, yMax, width, height, scale, offsetX: (width - viewWidth) / 2, offsetY: (height - viewHeight) / 2 };
    const topLeft = worldToCanvas([xMin, yMax], view);
    ctx.fillStyle = "#0b171a";
    ctx.fillRect(topLeft.x, topLeft.y, viewWidth, viewHeight);

    ctx.strokeStyle = "#13272b";
    ctx.lineWidth = 1;
    for (let x = Math.ceil(xMin); x <= xMax; x += 1) {
      const a = worldToCanvas([x, yMin], view), b = worldToCanvas([x, yMax], view);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    for (let y = Math.ceil(yMin); y <= yMax; y += 1) {
      const a = worldToCanvas([xMin, y], view), b = worldToCanvas([xMax, y], view);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }

    for (const obstacle of scene.perceived_obstacles || []) {
      const isBlocker = obstacle.name === "blocking_vehicle";
      if (obstacle.kind === "curb") {
        drawRotatedRect(obstacle.center, obstacle.length_m, obstacle.width_m, obstacle.yaw, view, "#5c6668", "#869294", 1);
      } else {
        drawRotatedRect(obstacle.center, obstacle.length_m, obstacle.width_m, obstacle.yaw, view,
          isBlocker ? "#532d31" : "#29383d", isBlocker ? "#ff786d" : "#75888d", 1.5);
        const label = worldToCanvas(obstacle.center, view);
        ctx.fillStyle = isBlocker ? "#ffb0a8" : "#c0ceca";
        ctx.font = "10px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText(isBlocker ? "OCCUPIED" : "PARKED CAR", label.x, label.y + 4);
      }
    }

    const candidate = scene.candidate;
    const estimate = candidate.estimate;
    drawRotatedRect(estimate.center, estimate.length_m, estimate.width_m, Math.PI / 2,
      view, "#f3bf6718", result.decision === "NO_GO" ? "#ff786d" : "#f3bf67", 2, [7, 5]);
    const slotLabel = worldToCanvas(estimate.center, view);
    ctx.fillStyle = "#f3bf67";
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText("CANDIDATE", slotLabel.x, slotLabel.y + 4);

    if (result.planner?.success) drawPath(result.planner.path, view, "#9ce6bf", [7, 5]);
    if (result.execution?.trajectory) drawPath(result.execution.trajectory, view, "#72d8dc");
    drawArrow(scene.ego_start, view, "#72d8dc");
    if (result.execution?.final_pose) drawArrow(result.execution.final_pose, view, "#9ce6bf");

    ctx.fillStyle = "#809793";
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.fillText("SYNTHETIC TOP-DOWN WORLD · METRES", 10, 16);
  }

  async function runAssessment() {
    const button = runButton;
    button.disabled = true;
    setText("run-status", "ASSESSING SYNTHETIC SCENE…");
    button.textContent = "Evaluating…";
    try {
      const seed = Number($("seed-input").value);
      if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) throw new Error("Seed must be an integer from 0 to 2,147,483,647.");
      const response = await fetch("/api/parking-search/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case: $("case-select").value, seed }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status}).`);
      lastResult = payload;
      updateMetrics(payload);
      drawScene(payload);
    } catch (error) {
      setText("run-status", "REQUEST FAILED");
      setText("decision-title", "ASSESSMENT UNAVAILABLE");
      setText("decision-message", error instanceof Error ? error.message : "Unexpected request failure.");
    } finally {
      button.disabled = false;
      button.textContent = "Assess and simulate";
    }
  }

  runButton.addEventListener("click", runAssessment);
  window.addEventListener("resize", () => { if (lastResult) drawScene(lastResult); });
})();
