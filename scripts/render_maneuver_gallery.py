#!/usr/bin/env python3
"""Plot all six manoeuvres for one seed: obstacles, target slot, plan, trace, final pose.

    python3 scripts/render_maneuver_gallery.py artifacts/gallery/seed7.png 7
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from ihateparking.scenario_generator import ScenarioConfig, ScenarioGenerator, ScenarioKind
from ihateparking.planner import HybridAStar, PlannerConfig
from ihateparking.controller import simulate_closed_loop
from ihateparking.geometry import distance, angle_error

output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/gallery/maneuvers.png")
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
output.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="#071114")
for ax, kind in zip(axes.flat, ScenarioKind):
    w = ScenarioGenerator().generate(ScenarioConfig(seed=seed, kind=kind)).to_world()
    p = HybridAStar(w, PlannerConfig(max_expansions=30000)).plan()
    r = simulate_closed_loop(w, p)
    g = w.slot.goal_pose(w.vehicle); f = r.poses[-1]
    ax.set_facecolor("#061013")
    for o in w.obstacles:
        ax.add_patch(Polygon(o.polygon(), closed=True, fc="#34454b", ec="#64787d"))
    ax.add_patch(Polygon(w.slot.polygon(), closed=True, fill=False, ec="#f3bf67", lw=2.5, ls="--"))
    ax.plot([q.x for q in p.poses], [q.y for q in p.poses], color="#9ce6bf", lw=1.2, ls=":", label="plan")
    ax.plot([q.x for q in r.poses], [q.y for q in r.poses], color="#72d8dc", lw=1.6, label="executed trace")
    for q in r.poses[::60]:
        ax.add_patch(Polygon(w.vehicle_polygon(q), closed=True, fill=False, ec="#72d8dc", lw=0.5, alpha=0.35))
    ax.add_patch(Polygon(w.vehicle_polygon(w.start_pose), closed=True, fill=False, ec="#8ba29f", lw=1.2, ls="--"))
    ax.add_patch(Polygon(w.vehicle_polygon(f), closed=True, fc="#72d8dc55", ec="#72d8dc", lw=2))
    nose = (f.x + 3.4*math.cos(f.yaw), f.y + 3.4*math.sin(f.yaw))
    ax.annotate("", xy=nose, xytext=(f.x, f.y), arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
    status = "PARKED" if r.success else "FAILED"
    gears = sum(1 for a, b in zip(r.gears, r.gears[1:]) if a != b)
    ax.set_title(f"{kind.value}  -  {status}\nerror {distance(f,g)*100:.1f} cm / {abs(math.degrees(angle_error(g.yaw,f.yaw))):.1f} deg  -  {gears} gear changes  -  {r.collisions} collisions",
                 color="#9ce6bf" if r.success else "#ff786d", fontsize=11, family="monospace")
    xs = [c[0] for o in w.obstacles for c in o.polygon()] + [q.x for q in r.poses]
    ys = [c[1] for o in w.obstacles for c in o.polygon()] + [q.y for q in r.poses]
    ax.set_xlim(max(w.bounds[0], min(xs)-1), min(w.bounds[1], max(xs)+1)); ax.set_ylim(max(w.bounds[2], min(ys)-1), min(w.bounds[3], max(ys)+1))
    ax.set_aspect("equal"); ax.tick_params(colors="#5e7774")
    for s in ax.spines.values(): s.set_color("#20363c")
axes.flat[0].legend(loc="lower right", facecolor="#0d1b1f", labelcolor="#e9f1ed", fontsize=9)
fig.suptitle(f"ihateparking - six manoeuvres, seed {seed}  (dashed orange = target slot, filled cyan = final pose, arrow = vehicle nose)", color="#e9f1ed", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(output, dpi=110, facecolor=fig.get_facecolor())
print("saved", output)
