"""Matplotlib rendering for portfolio-quality v1 artifacts."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from .controller import SimulationResult
from .planner import PlanResult
from .world import ParkingWorld


def render_run(world: ParkingWorld, plan: PlanResult, simulation: SimulationResult, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    xmin, xmax, ymin, ymax = world.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f3f5f7")
    ax.grid(True, alpha=0.18)

    for obstacle in world.obstacles:
        poly = obstacle.polygon()
        ax.add_patch(Polygon(poly, closed=True, facecolor="#3b4652", edgecolor="#17202a", linewidth=1.2))
        ax.text(obstacle.center[0], obstacle.center[1], obstacle.name.replace("_", " "), ha="center", va="center", color="white", fontsize=8)

    slot = world.slot.polygon()
    ax.add_patch(Polygon(slot, closed=True, facecolor="#65c18c", edgecolor="#1c7c54", linewidth=2, alpha=0.40, hatch="///"))
    ax.text(world.slot.center[0], world.slot.center[1], "TARGET SLOT", ha="center", va="center", color="#1c5c40", fontsize=9, weight="bold", rotation=90)

    if plan.poses:
        ax.plot([p.x for p in plan.poses], [p.y for p in plan.poses], "--", color="#f39c12", linewidth=1.3, label="Hybrid A* plan")
    if simulation.poses:
        ax.plot([p.x for p in simulation.poses], [p.y for p in simulation.poses], color="#2878b5", linewidth=2.0, label="Closed-loop trace")

    def draw_vehicle(pose, color, label=None, alpha=1.0):
        poly = world.vehicle_polygon(pose)
        ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor="#102a43", linewidth=1.2, alpha=alpha, label=label))
        ax.arrow(pose.x, pose.y, 0.9 * math.cos(pose.yaw), 0.9 * math.sin(pose.yaw), head_width=0.18, head_length=0.25, color="#102a43", length_includes_head=True)

    draw_vehicle(world.start_pose, "#f6c85f", "Start")
    if simulation.poses:
        draw_vehicle(simulation.poses[-1], "#5ac878" if simulation.success else "#e76f51", "Final")

    title = "ihateparking v1 — closed-loop autonomous parking"
    subtitle = f"planner={'OK' if plan.success else 'FAIL'} | controller={'OK' if simulation.success else 'FAIL'} | expanded={plan.expanded} | time={simulation.time:.1f}s"
    ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=13, weight="bold")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output
