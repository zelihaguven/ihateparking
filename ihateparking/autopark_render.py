"""Matplotlib figure for one autopark attempt: lot, bays, plan, trace, outcome."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from .autopark import AutoparkResult  # noqa: E402
from .parking_lot import ParkingLot  # noqa: E402

BACKGROUND = "#071114"
GROUND = "#061013"
TEXT = "#e9f1ed"
MUTED = "#8ba29f"
CAR = "#34454b"
CAR_EDGE = "#64787d"
BAY = "#3d5257"
TARGET = "#f3bf67"
PLAN = "#9ce6bf"
EGO = "#72d8dc"
FAIL = "#ff786d"


def render_autopark(lot: ParkingLot, result: AutoparkResult, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    world = result.world
    fig, ax = plt.subplots(figsize=(14, 10.5), facecolor=BACKGROUND)
    ax.set_facecolor(GROUND)

    for bay in lot.bays:
        is_target = bay.bay_id == result.bay_id
        ax.add_patch(Polygon(
            bay.slot.polygon(), closed=True, fill=False,
            ec=TARGET if is_target else BAY, lw=2.6 if is_target else 1.0, ls="--" if is_target else "-",
        ))
        ax.text(*bay.center, bay.bay_id, ha="center", va="center", fontsize=8,
                color=TARGET if is_target else MUTED, family="monospace", zorder=5)
    for obstacle in lot.obstacles:
        ax.add_patch(Polygon(obstacle.polygon(), closed=True, fc=CAR, ec=CAR_EDGE, lw=0.8))

    ax.add_patch(Polygon(world.vehicle_polygon(world.start_pose), closed=True, fill=False, ec=MUTED, lw=1.2, ls="--"))
    if result.chosen is not None and result.chosen.plan is not None:
        plan = result.chosen.plan
        ax.plot([p.x for p in plan.poses], [p.y for p in plan.poses], color=PLAN, lw=1.2, ls=":", label="plan")
    if result.simulation is not None:
        poses = result.simulation.poses
        ax.plot([p.x for p in poses], [p.y for p in poses], color=EGO, lw=1.8, label="executed trace")
        for pose in poses[::50]:
            ax.add_patch(Polygon(world.vehicle_polygon(pose), closed=True, fill=False, ec=EGO, lw=0.5, alpha=0.3))
        final = poses[-1]
        colour = EGO if result.parked else FAIL
        ax.add_patch(Polygon(world.vehicle_polygon(final), closed=True, fc=colour + "55", ec=colour, lw=2.2))
        nose = (final.x + 3.4 * math.cos(final.yaw), final.y + 3.4 * math.sin(final.yaw))
        ax.annotate("", xy=nose, xytext=(final.x, final.y), arrowprops=dict(arrowstyle="->", color="white", lw=1.6))

    summary = result.summary()
    colour = PLAN if result.parked else FAIL
    title = f"Target bay {result.bay_id}: {result.decision}"
    if summary["final_error_m"] is not None and result.parked:
        title += f"  -  error {summary['final_error_m'] * 100:.1f} cm / {summary['final_yaw_error_deg']:.1f} deg  -  0 collisions"
    ax.set_title(f"{title}\n{result.reason}", color=colour, fontsize=12, family="monospace", pad=12)
    options = "   |   ".join(
        f"{item['manoeuvre']}: " + (f"cost {item['cost']}" if item["planned"] else item["message"][:40])
        for item in summary["options"]
    )
    fig.text(0.5, 0.03, f"options evaluated  -  {options}", ha="center", color=MUTED, fontsize=9, family="monospace")

    xmin, xmax, ymin, ymax = lot.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.tick_params(colors="#5e7774")
    for spine in ax.spines.values():
        spine.set_color("#20363c")
    if result.simulation is not None:
        ax.legend(loc="upper left", facecolor="#0d1b1f", labelcolor=TEXT, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output
