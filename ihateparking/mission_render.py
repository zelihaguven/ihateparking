"""Figure for one perception-driven mission: the car's own map vs the truth."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from .geometry import vehicle_polygon  # noqa: E402
from .mission import MissionResult  # noqa: E402
from .occupancy_map import FREE, OCCUPIED  # noqa: E402
from .procedural_lot import ProceduralLot  # noqa: E402

BACKGROUND = "#071114"
TEXT = "#e9f1ed"
MUTED = "#8ba29f"
EGO = "#72d8dc"
GOOD = "#9ce6bf"
BAD = "#ff786d"
FREE_BAY = "#9ce6bf"


def _footprint(lot: ProceduralLot, pose) -> np.ndarray:
    v = lot.vehicle
    return vehicle_polygon(pose, v.length, v.width, v.rear_overhang)


def render_mission(lot: ProceduralLot, result: MissionResult, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(18, 9), facecolor=BACKGROUND)
    grid = result.grid
    states = grid.states()
    image = np.zeros(states.shape + (3,))
    image[...] = (0.02, 0.05, 0.06)
    image[states == FREE] = (0.10, 0.19, 0.21)
    image[states == OCCUPIED] = (0.92, 0.95, 0.94)
    size = grid.config.size_m
    extent = [grid.origin[0], grid.origin[0] + size, grid.origin[1], grid.origin[1] + size]
    left.set_facecolor("#05090a")
    left.imshow(image, origin="lower", extent=extent, interpolation="nearest")
    left.set_title("What the car built from its LiDAR (dark = never seen)", color=TEXT, fontsize=12)

    right.set_facecolor("#061013")
    for bay in lot.bays:
        right.add_patch(Polygon(bay.slot.polygon(), closed=True, fill=False, ec=FREE_BAY if not bay.occupied else "#2c4248", lw=1.2 if not bay.occupied else 0.7))
    for obstacle in lot.obstacles:
        right.add_patch(Polygon(obstacle.polygon(), closed=True, fc="#34454b" if obstacle.name.startswith("car") else "#8ba29f", ec="#64787d", lw=0.6))
    right.set_title("Ground truth (never given to the car); green = free bays", color=TEXT, fontsize=12)

    xs = [pose.x for pose in result.trace]
    ys = [pose.y for pose in result.trace]
    final = result.trace[-1]
    colour = GOOD if result.success else BAD
    for ax in (left, right):
        ax.plot(xs, ys, color=EGO, lw=1.6)
        ax.add_patch(Polygon(_footprint(lot, lot.start), closed=True, fill=False, ec=MUTED, lw=1.2, ls="--"))
        ax.add_patch(Polygon(_footprint(lot, final), closed=True, fc=colour + "55", ec=colour, lw=2.2))
        if result.target is not None:
            for heading in result.target.headings()[:1]:
                ax.add_patch(Polygon(_footprint(lot, result.target.goal(heading, lot.vehicle)), closed=True, fill=False, ec="#f3bf67", lw=1.4, ls=":"))
        ax.set_xlim(lot.bounds[0], lot.bounds[1])
        ax.set_ylim(lot.bounds[2], lot.bounds[3])
        ax.set_aspect("equal")
        ax.tick_params(colors="#5e7774")
        for spine in ax.spines.values():
            spine.set_color("#20363c")

    evaluation = result.evaluation
    verdict = "PARKED IN A FREE BAY" if result.success else result.outcome.replace("_", " ")
    detail = ""
    if evaluation:
        detail = f"  -  truth: bay {evaluation.get('bay')} ({evaluation.get('bay_kind')}), overlap {evaluation.get('overlap', 0) * 100:.0f}%, axis error {evaluation.get('axis_error_deg', 0):.1f} deg"
    fig.suptitle(
        f"Unseen lot seed {lot.seed}: {verdict}{detail}\n{result.message}  -  {result.time_s:.0f} s simulated, {result.distance_m:.0f} m driven, {result.collisions} collisions",
        color=colour, fontsize=13, family="monospace",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=90, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output
