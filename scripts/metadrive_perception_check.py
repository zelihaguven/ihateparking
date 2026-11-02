"""Does the stack's bay detector find MetaDrive's free spaces from MetaDrive's LiDAR?

Builds a MetaDrive parking lot, moves the ego car along the aisle (teleport,
no driving yet), integrates MetaDrive LiDAR scans into the stack's occupancy
grid, runs the bay detector and scores every detection against MetaDrive's
spaces. Run in the MetaDrive environment:

    .venv-metadrive/bin/python scripts/metadrive_perception_check.py [seed ...]
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihateparking.geometry import Pose  # noqa: E402
from ihateparking.metadrive_bridge import MetaDriveLot, space_overlap  # noqa: E402
from ihateparking.occupancy_map import FREE, OCCUPIED, OccupancyGrid  # noqa: E402
from ihateparking.slot_detector import detect_slots  # noqa: E402


def aisle_sweep(lot: MetaDriveLot) -> list[Pose]:
    entrances = [np.array(s.center) - s.length / 2.0 * np.array([math.cos(s.yaw), math.sin(s.yaw)]) for s in lot.spaces]
    middle = np.mean(entrances, axis=0)
    yaw = lot.spaces[0].yaw
    along = math.atan2(math.cos(yaw), -math.sin(yaw))  # square to the spaces
    return [Pose(float(middle[0] + t * math.cos(along)), float(middle[1] + t * math.sin(along)), along) for t in np.arange(-14.0, 14.1, 1.5)]


def check(seed: int, figure: Path | None) -> tuple[int, int, int]:
    lot = MetaDriveLot(seed)
    try:
        poses = aisle_sweep(lot)
        grid = OccupancyGrid((poses[0].x, poses[0].y))
        for pose in poses:
            lot.teleport(pose)
            lot.env.step([0.0, 0.0])
            grid.integrate(lot.scan())
        detections = detect_slots(grid, lot.vehicle)
        good = bad = 0
        for candidate in detections:
            best = max(max(space_overlap(space, candidate.goal(h, lot.vehicle), lot.vehicle) for h in candidate.headings()) * (0 if space.occupied else 1) for space in lot.spaces)
            if best >= 0.8:
                good += 1
            else:
                bad += 1
        free = sum(not s.occupied for s in lot.spaces)
        print(f"seed {seed}: {len(lot.spaces)} spaces, {free} free; detections {len(detections)}: {good} in a free space, {bad} not")
        if figure is not None:
            _draw(lot, grid, detections, figure)
        return good, bad, free
    finally:
        lot.close()


def _draw(lot, grid, detections, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    from ihateparking.geometry import vehicle_polygon

    states = grid.states()
    image = np.full(states.shape + (3,), 0.6)
    image[states == FREE] = (0.88, 0.92, 0.92)
    image[states == OCCUPIED] = (0.0, 0.0, 0.0)
    size = grid.config.size_m
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, origin="lower", extent=[grid.origin[0], grid.origin[0] + size, grid.origin[1], grid.origin[1] + size])
    for space in lot.spaces:
        c, s = math.cos(space.yaw), math.sin(space.yaw)
        corners = [(space.center[0] + c * a - s * b, space.center[1] + s * a + c * b) for a, b in ((-space.length / 2, -space.width / 2), (space.length / 2, -space.width / 2), (space.length / 2, space.width / 2), (-space.length / 2, space.width / 2))]
        ax.add_patch(Polygon(corners, fill=False, ec="red" if space.occupied else "green", lw=1))
    for candidate in detections:
        ax.add_patch(Polygon(vehicle_polygon(candidate.goal(candidate.axis_yaw, lot.vehicle), lot.vehicle.length, lot.vehicle.width, lot.vehicle.rear_overhang), fill=False, ec="blue", lw=2))
    xs = [s.center[0] for s in lot.spaces]
    ys = [s.center[1] for s in lot.spaces]
    ax.set_xlim(min(xs) - 12, max(xs) + 12)
    ax.set_ylim(min(ys) - 12, max(ys) + 12)
    ax.set_title("MetaDrive lot: stack map from MetaDrive LiDAR; green = free spaces, blue = detections")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv: list[str]) -> int:
    seeds = [int(a) for a in argv] or [0]
    totals = np.zeros(3, dtype=int)
    for seed in seeds:
        totals += check(seed, Path(f"artifacts/metadrive/perception_{seed}.png"))
    print(f"total: {totals[0]} in a free space, {totals[1]} not, {totals[2]} free spaces")
    return 1 if totals[1] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
