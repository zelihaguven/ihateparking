"""Run the full parking mission inside MetaDrive (Bullet physics, MetaDrive LiDAR).

The mission loop, map, detector, planner and tracker are the stack's own,
unchanged; MetaDrive is the world (``MetaDrivePlant``). Run in the
MetaDrive environment:

    .venv-metadrive/bin/python scripts/metadrive_mission.py [--localization truth|odometry|scan_matching] seed ...
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihateparking.metadrive_bridge import MetaDriveLot, MetaDrivePlant  # noqa: E402
from ihateparking.mission import MissionConfig, run_mission  # noqa: E402
from ihateparking.occupancy_map import FREE, OCCUPIED  # noqa: E402


def draw(lot: MetaDriveLot, result, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    from ihateparking.geometry import vehicle_polygon

    grid = result.grid
    states = grid.states()
    image = np.full(states.shape + (3,), 0.6)
    image[states == FREE] = (0.88, 0.92, 0.92)
    image[states == OCCUPIED] = (0.0, 0.0, 0.0)
    size = grid.config.size_m
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image, origin="lower", extent=[grid.origin[0], grid.origin[0] + size, grid.origin[1], grid.origin[1] + size])
    for space in lot.spaces:
        c, s = math.cos(space.yaw), math.sin(space.yaw)
        corners = [(space.center[0] + c * a - s * b, space.center[1] + s * a + c * b) for a, b in ((-space.length / 2, -space.width / 2), (space.length / 2, -space.width / 2), (space.length / 2, space.width / 2), (-space.length / 2, space.width / 2))]
        ax.add_patch(Polygon(corners, fill=False, ec="red" if space.occupied else "green", lw=1))
    ax.plot([p.x for p in result.trace], [p.y for p in result.trace], color="teal", lw=1.5)
    final = result.trace[-1]
    ax.add_patch(Polygon(vehicle_polygon(final, lot.vehicle.length, lot.vehicle.width, lot.vehicle.rear_overhang), fc="#4cc9f055", ec="blue", lw=2))
    xs = [s.center[0] for s in lot.spaces] + [p.x for p in result.trace]
    ys = [s.center[1] for s in lot.spaces] + [p.y for p in result.trace]
    ax.set_xlim(min(xs) - 8, max(xs) + 8)
    ax.set_ylim(min(ys) - 8, max(ys) + 8)
    ax.set_aspect("equal")
    ax.set_title(f"MetaDrive: {result.outcome}  {result.evaluation}", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", type=int, nargs="+")
    parser.add_argument("--localization", default="truth")
    parser.add_argument("--spaces-per-side", type=int, default=6)
    args = parser.parse_args()
    failures = 0
    for seed in args.seeds:
        lot = MetaDriveLot(seed, spaces_per_side=args.spaces_per_side)
        try:
            config = MissionConfig(localization=args.localization, max_time_s=300.0)
            result = run_mission(None, config, seed=seed, plant=MetaDrivePlant(lot))
            free = sum(not s.occupied for s in lot.spaces)
            print(f"seed {seed}: {result.outcome:15s} success={result.success} {result.time_s:6.1f} s {result.distance_m:6.1f} m  free spaces {free}  {result.evaluation}")
            for event in result.events[-4:]:
                print(f"    {event.time_s:7.1f} s  {event.kind:8s} {event.message}")
            draw(lot, result, Path(f"artifacts/metadrive/mission_{seed}.png"))
            failures += not result.success
        finally:
            lot.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
