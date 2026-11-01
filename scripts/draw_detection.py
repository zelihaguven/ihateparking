"""Draw a lot's aisle-sweep map with the ground-truth bays and the detector's bays.

    python scripts/draw_detection.py 8061 [mixed|parallel] [x y radius]

Writes artifacts/detections/<scenario>_<seed>.png: the map (black occupied,
pale free, grey unknown), free truth bays green, occupied red, detections
blue (with their source rule), true or false.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_detector import aisle_sweep  # noqa: E402
from ihateparking.mission import evaluate_parking  # noqa: E402
from ihateparking.occupancy_map import FREE, OCCUPIED  # noqa: E402
from ihateparking.slot_detector import detect_slots  # noqa: E402


def _corners(center, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(center[0] + c * a - s * b, center[1] + s * a + c * b) for a, b in ((-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2))]


def main(argv: list[str]) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    seed = int(argv[0])
    scenario = argv[1] if len(argv) > 1 else "mixed"
    lot, grid = aisle_sweep(seed, scenario)
    states = grid.states()
    image = np.full(states.shape + (3,), 0.6)
    image[states == FREE] = (0.9, 0.93, 0.93)
    image[states == OCCUPIED] = (0.0, 0.0, 0.0)
    size = grid.config.size_m
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, origin="lower", extent=[grid.origin[0], grid.origin[0] + size, grid.origin[1], grid.origin[1] + size])
    for bay in lot.bays:
        ax.add_patch(Polygon(_corners(bay.slot.center, bay.slot.yaw, bay.slot.length, bay.slot.width), fill=False, ec="red" if bay.occupied else "green", lw=1))
    for candidate in detect_slots(grid, lot.vehicle):
        verdicts = [evaluate_parking(lot, candidate.goal(h, lot.vehicle)) for h in candidate.headings()]
        true = any(v["in_bay"] for v in verdicts)
        ax.add_patch(Polygon(_corners(candidate.center, candidate.axis_yaw, lot.vehicle.length, lot.vehicle.width), fc="#3a86ff55" if true else "#ff006e88", ec="blue" if true else "magenta", lw=2))
        ax.annotate(f"{getattr(candidate, 'source', '')} {'ok' if true else 'FALSE'}", candidate.center, fontsize=7, color="blue" if true else "magenta")
    ax.plot(lot.start.x, lot.start.y, "k^")
    if len(argv) >= 5:
        x, y, r = float(argv[2]), float(argv[3]), float(argv[4])
        ax.set_xlim(x - r, x + r)
        ax.set_ylim(y - r, y + r)
    ax.set_title(f"{scenario} {seed}", fontsize=9)
    out = Path(__file__).resolve().parents[1] / "artifacts" / "detections" / f"{scenario}_{seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
