"""Bay-detector precision and recall in seconds, without running missions.

Each lot is mapped by driving straight along its start aisle (cached under
``artifacts/sweeps``); every detection is then scored against the ground
truth. Precision must stay at 100%: one false detection is a wrong park.
Recall is split by bay kind and by how many neighbours of the free bay are
occupied (0, 1 or 2), because each case uses different evidence.

    python scripts/eval_detector.py 6000 6120            # mixed lots
    python scripts/eval_detector.py 9000 9060 parallel   # parallel scenario
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import os
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ihateparking.geometry import Pose  # noqa: E402
from ihateparking.lidar import scan  # noqa: E402
from ihateparking.mission import evaluate_parking, scenario_lot  # noqa: E402
from ihateparking.occupancy_map import UNKNOWN, OccupancyGrid  # noqa: E402
from ihateparking.slot_detector import detect_slots  # noqa: E402
from ihateparking.world import ParkingWorld  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "artifacts" / "sweeps"


def aisle_sweep(seed: int, scenario: str):
    lot = scenario_lot(seed, scenario)
    path = CACHE / f"{scenario}_{seed}.pkl"
    if path.exists():
        return lot, pickle.loads(path.read_bytes())
    world = ParkingWorld(lot.bounds, list(lot.obstacles), lot.bays[0].slot, lot.start, lot.vehicle)
    grid = OccupancyGrid((lot.start.x, lot.start.y))
    rng = np.random.default_rng(seed)
    for d in np.arange(-30.0, 30.1, 1.0):
        pose = Pose(lot.start.x + d * math.cos(lot.start.yaw), lot.start.y + d * math.sin(lot.start.yaw), lot.start.yaw)
        if not world.is_collision(pose):
            grid.integrate(scan(pose, lot.obstacles, lot.vehicle, rng=rng))
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(grid))
    return lot, grid


def _neighbours(lot, bay):
    pitch = bay.slot.length if bay.kind == "parallel" else bay.slot.width / (math.sin(math.pi / 4) if bay.kind == "angled" else 1.0)
    return [o for o in lot.bays if o is not bay and o.kind == bay.kind and math.dist(o.slot.center, bay.slot.center) < pitch * 1.45]


def _seen(grid, states, slot) -> bool:
    us, vs = np.meshgrid(np.arange(-slot.length / 2 + 0.1, slot.length / 2 - 0.1, 0.1), np.arange(-slot.width / 2 + 0.1, slot.width / 2 - 0.1, 0.1))
    c, s = math.cos(slot.yaw), math.sin(slot.yaw)
    rows, cols = grid.to_cell((slot.center[0] + c * us - s * vs).ravel(), (slot.center[1] + s * us + c * vs).ravel())
    inside = (rows >= 0) & (rows < states.shape[0]) & (cols >= 0) & (cols < states.shape[1])
    return bool(inside.all() and np.mean(states[rows, cols] != UNKNOWN) >= 0.95)


def evaluate(args):
    seed, scenario = args
    lot, grid = aisle_sweep(seed, scenario)
    states = grid.states()
    detections = detect_slots(grid, lot.vehicle)
    good, false, errors = 0, [], []
    for candidate in detections:
        verdicts = [evaluate_parking(lot, candidate.goal(heading, lot.vehicle)) for heading in candidate.headings()]
        if any(v["in_bay"] for v in verdicts):
            good += 1
            hit = next(v for v in verdicts if v["in_bay"])
            errors.append((hit["bay_kind"], hit["center_error_m"]))
        else:
            v = verdicts[0]
            false.append((round(candidate.center[0], 1), round(candidate.center[1], 1), v.get("bay"), v.get("overlap")))
    recall = {}
    for bay in lot.free_bays:
        if not _seen(grid, states, bay.slot):
            continue
        flanks = min(2, sum(o.occupied for o in _neighbours(lot, bay)))
        found = any(math.dist(c.center, bay.slot.center) < 1.0 for c in detections)
        tally = recall.setdefault((bay.kind, flanks), [0, 0])
        tally[0] += found
        tally[1] += 1
    return seed, good, false, recall, errors


def main(argv: list[str]) -> int:
    first, last = int(argv[0]), int(argv[1])
    scenario = argv[2] if len(argv) > 2 else "mixed"
    good = bad = 0
    recall: dict = {}
    errors: dict = {}
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as executor:
        for seed, g, false, rec, errs in executor.map(evaluate, [(seed, scenario) for seed in range(first, last)]):
            good += g
            bad += len(false)
            if false:
                print("FALSE", seed, false)
            for kind, error in errs:
                errors.setdefault(kind, []).append(error)
            for key, (found, total) in rec.items():
                tally = recall.setdefault(key, [0, 0])
                tally[0] += found
                tally[1] += total
    print(f"detections {good + bad}: true {good}, false {bad}  precision {good / max(1, good + bad):.3f}")
    for key in sorted(recall):
        print(f"  recall {key[0]:13s} flanks={key[1]}  {recall[key][0]}/{recall[key][1]}")
    for kind in sorted(errors):
        values = np.array(errors[kind])
        print(f"  centre error {kind:13s} median {np.median(values):.2f} m  p95 {np.percentile(values, 95):.2f} m  (n={len(values)})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
