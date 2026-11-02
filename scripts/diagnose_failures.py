"""Run missions and classify why the failed ones did not park.

For every lot that ends without a correct park, each free bay is looked
up in the car's final map:

- ``unseen``: the car never saw the bay (exploration);
- ``seen, not detected``: seen but the detector did not accept it, split by
  how many neighbours are occupied (0, 1, 2) and bay kind;
- ``detected, rejected``: accepted but planning refused it ("no route");
- ``detected, not tried``: accepted at the end but never attempted.

A lot is classified by its most advanced free bay. Use development lots
only, never acceptance lots.

    python scripts/diagnose_failures.py 6000 6120 [--localization scan_matching] [--scenario mixed]
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_detector import _neighbours, _seen  # noqa: E402
from ihateparking.mission import MissionConfig, run_mission, scenario_lot  # noqa: E402
from ihateparking.slot_detector import detect_slots  # noqa: E402

STAGES = ("unseen", "seen, not detected", "detected, not tried", "detected, rejected")
POINT = re.compile(r"\(([-\d.]+), ([-\d.]+)\)")


def diagnose(args: tuple[int, str, str]) -> dict:
    seed, localization, scenario = args
    lot = scenario_lot(seed, scenario)
    result = run_mission(lot, MissionConfig(localization=localization), seed=seed)
    row = {"seed": seed, "outcome": result.outcome, "success": result.success, "bays": []}
    if result.success:
        return row
    states = result.grid.states()
    detections = detect_slots(result.grid, lot.vehicle)
    rejected = [tuple(map(float, POINT.search(e.message).groups())) for e in result.events if e.kind == "reject" and POINT.search(e.message)]
    for bay in lot.free_bays:
        flanks = min(2, sum(o.occupied for o in _neighbours(lot, bay)))
        if not _seen(result.grid, states, bay.slot):
            stage = "unseen"
        elif not any(math.dist(c.center, bay.slot.center) < 1.2 for c in detections):
            stage = "seen, not detected"
        elif any(math.dist(p, bay.slot.center) < 2.0 for p in rejected):
            stage = "detected, rejected"
        else:
            stage = "detected, not tried"
        row["bays"].append({"bay": bay.bay_id, "kind": bay.kind, "flanks": flanks, "stage": stage})
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=int)
    parser.add_argument("last", type=int)
    parser.add_argument("--localization", default="scan_matching")
    parser.add_argument("--scenario", default="mixed")
    parser.add_argument("--json", default="artifacts/mission/failures.json")
    args = parser.parse_args()
    jobs = [(seed, args.localization, args.scenario) for seed in range(args.first, args.last)]
    with ProcessPoolExecutor(max_workers=min(13, os.cpu_count() or 1)) as executor:
        rows = list(executor.map(diagnose, jobs))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    failed = [row for row in rows if not row["success"]]
    print(f"{len(rows) - len(failed)}/{len(rows)} parked correctly")
    lots, bays = Counter(), Counter()
    for row in failed:
        if not row["bays"]:
            lots[(row["outcome"], "no free bay")] += 1
            continue
        best = max(row["bays"], key=lambda b: STAGES.index(b["stage"]))
        lots[(row["outcome"], best["stage"])] += 1
        for bay in row["bays"]:
            bays[(bay["stage"], bay["kind"], bay["flanks"])] += 1
    print("failed lots by their most advanced free bay:")
    for key, count in lots.most_common():
        print(f"  {count:3d}  {key[0]:15s} {key[1]}")
    print("free bays in failed lots (stage, kind, occupied neighbours):")
    for key, count in sorted(bays.items(), key=lambda item: -item[1]):
        print(f"  {count:3d}  {key[0]:20s} {key[1]:13s} {key[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
