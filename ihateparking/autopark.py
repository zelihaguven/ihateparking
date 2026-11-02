"""Park in a chosen bay of a parking lot, deciding the manoeuvre autonomously.

The caller names only the target bay (or none, to let the vehicle pick the
nearest free one). The vehicle then:

1. refuses an occupied or unknown bay without moving;
2. enumerates the final headings the bay allows - nose-in and tail-in for
   perpendicular and angled bays, both directions along the kerb for
   parallel bays - and drops any whose footprint does not fit;
3. plans every remaining option with Hybrid A* and ranks the plans: those
   ending exactly on the goal first, then by cost (length, reverse driving
   and gear changes);
4. executes the best plan in closed loop with the independent
   collision guard, falling back to the next option if execution fails.

    python -m ihateparking.autopark --seed 3 --bay B6
    python -m ihateparking.autopark --seed 3            # nearest free bay
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import time
from typing import Optional

from .controller import ControllerConfig, SimulationResult, simulate_closed_loop
from .geometry import Pose, angle_error, distance, wrap_angle
from .holonomic_heuristic import build_distance_field
from .parking_lot import DEFAULT_START, Bay, ParkingLot, build_lot
from .planner import HybridAStar, PlannerConfig, PlanResult
from .world import ParkingSlot, ParkingWorld

DEFAULT_MAX_EXPANSIONS = 30_000
MAX_BAYS_TRIED = 6


@dataclass(frozen=True)
class ApproachOption:
    label: str
    goal: Pose


@dataclass
class OptionOutcome:
    label: str
    goal: Pose
    fits: bool
    planned: bool = False
    cost: float = math.inf
    expansions: int = 0
    plan_time_s: float = 0.0
    message: str = ""
    plan: Optional[PlanResult] = field(default=None, repr=False)

    @property
    def exact(self) -> bool:
        return "Reeds-Shepp" in self.message


@dataclass
class AutoparkResult:
    bay_id: str
    decision: str  # PARKED, REFUSED or FAILED
    reason: str
    world: ParkingWorld
    options: list[OptionOutcome]
    chosen: Optional[OptionOutcome] = None
    simulation: Optional[SimulationResult] = None

    @property
    def parked(self) -> bool:
        return self.decision == "PARKED"

    def summary(self) -> dict:
        final = self.simulation.poses[-1] if self.simulation else None
        goal = self.chosen.goal if self.chosen else None
        return {
            "bay": self.bay_id,
            "decision": self.decision,
            "reason": self.reason,
            "chosen_manoeuvre": self.chosen.label if self.chosen else None,
            "options": [
                {
                    "manoeuvre": item.label,
                    "fits": item.fits,
                    "planned": item.planned,
                    "cost": None if math.isinf(item.cost) else round(item.cost, 2),
                    "exact_goal": item.planned and item.exact,
                    "expansions": item.expansions,
                    "plan_time_s": round(item.plan_time_s, 2),
                    "message": item.message,
                }
                for item in self.options
            ],
            "gear_changes": _gear_changes(self.simulation.gears) if self.simulation else None,
            "manoeuvre_time_s": round(self.simulation.time, 1) if self.simulation else None,
            "final_error_m": round(distance(final, goal), 3) if final and goal else None,
            "final_yaw_error_deg": round(abs(math.degrees(angle_error(goal.yaw, final.yaw))), 2) if final and goal else None,
            "collisions": self.simulation.collisions if self.simulation else 0,
        }


def _time_budget(controller: ControllerConfig, plan: PlanResult) -> ControllerConfig:
    """Lot routes are long; give the tracker time in proportion to the plan.

    The ramp-down before every cusp makes average speed well below the
    cruise speed, hence the factor of three plus a fixed allowance.
    """
    needed = 3.0 * plan.path_length / controller.speed + 20.0
    return replace(controller, max_time=max(controller.max_time, needed))


def _gear_changes(gears: list[int]) -> int:
    moving = [gear for gear in gears if gear != 0]
    return sum(1 for a, b in zip(moving, moving[1:]) if a != b)


def lot_world(lot: ParkingLot, bay: Bay, start: Pose) -> ParkingWorld:
    """Collision world for one attempt: every parked car and kerb is an obstacle."""
    return ParkingWorld(lot.bounds, lot.obstacles, bay.slot, start, lot.vehicle)


def approach_options(bay: Bay, lot: ParkingLot) -> list[ApproachOption]:
    """Final headings the bay allows; the planner decides how to get there."""
    slot = bay.slot

    def goal(yaw: float) -> Pose:
        return ParkingSlot(slot.center, slot.length, slot.width, wrap_angle(yaw)).goal_pose(lot.vehicle)

    if bay.kind == "parallel":
        return [
            ApproachOption("parallel, facing east", goal(slot.yaw)),
            ApproachOption("parallel, facing west", goal(slot.yaw + math.pi)),
        ]
    return [
        ApproachOption("nose-in", goal(slot.yaw)),
        ApproachOption("tail-in", goal(slot.yaw + math.pi)),
    ]


def _describe(option: OptionOutcome) -> str:
    plan = option.plan
    if plan is None:
        return option.label
    final_gear = next((gear for gear in reversed(plan.gears) if gear != 0), 1)
    entry = "reversing in" if final_gear < 0 else "driving in forwards"
    return f"{option.label} ({entry}, {_gear_changes(plan.gears)} gear changes)"


def park_in_bay(
    lot: ParkingLot,
    bay_id: str,
    start: Pose = DEFAULT_START,
    planner: PlannerConfig | None = None,
    controller: ControllerConfig | None = None,
) -> AutoparkResult:
    bay = lot.bay(bay_id)
    world = lot_world(lot, bay, start)
    if not lot.is_free(bay_id):
        return AutoparkResult(bay_id, "REFUSED", f"bay {bay_id} is occupied; the vehicle does not move", world, [])
    if world.is_collision(start):
        return AutoparkResult(bay_id, "REFUSED", "start pose is in collision", world, [])

    planner = planner or PlannerConfig(max_expansions=DEFAULT_MAX_EXPANSIONS)
    outcomes: list[OptionOutcome] = []
    for option in approach_options(bay, lot):
        outcome = OptionOutcome(option.label, option.goal, fits=not world.is_collision(option.goal))
        if not outcome.fits:
            outcome.message = "footprint does not fit with the safety margin"
            outcomes.append(outcome)
            continue
        started = time.perf_counter()
        plan = HybridAStar(world, planner).plan(start=start, goal=option.goal)
        outcome.plan_time_s = time.perf_counter() - started
        outcome.planned = plan.success
        outcome.expansions = plan.expanded
        outcome.message = plan.message
        if plan.success:
            outcome.cost = plan.cost
            outcome.plan = plan
        outcomes.append(outcome)

    # Prefer plans that end exactly on the goal over tolerance-only plans:
    # the latter can leave the car up to 0.35 m / 12 degrees off.
    planned = sorted((item for item in outcomes if item.planned), key=lambda item: (not item.exact, item.cost))
    if not planned:
        reason = "no collision-free route found for any approach; the vehicle does not move"
        return AutoparkResult(bay_id, "FAILED", reason, world, outcomes)

    controller = controller or ControllerConfig()
    last_simulation: Optional[SimulationResult] = None
    for option in planned:
        # The world's goal is used by the controller's final check.
        attempt_world = ParkingWorld(
            world.bounds, world.obstacles,
            ParkingSlot(bay.slot.center, bay.slot.length, bay.slot.width, option.goal.yaw),
            start, world.vehicle,
        )
        simulation = simulate_closed_loop(attempt_world, option.plan, _time_budget(controller, option.plan))
        last_simulation = simulation
        if simulation.success:
            others = [item.label for item in planned if item is not option]
            reason = f"parked {_describe(option)}"
            if others:
                reason += f"; preferred over {', '.join(others)}"
            return AutoparkResult(bay_id, "PARKED", reason, attempt_world, outcomes, option, simulation)
    return AutoparkResult(
        bay_id, "FAILED", f"execution did not reach the bay: {last_simulation.message}",
        world, outcomes, planned[-1], last_simulation,
    )


def rank_free_bays(lot: ParkingLot, start: Pose = DEFAULT_START) -> list[tuple[float, Bay]]:
    """Free bays ordered by obstacle-aware driving distance from the start."""
    probe = ParkingWorld(lot.bounds, lot.obstacles, lot.bays[0].slot, start, lot.vehicle)
    field_from_start = build_distance_field(probe, (start.x, start.y))
    ranked = []
    for bay in lot.free_bays:
        reach = field_from_start.lookup(*bay.center)
        if math.isinf(reach):
            reach = distance(start, Pose(*bay.center, 0.0)) + 100.0
        ranked.append((reach, bay))
    ranked.sort(key=lambda item: item[0])
    return ranked


def find_and_park(lot: ParkingLot, start: Pose = DEFAULT_START, max_bays: int = MAX_BAYS_TRIED, **kwargs) -> list[AutoparkResult]:
    """Try free bays nearest-first until one succeeds; returns every attempt."""
    attempts = []
    for _, bay in rank_free_bays(lot, start)[:max_bays]:
        result = park_in_bay(lot, bay.bay_id, start, **kwargs)
        attempts.append(result)
        if result.parked:
            break
    return attempts


def _pose_dict(pose: Pose) -> dict:
    return {"x": round(pose.x, 4), "y": round(pose.y, 4), "yaw": round(pose.yaw, 5)}


def lot_payload(lot: ParkingLot, start: Pose = DEFAULT_START) -> dict:
    """JSON-ready lot geometry for a browser view."""
    return {
        "seed": lot.seed,
        "bounds": list(lot.bounds),
        "start": _pose_dict(start),
        "vehicle": {
            "length": lot.vehicle.length,
            "width": lot.vehicle.width,
            "rear_overhang": lot.vehicle.rear_overhang,
        },
        "bays": [
            {
                "id": bay.bay_id,
                "kind": bay.kind,
                "occupied": bay.bay_id in lot.occupied,
                "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in bay.slot.polygon()],
                "center": list(bay.center),
            }
            for bay in lot.bays
        ],
        "obstacles": [
            {"name": item.name, "polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in item.polygon()]}
            for item in lot.obstacles
        ],
    }


def result_payload(result: AutoparkResult) -> dict:
    """Summary plus the plan and executed trace of one attempt."""
    payload = result.summary()
    payload["goal"] = _pose_dict(result.chosen.goal) if result.chosen else None
    payload["plan"] = (
        [_pose_dict(pose) for pose in result.chosen.plan.poses]
        if result.chosen is not None and result.chosen.plan is not None else []
    )
    if result.simulation is not None:
        simulation = result.simulation
        gears = [simulation.gears[0] if simulation.gears else 0, *simulation.gears]
        payload["trace"] = [
            {**_pose_dict(pose), "gear": gear}
            for pose, gear in zip(simulation.poses, gears)
        ]
        payload["dt_s"] = round(simulation.time / max(1, len(simulation.poses) - 1), 4)
    else:
        payload["trace"] = []
        payload["dt_s"] = None
    return payload


def run_autopark(seed: int, occupancy: float = 0.5, bay_id: str | None = None) -> dict:
    """Build the lot, park in ``bay_id`` (or the nearest free bay) and serialize."""
    lot = build_lot(seed, occupancy)
    attempts = [park_in_bay(lot, bay_id)] if bay_id else find_and_park(lot)
    return {
        "lot": lot_payload(lot),
        "requested_bay": bay_id,
        "attempts": [result_payload(item) for item in attempts],
        "decision": attempts[-1].decision if attempts else "NO_FREE_BAY",
        "research_note": "Synthetic lot with ground-truth parked cars; not a real-vehicle interface.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Park in a bay of a synthetic lot; the vehicle chooses the manoeuvre")
    parser.add_argument("--seed", type=int, default=3, help="lot occupancy seed")
    parser.add_argument("--occupancy", type=float, default=0.5)
    parser.add_argument("--bay", help="target bay id, e.g. B6; omit to pick the nearest free bay")
    parser.add_argument("--list", action="store_true", help="list bays and occupancy, then exit")
    parser.add_argument("--max-expansions", type=int, default=DEFAULT_MAX_EXPANSIONS)
    parser.add_argument("--output", default="artifacts/autopark/autopark.png", help="figure path; '' to skip")
    args = parser.parse_args(argv)

    lot = build_lot(args.seed, args.occupancy)
    if args.list:
        for bay in lot.bays:
            status = "occupied" if bay.bay_id in lot.occupied else "free"
            print(f"{bay.bay_id:4s} {bay.kind:14s} {status}")
        return 0

    planner = PlannerConfig(max_expansions=args.max_expansions)
    if args.bay:
        attempts = [park_in_bay(lot, args.bay, planner=planner)]
    else:
        attempts = find_and_park(lot, planner=planner)
        if not attempts:
            print("no free bay in this lot")
            return 1
    for attempt in attempts:
        print(json.dumps(attempt.summary(), indent=2))
    result = attempts[-1]
    if args.output:
        from .autopark_render import render_autopark

        print(f"[figure] {render_autopark(lot, result, Path(args.output))}")
    return 0 if result.parked else 2


if __name__ == "__main__":
    raise SystemExit(main())
