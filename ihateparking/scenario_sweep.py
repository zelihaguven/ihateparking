"""Automated scenario sweep: plan and execute every manoeuvre over many seeds.

Each case generates a seeded scenario, plans with Hybrid A*, executes the plan
in the closed-loop simulator and records the README experiment-protocol
metrics. The sweep is deterministic for a given seed range and configuration;
only the wall-clock planning times vary between machines.

    python -m ihateparking.scenario_sweep --seeds 100 --workers 8
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable, Optional

from .controller import ControllerConfig, simulate_closed_loop
from .geometry import angle_error, distance, polygon_distance
from .planner import HybridAStar, PlannerConfig
from .scenario_generator import ScenarioConfig, ScenarioGenerator, ScenarioKind
from .world import ParkingWorld

DEFAULT_MAX_EXPANSIONS = 30_000
# Clearance is measured on every Nth executed pose; the control step is 5 cm.
CLEARANCE_STRIDE = 2


@dataclass(frozen=True)
class SweepCase:
    kind: str
    seed: int
    planner: PlannerConfig = field(default_factory=lambda: PlannerConfig(max_expansions=DEFAULT_MAX_EXPANSIONS))
    controller: ControllerConfig = field(default_factory=ControllerConfig)


@dataclass(frozen=True)
class CaseResult:
    kind: str
    seed: int
    outcome: str
    planner_success: bool
    planner_message: str
    expansions: int
    plan_time_s: float
    path_length_m: Optional[float]
    planned_gear_changes: Optional[int]
    parked: bool
    collisions: int
    controller_message: Optional[str]
    manoeuvre_time_s: Optional[float]
    executed_gear_changes: Optional[int]
    final_position_error_m: Optional[float]
    final_yaw_error_deg: Optional[float]
    min_obstacle_clearance_m: Optional[float]


def gear_changes(gears: Iterable[int]) -> int:
    """Count direction reversals, ignoring the 0 placeholder of a start pose."""
    moving = [gear for gear in gears if gear != 0]
    return sum(1 for a, b in zip(moving, moving[1:]) if a != b)


def min_obstacle_clearance(world: ParkingWorld, poses: list, stride: int = CLEARANCE_STRIDE) -> float:
    """Smallest footprint-to-obstacle distance (m) along an executed trace."""
    if not poses:
        return math.inf
    sampled = poses[::stride]
    if sampled[-1] is not poses[-1]:
        sampled.append(poses[-1])
    best = math.inf
    for pose in sampled:
        footprint = world.vehicle_polygon(pose)
        for obstacle in world.obstacles:
            best = min(best, polygon_distance(footprint, obstacle.polygon()))
    return best


def run_case(case: SweepCase) -> CaseResult:
    world = ScenarioGenerator().generate(ScenarioConfig(seed=case.seed, kind=ScenarioKind(case.kind))).to_world()
    started = time.perf_counter()
    plan = HybridAStar(world, case.planner).plan()
    plan_time = time.perf_counter() - started
    if not plan.success:
        return CaseResult(
            case.kind, case.seed, "no_plan", False, plan.message, plan.expanded, plan_time,
            None, None, False, 0, None, None, None, None, None, None,
        )

    simulation = simulate_closed_loop(world, plan, case.controller)
    final = simulation.poses[-1]
    goal = world.slot.goal_pose(world.vehicle)
    if simulation.success:
        outcome = "parked"
    elif simulation.collisions:
        outcome = "guard_stop"
    else:
        outcome = "missed_goal"
    return CaseResult(
        kind=case.kind,
        seed=case.seed,
        outcome=outcome,
        planner_success=True,
        planner_message=plan.message,
        expansions=plan.expanded,
        plan_time_s=plan_time,
        path_length_m=plan.path_length,
        planned_gear_changes=gear_changes(plan.gears),
        parked=simulation.success,
        collisions=simulation.collisions,
        controller_message=simulation.message,
        manoeuvre_time_s=simulation.time,
        executed_gear_changes=gear_changes(simulation.gears),
        final_position_error_m=distance(final, goal),
        final_yaw_error_deg=abs(math.degrees(angle_error(goal.yaw, final.yaw))),
        min_obstacle_clearance_m=min_obstacle_clearance(world, simulation.poses),
    )


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _median(values: list[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    """Aggregate per manoeuvre kind and overall."""
    groups: dict[str, list[CaseResult]] = {}
    for result in results:
        groups.setdefault(result.kind, []).append(result)
    summary = {kind: _summarize_group(items) for kind, items in groups.items()}
    summary["all"] = _summarize_group(results)
    return summary


def _summarize_group(items: list[CaseResult]) -> dict[str, Any]:
    executed = [item for item in items if item.planner_success]
    parked = [item for item in items if item.parked]
    plan_times = [item.plan_time_s for item in items]
    clearances = [item.min_obstacle_clearance_m for item in executed if item.min_obstacle_clearance_m is not None]
    outcomes: dict[str, int] = {}
    for item in items:
        outcomes[item.outcome] = outcomes.get(item.outcome, 0) + 1
    count = len(items)
    return {
        "cases": count,
        "success_rate": len(parked) / count if count else 0.0,
        "planner_success_rate": len(executed) / count if count else 0.0,
        "outcomes": outcomes,
        "collisions": sum(item.collisions for item in items),
        "final_position_error_m": {
            "median": _median([item.final_position_error_m for item in parked]),
            "p95": _percentile([item.final_position_error_m for item in parked], 0.95),
        },
        "final_yaw_error_deg": {
            "median": _median([item.final_yaw_error_deg for item in parked]),
            "p95": _percentile([item.final_yaw_error_deg for item in parked], 0.95),
        },
        "min_obstacle_clearance_m": min(clearances) if clearances else None,
        "manoeuvre_time_s": {"median": _median([item.manoeuvre_time_s for item in parked])},
        "path_length_m": {"median": _median([item.path_length_m for item in parked])},
        "planned_gear_changes": {"median": _median([item.planned_gear_changes for item in parked])},
        "plan_time_s": {"median": _median(plan_times), "p95": _percentile(plan_times, 0.95), "max": max(plan_times, default=None)},
        "expansions": {"median": _median([item.expansions for item in items])},
    }


def run_sweep(
    seeds: int,
    seed_offset: int = 0,
    kinds: Iterable[ScenarioKind] | None = None,
    planner: PlannerConfig | None = None,
    controller: ControllerConfig | None = None,
    workers: int = 1,
) -> list[CaseResult]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    planner = planner or PlannerConfig(max_expansions=DEFAULT_MAX_EXPANSIONS)
    controller = controller or ControllerConfig()
    selected = tuple(kinds or tuple(ScenarioKind))
    cases = [
        SweepCase(kind.value, seed, planner, controller)
        for kind in selected
        for seed in range(seed_offset, seed_offset + seeds)
    ]
    if workers <= 1:
        return [run_case(case) for case in cases]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run_case, cases, chunksize=4))


def _format(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def markdown_table(summary: dict[str, Any]) -> str:
    header = (
        "| Manoeuvre | Parked | Planned | Collisions | Final pos. err. (median / p95) "
        "| Final yaw err. (median / p95) | Min clearance | Median time | Plan time (median / max) |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for kind, item in summary.items():
        rows.append(
            f"| {kind} | {item['success_rate']:.0%} | {item['planner_success_rate']:.0%} | {item['collisions']} "
            f"| {_format(item['final_position_error_m']['median'], 3)} / {_format(item['final_position_error_m']['p95'], 3)} m "
            f"| {_format(item['final_yaw_error_deg']['median'], 1)} / {_format(item['final_yaw_error_deg']['p95'], 1)}° "
            f"| {_format(item['min_obstacle_clearance_m'], 3, ' m')} "
            f"| {_format(item['manoeuvre_time_s']['median'], 1, ' s')} "
            f"| {_format(item['plan_time_s']['median'], 2)} / {_format(item['plan_time_s']['max'], 1)} s |"
        )
    return header + "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan and execute every parking manoeuvre over a seed range")
    parser.add_argument("--seeds", type=int, default=100, help="seeds per manoeuvre kind")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--kinds", nargs="*", choices=[kind.value for kind in ScenarioKind], help="default: all six")
    parser.add_argument("--max-expansions", type=int, default=DEFAULT_MAX_EXPANSIONS)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", default="artifacts/scenario_sweep.json")
    args = parser.parse_args(argv)

    kinds = [ScenarioKind(kind) for kind in args.kinds] if args.kinds else None
    started = time.perf_counter()
    results = run_sweep(
        args.seeds,
        args.seed_offset,
        kinds,
        PlannerConfig(max_expansions=args.max_expansions),
        workers=args.workers,
    )
    summary = summarize(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "seeds_per_kind": args.seeds,
        "seed_offset": args.seed_offset,
        "planner": asdict(PlannerConfig(max_expansions=args.max_expansions)),
        "controller": asdict(ControllerConfig()),
        "summary": summary,
        "cases": [asdict(result) for result in results],
    }, indent=2), encoding="utf-8")
    print(markdown_table(summary))
    print(f"[sweep] {len(results)} cases in {time.perf_counter() - started:.1f}s -> {output}")
    return 0 if all(result.collisions == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
