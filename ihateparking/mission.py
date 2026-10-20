"""Park in a lot the vehicle has never seen, using only its own LiDAR.

The loop is sense -> map -> detect -> decide -> plan -> drive:

* every ``scan_period_s`` the simulated LiDAR scans and the occupancy grid
  is updated;
* if the grid shows a verified bay (``slot_detector``), the vehicle plans
  nose-in and tail-in on the *map* and drives the best plan;
* otherwise it explores: it picks the frontier (free cells next to unknown
  ones) with the best size-to-distance ratio and drives part of the way,
  then re-decides with the updated map;
* a plan is re-checked against the map after every scan and replanned if
  newly seen obstacles block it.

The ground-truth lot is used only to simulate the sensor, to run the
independent collision guard and, after the mission, to score where the car
ended up. The stack itself never reads it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np
from scipy import ndimage

from .controller import ControllerConfig, PlanTracker
from .geometry import Pose, distance, wrap_angle
from .holonomic_heuristic import build_distance_field
from .lidar import LidarConfig, scan
from .occupancy_map import FREE, OCCUPIED, UNKNOWN, GridConfig, MapWorld, OccupancyGrid
from .planner import HybridAStar, PlannerConfig, PlanResult
from .procedural_lot import ProceduralLot
from .slot_detector import SlotCandidate, detect_slots
from .world import ParkingWorld


@dataclass(frozen=True)
class MissionConfig:
    dt: float = 0.05
    scan_period_s: float = 0.2
    max_time_s: float = 420.0
    explore_horizon_m: float = 6.0
    park_planner: PlannerConfig = PlannerConfig(max_expansions=12_000)
    # Exploration needs reach, not centimetre accuracy: a coarser lattice.
    explore_planner: PlannerConfig = PlannerConfig(
        xy_resolution=0.5,
        yaw_bins=36,
        primitive_length=0.8,
        primitive_sample=0.2,
        max_expansions=4_000,
        goal_position_tolerance=1.2,
        goal_yaw_tolerance=math.pi,
        exact_goal_extra_expansions=200,
        analytic_radius=4.0,
        analytic_interval=15,
    )
    controller: ControllerConfig = ControllerConfig(speed=0.8)
    lidar: LidarConfig = LidarConfig()
    grid: GridConfig = GridConfig()
    candidates_per_decision: int = 3
    frontier_min_cells: int = 12
    frontier_candidates: int = 12
    frontier_min_gain: int = 40
    # Clearance from occupied cells at an exploration goal's footprint centre:
    # enough to keep goals in aisles rather than inside bays.
    frontier_clearance_m: float = 2.4
    recovery_expansions: int = 12_000
    max_recoveries: int = 3
    frontier_max_visits: int = 3
    inspection_candidates: int = 2
    inspection_standoff_m: float = 3.5
    escape_margin_m: float = 0.03
    frontier_attempts: int = 6
    lookahead_check_m: float = 5.0


@dataclass
class MissionEvent:
    time_s: float
    kind: str
    message: str


@dataclass
class MissionResult:
    outcome: str  # PARKED, NO_SPACE_FOUND, TIMEOUT, GUARD_STOP
    success: bool
    message: str
    trace: list[Pose]
    gears: list[int]
    events: list[MissionEvent]
    time_s: float
    distance_m: float
    collisions: int
    target: Optional[SlotCandidate]
    grid: OccupancyGrid
    evaluation: dict = field(default_factory=dict)


def _truth_world(lot: ProceduralLot) -> ParkingWorld:
    any_slot = lot.bays[0].slot if lot.bays else None
    return ParkingWorld(lot.bounds, list(lot.obstacles), any_slot, lot.start, lot.vehicle)


def _best_plan(world: MapWorld, start: Pose, candidate: SlotCandidate, config: PlannerConfig) -> Optional[tuple[PlanResult, float]]:
    """Plan both final headings; prefer exact-goal plans, then cost."""
    best = None
    for heading in candidate.headings():
        goal = candidate.goal(heading, world.vehicle)
        if world.is_collision(goal):
            continue
        plan = HybridAStar(world, config).plan(start=start, goal=goal)
        if not plan.success:
            continue
        exact = "Reeds-Shepp" in plan.message
        key = (not exact, plan.cost)
        if best is None or key < best[0]:
            best = (key, plan, heading)
    if best is None:
        return None
    return best[1], best[2]


def _visible_unknown(grid: OccupancyGrid, states: np.ndarray, origin: tuple[float, float], max_range: float, beams: int = 180) -> int:
    """Unknown cells a scan from ``origin`` would reveal, given the current map.

    Beams stop at the first occupied cell; cells beyond the grid count as
    blocked. This is the information gain of a candidate viewpoint.
    """
    res = grid.config.resolution
    angles = np.linspace(0.0, 2 * math.pi, beams, endpoint=False)
    steps = np.arange(res, max_range, res)
    xs = origin[0] + np.cos(angles)[:, None] * steps
    ys = origin[1] + np.sin(angles)[:, None] * steps
    rows, cols = grid.to_cell(xs, ys)
    inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    cells = np.full(xs.shape, OCCUPIED, dtype=np.uint8)
    cells[inside] = states[rows[inside], cols[inside]]
    blocked = np.maximum.accumulate(cells == OCCUPIED, axis=1)
    seen = (cells == UNKNOWN) & ~blocked
    return int(np.unique(rows[seen] * grid.shape[1] + cols[seen]).size)


def _frontier_goal(grid: OccupancyGrid, world: MapWorld, pose: Pose, config: MissionConfig, avoid: list[tuple[float, float]]) -> Optional[tuple[Pose, tuple[float, float]]]:
    """Next best view: the frontier viewpoint revealing most unknown space per metre.

    For the largest frontier clusters, take the nearest roomy cell as the
    viewpoint, count the unknown cells a scan from there would reveal, and
    divide by the driving distance.
    """
    states = world.states
    frontier = (states == FREE) & ndimage.binary_dilation(states == UNKNOWN, iterations=1)
    labels, count = ndimage.label(frontier, structure=np.ones((3, 3)))
    if count == 0:
        return None
    sizes = ndimage.sum(frontier, labels, index=np.arange(1, count + 1))
    roomy_rows, roomy_cols = np.nonzero(world.distance >= config.frontier_clearance_m)
    if roomy_rows.size == 0:
        return None
    roomy_x, roomy_y = grid.cell_center(roomy_rows, roomy_cols)
    field_from_pose = build_distance_field(world, (pose.x, pose.y), resolution=0.5, inflation=world.vehicle.width / 2)
    best = None
    for label in np.argsort(-sizes)[: config.frontier_candidates] + 1:
        if sizes[label - 1] < config.frontier_min_cells:
            continue
        rows, cols = np.nonzero(labels == label)
        fx, fy = grid.cell_center(rows, cols)
        centroid = (float(fx.mean()), float(fy.mean()))
        nearest = int(np.argmin((roomy_x - centroid[0]) ** 2 + (roomy_y - centroid[1]) ** 2))
        target = (float(roomy_x[nearest]), float(roomy_y[nearest]))
        if math.dist(target, centroid) > 6.0:
            continue
        if any(math.dist(centroid, other) < 3.0 or math.dist(target, other) < 2.5 for other in avoid):
            continue
        travel = field_from_pose.lookup(*target)
        if math.isinf(travel) or travel < 2.0:
            continue
        gain = _visible_unknown(grid, states, target, config.lidar.max_range)
        if gain < config.frontier_min_gain:
            continue
        score = gain / (travel + 8.0)
        if best is None or score > best[0]:
            best = (score, target, centroid)
    if best is None:
        return None
    _, target, centroid = best
    # Arrive heading the way the car is travelling (no U-turn needed);
    # otherwise the nearest free heading.
    facing = math.atan2(target[1] - pose.y, target[0] - pose.x)
    offset = world.vehicle.length / 2 - world.vehicle.rear_overhang
    for turn in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
        heading = facing + math.radians(turn)
        # Rear-axle pose that puts the footprint centre on the chosen cell.
        goal = Pose(target[0] - offset * math.cos(heading), target[1] - offset * math.sin(heading), wrap_angle(heading))
        if not world.is_collision(goal):
            return goal, target
    return None, target


def _inspection_plan(grid: OccupancyGrid, world: MapWorld, pose: Pose, config: MissionConfig, inspected: list[tuple[float, float]], rejected: list[tuple[float, float]]) -> Optional[PlanResult]:
    """Plan to a viewpoint in front of the nearest promising, uninspected gap."""
    vehicle = world.vehicle
    offset = vehicle.length / 2 - vehicle.rear_overhang
    gaps = [
        gap for gap in detect_slots(grid, vehicle, tentative=True)
        if gap.row == "side_by_side"
        and all(math.dist(gap.center, other) > 1.5 for other in inspected + rejected)
    ]
    gaps.sort(key=lambda gap: math.dist(gap.center, (pose.x, pose.y)))
    for gap in gaps[: config.inspection_candidates]:
        inspected.append(gap.center)
        into = np.array([math.cos(gap.axis_yaw), math.sin(gap.axis_yaw)])
        view = np.array(gap.center) - into * (vehicle.length / 2 + config.inspection_standoff_m)
        # Face into the gap if there is room; otherwise stand along the aisle.
        for heading in (gap.axis_yaw, gap.axis_yaw + math.pi / 2, gap.axis_yaw - math.pi / 2):
            goal = Pose(view[0] - offset * math.cos(heading), view[1] - offset * math.sin(heading), wrap_angle(heading))
            if world.is_collision(goal):
                continue
            plan = HybridAStar(world, config.explore_planner).plan(start=pose, goal=goal)
            if plan.success:
                return plan
            break
    return None


def _recover(grid: OccupancyGrid, world: MapWorld, pose: Pose, config: MissionConfig, unreachable: list[tuple[float, float]]) -> tuple[Optional[PlanResult], Optional[Pose]]:
    """Get out of a tight spot: a bigger search to the best frontier, then
    a short move to the most open nearby place."""
    # The fine lattice finds the tight multi-point turns the coarse one misses.
    bigger = PlannerConfig(
        max_expansions=config.recovery_expansions,
        goal_position_tolerance=config.explore_planner.goal_position_tolerance,
        goal_yaw_tolerance=math.pi,
    )
    found = _frontier_goal(grid, world, pose, config, [])
    if found is not None and found[0] is not None:
        plan = HybridAStar(world, bigger).plan(start=pose, goal=found[0])
        if plan.success:
            return plan, found[0]
    return _open_space_plan(grid, world, world, pose, config, bigger)


def _open_space_plan(grid: OccupancyGrid, plan_world: MapWorld, check_world: MapWorld, pose: Pose, config: MissionConfig, planner: PlannerConfig) -> tuple[Optional[PlanResult], Optional[Pose]]:
    """Short move to the most open reachable place within 8 m.

    Goals must be free in ``check_world`` (full margin); the route is planned
    in ``plan_world``, which may use a reduced margin to get the car out.
    """
    world = check_world
    rows, cols = np.nonzero(world.distance >= config.frontier_clearance_m)
    if rows.size == 0:
        return None, None
    xs, ys = grid.cell_center(rows, cols)
    near = np.hypot(xs - pose.x, ys - pose.y)
    choice = np.nonzero((near > 3.0) & (near < 8.0))[0]
    if choice.size == 0:
        return None, None
    best = choice[np.argsort(-world.distance[rows[choice], cols[choice]])[:5]]
    offset = world.vehicle.length / 2 - world.vehicle.rear_overhang
    for index in best:
        for heading in np.linspace(-math.pi, math.pi, 8, endpoint=False):
            goal = Pose(xs[index] - offset * math.cos(heading), ys[index] - offset * math.sin(heading), float(heading))
            if world.is_collision(goal):
                continue
            plan = HybridAStar(plan_world, planner).plan(start=pose, goal=goal)
            if plan.success:
                return plan, goal
            break
    return None, None


def evaluate_parking(lot: ProceduralLot, pose: Pose) -> dict:
    """Score the final pose against the ground-truth bays (never used by the stack)."""
    vehicle = lot.vehicle
    offset = vehicle.length / 2 - vehicle.rear_overhang
    center = np.array([pose.x + offset * math.cos(pose.yaw), pose.y + offset * math.sin(pose.yaw)])
    us, vs = np.meshgrid(np.arange(-vehicle.length / 2, vehicle.length / 2 + 1e-9, 0.2), np.arange(-vehicle.width / 2, vehicle.width / 2 + 1e-9, 0.2))
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    points = np.column_stack([(center[0] + c * us - s * vs).ravel(), (center[1] + s * us + c * vs).ravel()])
    best = (0.0, None)
    for bay in lot.bays:
        bc, bs = math.cos(bay.slot.yaw), math.sin(bay.slot.yaw)
        rel = points - np.array(bay.slot.center)
        inside = (np.abs(rel @ np.array([bc, bs])) <= bay.slot.length / 2) & (np.abs(rel @ np.array([-bs, bc])) <= bay.slot.width / 2)
        fraction = float(inside.mean())
        if fraction > best[0]:
            best = (fraction, bay)
    fraction, bay = best
    if bay is None:
        return {"in_bay": False, "bay": None, "overlap": 0.0}
    difference = abs(wrap_angle(pose.yaw - bay.slot.yaw))
    axis_error = math.degrees(min(difference, math.pi - difference))
    return {
        "in_bay": fraction >= 0.8 and axis_error <= 15.0 and not bay.occupied,
        "bay": bay.bay_id,
        "bay_kind": bay.kind,
        "bay_was_free": not bay.occupied,
        "overlap": round(fraction, 3),
        "axis_error_deg": round(axis_error, 1),
        "center_error_m": round(float(np.hypot(*(center - np.array(bay.slot.center)))), 3),
    }


def run_mission(lot: ProceduralLot, config: MissionConfig = MissionConfig(), seed: int = 0) -> MissionResult:
    rng = np.random.default_rng(seed)
    vehicle = lot.vehicle
    truth = _truth_world(lot)
    pose = lot.start
    grid = OccupancyGrid((pose.x, pose.y), config.grid)
    trace, gears, events = [pose], [], []
    t = 0.0
    travelled = 0.0
    steps_per_scan = max(1, int(round(config.scan_period_s / config.dt)))
    mode = "decide"
    tracker: Optional[PlanTracker] = None
    plan: Optional[PlanResult] = None
    target: Optional[SlotCandidate] = None
    rejected: list[tuple[float, float]] = []
    # Frontiers that could not be planned to from the current position. A
    # frontier being driven towards is not excluded: it stays the target
    # until it is seen. The list resets after every leg, as the car moved.
    unreachable: list[tuple[float, float]] = []
    # A frontier aimed at repeatedly without disappearing is one the sensor
    # cannot see into from anywhere reachable (e.g. behind a row): drop it.
    frontier_visits: dict[tuple[int, int], int] = {}
    exhausted: list[tuple[float, float]] = []
    inspected: list[tuple[float, float]] = []
    leg_start = 0.0
    recoveries = 0
    escapes = 0
    step = 0
    controller = config.controller

    def log(kind: str, message: str) -> None:
        events.append(MissionEvent(round(t, 2), kind, message))

    def finish(outcome: str, message: str, collisions: int = 0) -> MissionResult:
        log("end", message)
        evaluation = evaluate_parking(lot, pose) if outcome == "PARKED" else {}
        success = outcome == "PARKED" and evaluation.get("in_bay", False) and collisions == 0
        return MissionResult(outcome, success, message, trace, gears, events, t, travelled, collisions, target, grid, evaluation)

    grid.integrate(scan(pose, lot.obstacles, vehicle, config.lidar, rng))
    while t < config.max_time_s:
        if mode == "decide":
            world = MapWorld(grid, vehicle, pose)
            full_margin = MapWorld(grid, vehicle, pose, unknown_is_obstacle=False)
            if full_margin.is_collision(pose) and escapes < config.max_recoveries:
                # Structure seen only on arrival sits inside the safety margin
                # around the car, so every plan from here would be refused.
                # Ease away first, planning with a reduced margin; the
                # independent guard still checks for real contact.
                escapes += 1
                reduced = MapWorld(grid, vehicle, pose, collision_margin=config.escape_margin_m, unknown_is_obstacle=False)
                bigger = PlannerConfig(max_expansions=config.recovery_expansions, goal_position_tolerance=1.2, goal_yaw_tolerance=math.pi)
                plan, goal = _open_space_plan(grid, reduced, full_margin, pose, config, bigger)
                if plan is not None:
                    log("recover", f"too close to newly seen structure; easing away to ({goal.x:.1f}, {goal.y:.1f})")
                    mode = "explore"
                    tracker = PlanTracker(plan, vehicle.wheelbase, vehicle.max_steering, controller)
                    leg_start = travelled
                    continue
            candidates = [
                item for item in detect_slots(grid, vehicle)
                if all(math.dist(item.center, other) > 1.5 for other in rejected)
            ]
            candidates.sort(key=lambda item: math.dist(item.center, (pose.x, pose.y)))
            plan, target = None, None
            for candidate in candidates[: config.candidates_per_decision]:
                found = _best_plan(world, pose, candidate, config.park_planner)
                if found is None:
                    rejected.append(candidate.center)
                    log("reject", f"bay candidate at ({candidate.center[0]:.1f}, {candidate.center[1]:.1f}) has no route yet")
                    continue
                plan, heading = found
                target = candidate
                entry = "reversing in" if next((g for g in reversed(plan.gears) if g), 1) < 0 else "driving in"
                log("park", f"verified bay at ({candidate.center[0]:.1f}, {candidate.center[1]:.1f}); {entry}")
                mode = "park"
                break
            if mode != "park":
                # Exploration is optimistic: unknown space counts as passable,
                # and the plan is re-checked after every scan. Parking is not:
                # the bay and its approach must be seen free.
                optimistic = MapWorld(grid, vehicle, pose, unknown_is_obstacle=False)
                plan = _inspection_plan(grid, optimistic, pose, config, inspected, rejected)
                if plan is not None:
                    log("inspect", f"promising gap at ({inspected[-1][0]:.1f}, {inspected[-1][1]:.1f}); driving up to look into it")
                    mode = "inspect"
                    tracker = PlanTracker(plan, vehicle.wheelbase, vehicle.max_steering, controller)
                    leg_start = travelled
                    continue
                for _ in range(config.frontier_attempts):
                    found_frontier = _frontier_goal(grid, optimistic, pose, config, unreachable + exhausted)
                    if found_frontier is None:
                        break
                    goal, centroid = found_frontier
                    if goal is None:
                        unreachable.append(centroid)
                        continue
                    attempt = HybridAStar(optimistic, config.explore_planner).plan(start=pose, goal=goal)
                    if attempt.success:
                        plan = attempt
                        unreachable.clear()
                        # Viewpoints aimed at repeatedly without the frontier
                        # going away cannot reveal it: stop returning there.
                        key = (round(centroid[0] / 2.5), round(centroid[1] / 2.5))
                        frontier_visits[key] = frontier_visits.get(key, 0) + 1
                        if frontier_visits[key] >= config.frontier_max_visits:
                            exhausted.append(centroid)
                        break
                    unreachable.append(centroid)
                    log("explore", f"frontier near ({goal.x:.1f}, {goal.y:.1f}) unreachable; trying another")
                if plan is None and recoveries < config.max_recoveries:
                    recoveries += 1
                    escape_world = optimistic
                    if optimistic.is_collision(pose):
                        # Newly seen structure is inside the safety margin of
                        # where the car stands; every plan from here would be
                        # refused. Escape with the margin reduced (the
                        # independent guard still checks real contact).
                        escape_world = MapWorld(grid, vehicle, pose, collision_margin=config.escape_margin_m, unknown_is_obstacle=False)
                    plan, goal = _recover(grid, escape_world, pose, config, unreachable)
                    if plan is not None:
                        unreachable.clear()
                        log("recover", f"boxed in; manoeuvring to open space at ({goal.x:.1f}, {goal.y:.1f})")
                if plan is None:
                    return finish("NO_SPACE_FOUND", "no verified bay and no reachable frontier left; the vehicle stops")
                if not events or events[-1].kind != "recover":
                    log("explore", f"no verified bay yet; exploring towards ({goal.x:.1f}, {goal.y:.1f})")
                mode = "explore"
            tracker = PlanTracker(plan, vehicle.wheelbase, vehicle.max_steering, controller)
            leg_start = travelled

        command = tracker.command(pose)
        if command is None:
            if mode == "park":
                return finish("PARKED", f"parked after {t:.1f} s and {travelled:.1f} m")
            recoveries = 0
            escapes = 0
            # "No route" was relative to where the car stood; retry from here.
            rejected.clear()
            mode = "decide"
            continue
        signed = command.speed * command.gear
        next_pose = Pose(
            pose.x + signed * math.cos(pose.yaw) * config.dt,
            pose.y + signed * math.sin(pose.yaw) * config.dt,
            wrap_angle(pose.yaw + signed / vehicle.wheelbase * math.tan(command.steering) * config.dt),
        )
        if truth.is_collision(next_pose, extra_margin=-truth.collision_margin):
            return finish("GUARD_STOP", "collision guard rejected the next pose; the vehicle stops", collisions=1)
        travelled += distance(pose, next_pose)
        pose = next_pose
        trace.append(pose)
        gears.append(command.gear)
        t += config.dt
        step += 1
        if step % steps_per_scan:
            continue
        grid.integrate(scan(pose, lot.obstacles, vehicle, config.lidar, rng))
        if mode == "explore" and travelled - leg_start >= config.explore_horizon_m:
            mode = "decide"
            continue
        if not _plan_still_clear(grid, vehicle, pose, plan, config.lookahead_check_m):
            log("replan", "newly seen obstacle blocks the plan; replanning")
            if mode == "park" and target is not None:
                rejected.append(target.center)
            mode = "decide"
    return finish("TIMEOUT", f"no parking within {config.max_time_s:.0f} s")


def _plan_still_clear(grid: OccupancyGrid, vehicle, pose: Pose, plan: Optional[PlanResult], horizon: float) -> bool:
    if plan is None:
        return True
    world = MapWorld(grid, vehicle, pose, unknown_is_obstacle=False)
    nearest = min(range(len(plan.poses)), key=lambda index: distance(plan.poses[index], pose))
    covered = 0.0
    for previous, current in zip(plan.poses[nearest:], plan.poses[nearest + 1:]):
        covered += distance(previous, current)
        if world.is_collision(current, extra_margin=-0.05):
            return False
        if covered > horizon:
            break
    return True


def _median(values) -> Optional[float]:
    values = list(values)
    return float(np.median(values)) if values else None


def _run_seed(seed: int) -> dict:
    from .procedural_lot import generate_lot

    import time as _time

    started = _time.perf_counter()
    result = run_mission(generate_lot(seed), seed=seed)
    return {
        "seed": seed,
        "outcome": result.outcome,
        "success": result.success,
        "time_s": round(result.time_s, 1),
        "distance_m": round(result.distance_m, 1),
        "collisions": result.collisions,
        "evaluation": result.evaluation,
        "message": result.message,
        "wall_s": round(_time.perf_counter() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import json
    import os
    from pathlib import Path

    from .procedural_lot import generate_lot

    parser = argparse.ArgumentParser(description="Park in never-seen lots using only the simulated LiDAR")
    parser.add_argument("--seed", type=int, default=1000, help="lot seed for a single mission")
    parser.add_argument("--output", default="artifacts/mission/mission.png", help="figure for a single mission")
    parser.add_argument("--benchmark", metavar="FIRST:LAST", help="run every seed in [FIRST, LAST) and summarise")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--json", default="artifacts/mission/benchmark.json")
    args = parser.parse_args(argv)

    if not args.benchmark:
        from .mission_render import render_mission

        lot = generate_lot(args.seed)
        result = run_mission(lot, seed=args.seed)
        for event in result.events:
            print(f"{event.time_s:7.1f} s  {event.kind:8s} {event.message}")
        print(json.dumps({"outcome": result.outcome, "success": result.success, "evaluation": result.evaluation}, indent=2))
        print(f"[figure] {render_mission(lot, result, Path(args.output))}")
        return 0 if result.success else 2

    first, last = (int(part) for part in args.benchmark.split(":"))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_seed, seed) for seed in range(first, last)]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"seed {row['seed']:5d}  {row['outcome']:15s} success={row['success']}  {row['time_s']:6.1f} s  {row['distance_m']:6.1f} m  {row['evaluation'].get('bay_kind', '')}", flush=True)
    rows.sort(key=lambda row: row["seed"])
    outcomes: dict[str, int] = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    parked = [row for row in rows if row["outcome"] == "PARKED"]
    summary = {
        "lots": len(rows),
        "success": sum(row["success"] for row in rows),
        "outcomes": outcomes,
        "parked_in_wrong_place": sum(1 for row in parked if not row["success"]),
        "collisions": sum(row["collisions"] for row in rows),
        "median_time_s_when_parked": float(np.median([row["time_s"] for row in parked])) if parked else None,
        "max_time_s_when_parked": max((row["time_s"] for row in parked), default=None),
        "median_center_error_m": _median(row["evaluation"]["center_error_m"] for row in parked),
        "max_center_error_m": max((row["evaluation"]["center_error_m"] for row in parked), default=None),
        "median_axis_error_deg": _median(row["evaluation"]["axis_error_deg"] for row in parked),
        "max_axis_error_deg": max((row["evaluation"]["axis_error_deg"] for row in parked), default=None),
    }
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "missions": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
