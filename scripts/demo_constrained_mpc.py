#!/usr/bin/env python3
"""Compare the existing Pure Pursuit baseline with a polygon-constrained MPC.

Research/demo only: this script simulates one low-speed, reverse-in perpendicular
parking maneuver in a 2.5 m slot. It uses static rectangular obstacles and does
not connect to CARLA, ROS 2, a vehicle, or actuators.

Install the optional optimizer and run from the repository root:
    python -m pip install -e '.[mpc]'
    python scripts/demo_constrained_mpc.py

The MPC optimizes acceleration and steering over a short horizon using the
rear-axle kinematic bicycle model. Hard SLSQP inequality constraints keep the
predicted vehicle polygon clear of obstacle polygons and world boundaries.
The collision constraint uses the separating-axis gap between convex polygons;
it is conservative for these rectangles and is not a formal safety guarantee.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from importlib import import_module
import json
import math
from pathlib import Path
import sys
import time
from typing import cast

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

# Running a script by path puts `scripts/` on sys.path, not the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ihateparking.controller import ControllerConfig, SimulationResult, pure_pursuit, simulate_closed_loop
from ihateparking.geometry import Pose, angle_error, distance, wrap_angle
from ihateparking.planner import PlanResult
from ihateparking.world import ParkingSlot, ParkingWorld, RectangleObstacle, VehicleSpec

try:
    minimize = getattr(import_module("scipy.optimize"), "minimize")
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    minimize = None


@dataclass(frozen=True)
class MPCConfig:
    dt: float = 0.18
    horizon_steps: int = 8
    cruise_speed_mps: float = 0.55
    max_speed_mps: float = 0.70
    max_accel_mps2: float = 0.9
    max_steering_rate_rad_s: float = 0.65
    extra_clearance_m: float = 0.05
    lookahead_m: float = 0.55
    max_iterations: int = 40
    warm_start: bool = True
    position_weight: float = 36.0
    heading_weight: float = 10.0
    speed_weight: float = 2.5
    acceleration_weight: float = 0.45
    steering_weight: float = 0.12
    steering_rate_weight: float = 1.5
    terminal_position_weight: float = 110.0
    terminal_heading_weight: float = 35.0
    feasibility_tolerance: float = 2e-4


@dataclass
class Rollout:
    success: bool
    poses: list[Pose]
    steerings: list[float]
    speeds_mps: list[float]
    elapsed_s: float
    collisions: int
    min_clearance_m: float
    clearances_m: list[float]
    solve_times_s: list[float]
    message: str
    controller_call_times_s: list[float] = field(default_factory=list)


@dataclass
class DemoComparison:
    world: ParkingWorld
    reference: list[Pose]
    pure_pursuit: Rollout
    constrained_mpc: Rollout
    config: MPCConfig


def tight_slot_world() -> ParkingWorld:
    """A reverse-in slot with 0.30 m nominal lateral clearance before margins."""
    slot_center = (7.5, 8.0)
    vehicle = VehicleSpec(length=4.4, width=1.9, wheelbase=2.7, rear_overhang=0.7)
    obstacles = [
        # The inner faces are 2.50 m apart; the vehicle is 1.90 m wide.
        RectangleObstacle("parked_left", (5.25, 8.0), 4.5, 2.0, math.pi / 2),
        RectangleObstacle("parked_right", (9.75, 8.0), 4.5, 2.0, math.pi / 2),
        RectangleObstacle("back_curb", (7.5, 11.95), 13.0, 0.25, 0.0),
    ]
    return ParkingWorld(
        bounds=(0.0, 16.0, 0.0, 12.0),
        obstacles=obstacles,
        slot=ParkingSlot(slot_center, length=4.6, width=2.5, yaw=-math.pi / 2),
        # Face out of the slot; reverse motion travels north into the space.
        start_pose=Pose(7.55, 4.2, -math.pi / 2),
        vehicle=vehicle,
        collision_margin=0.12,
    )


def make_reference(start: Pose, goal: Pose, samples: int = 100) -> list[Pose]:
    """Create a smooth, feasible reverse-in reference with a small lateral shift."""
    poses: list[Pose] = []
    dy = goal.y - start.y
    if dy <= 0:
        raise ValueError("This demonstration expects a northbound reverse maneuver.")
    for t in np.linspace(0.0, 1.0, samples):
        smooth = 3.0 * t * t - 2.0 * t * t * t
        x = start.x + (goal.x - start.x) * smooth
        y = start.y + dy * t
        dx_dt = (goal.x - start.x) * 6.0 * t * (1.0 - t)
        dy_dt = dy
        travel_heading = math.atan2(dy_dt, dx_dt)
        vehicle_yaw = wrap_angle(travel_heading + math.pi)  # reverse gear
        poses.append(Pose(float(x), float(y), vehicle_yaw))
    poses[0] = start
    poses[-1] = goal
    return poses


def path_arclength(path: list[Pose]) -> np.ndarray:
    lengths = [0.0]
    for a, b in zip(path, path[1:]):
        lengths.append(lengths[-1] + distance(a, b))
    return np.asarray(lengths, dtype=float)


def sample_path(path: list[Pose], arc: np.ndarray, s: float) -> Pose:
    s = float(np.clip(s, 0.0, arc[-1]))
    i = int(np.searchsorted(arc, s, side="right") - 1)
    i = min(max(i, 0), len(path) - 2)
    span = arc[i + 1] - arc[i]
    alpha = 0.0 if span <= 1e-12 else (s - arc[i]) / span
    a, b = path[i], path[i + 1]
    return Pose(
        a.x + alpha * (b.x - a.x),
        a.y + alpha * (b.y - a.y),
        wrap_angle(a.yaw + alpha * angle_error(b.yaw, a.yaw)),
    )


def _polygon_axes(polygon: np.ndarray) -> list[np.ndarray]:
    axes = []
    for p, q in zip(polygon, np.vstack((polygon[1:], polygon[:1]))):
        edge = q - p
        axis = np.array([-edge[1], edge[0]], dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm > 1e-12:
            axes.append(axis / norm)
    return axes


def separating_axis_gap(first: np.ndarray, second: np.ndarray) -> float:
    """Signed SAT gap: positive means separated, negative means overlapping.

    For rectangles, checking axes from both polygons is sufficient. When
    separated, the maximum projection gap is a conservative clearance proxy.
    """
    gaps = []
    for axis in _polygon_axes(first) + _polygon_axes(second):
        p1, p2 = first @ axis, second @ axis
        gaps.append(float(max(p2.min() - p1.max(), p1.min() - p2.max())))
    return max(gaps) if gaps else -math.inf


def world_clearance(world: ParkingWorld, pose: Pose) -> float:
    """Minimum raw polygon/bounds clearance (before configured safety margins)."""
    footprint = world.vehicle_polygon(pose)
    xmin, xmax, ymin, ymax = world.bounds
    clearances = [
        float(footprint[:, 0].min() - xmin),
        float(xmax - footprint[:, 0].max()),
        float(footprint[:, 1].min() - ymin),
        float(ymax - footprint[:, 1].max()),
    ]
    clearances.extend(
        separating_axis_gap(footprint, obstacle.polygon())
        for obstacle in world.obstacles
    )
    return min(clearances)


def _rollout_controls(
    pose0: Pose,
    speed0: float,
    controls: np.ndarray,
    gear: int,
    dt: float,
    wheelbase: float,
) -> tuple[list[Pose], list[float]]:
    """Midpoint integration of the signed rear-axle bicycle model."""
    poses = [pose0]
    speeds = [float(speed0)]
    pose = pose0
    speed = float(speed0)
    for acceleration, steering in controls:
        next_speed = speed + float(acceleration) * dt
        mid_speed = 0.5 * (speed + next_speed)
        signed_distance = gear * mid_speed * dt
        yaw_delta = signed_distance / wheelbase * math.tan(float(steering))
        mid_yaw = pose.yaw + 0.5 * yaw_delta
        pose = Pose(
            pose.x + signed_distance * math.cos(mid_yaw),
            pose.y + signed_distance * math.sin(mid_yaw),
            wrap_angle(pose.yaw + yaw_delta),
        )
        speed = next_speed
        poses.append(pose)
        speeds.append(speed)
    return poses, speeds


class PolygonConstrainedMPC:
    """Direct-shooting nonlinear MPC with hard polygon-clearance constraints."""

    def __init__(self, world: ParkingWorld, path: list[Pose], config: MPCConfig):
        self.world = world
        self.path = path
        self.arc = path_arclength(path)
        self.cfg = config
        self.gear = -1  # gear is discrete and fixed for this one-segment demo
        self.prev_steering = 0.0
        self.previous_controls: np.ndarray | None = None

    def _clearance_constraints(self, poses: list[Pose]) -> np.ndarray:
        margin = self.world.collision_margin + self.cfg.extra_clearance_m
        xmin, xmax, ymin, ymax = self.world.bounds
        values: list[float] = []
        for pose in poses[1:]:
            footprint = self.world.vehicle_polygon(pose)
            # Hard rectangle separation constraints against every parked actor.
            for obstacle in self.world.obstacles:
                values.append(
                    separating_axis_gap(footprint, obstacle.polygon()) - margin
                )
            # Constrain all four footprint extremes inside the shrunken world.
            values.extend(
                (
                    float(footprint[:, 0].min() - xmin - margin),
                    float(xmax - margin - footprint[:, 0].max()),
                    float(footprint[:, 1].min() - ymin - margin),
                    float(ymax - margin - footprint[:, 1].max()),
                )
            )
        return np.asarray(values, dtype=float)

    def solve(
        self,
        pose: Pose,
        speed: float,
        progress_s: float,
        previous_steering: float,
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        cfg = self.cfg
        n = cfg.horizon_steps
        dt = cfg.dt
        wb = self.world.vehicle.wheelbase
        max_steer = self.world.vehicle.max_steering
        max_accel = cfg.max_accel_mps2
        total = float(self.arc[-1])

        def unpack(flat: np.ndarray) -> np.ndarray:
            return flat.reshape(n, 2)

        def predict(flat: np.ndarray) -> tuple[list[Pose], list[float]]:
            return _rollout_controls(pose, speed, unpack(flat), self.gear, dt, wb)

        def objective(flat: np.ndarray) -> float:
            controls = unpack(flat)
            predicted_poses, predicted_speeds = predict(flat)
            cost = 0.0
            for k in range(1, n + 1):
                travel_ref = min(
                    total,
                    progress_s + cfg.cruise_speed_mps * dt * k,
                )
                ref = sample_path(self.path, self.arc, travel_ref)
                remaining = max(0.0, total - travel_ref)
                target_speed = min(
                    cfg.cruise_speed_mps,
                    math.sqrt(2.0 * max_accel * remaining),
                )
                state = predicted_poses[k]
                speed_k = predicted_speeds[k]
                accel_k, steer_k = controls[k - 1]
                yaw_error = angle_error(ref.yaw, state.yaw)
                cost += cfg.position_weight * (
                    (state.x - ref.x) ** 2 + (state.y - ref.y) ** 2
                )
                cost += cfg.heading_weight * yaw_error**2
                cost += cfg.speed_weight * (speed_k - target_speed) ** 2
                cost += cfg.acceleration_weight * accel_k**2
                cost += cfg.steering_weight * steer_k**2
                delta_prev = previous_steering if k == 1 else controls[k - 2, 1]
                cost += cfg.steering_rate_weight * (steer_k - delta_prev) ** 2
            terminal = predicted_poses[-1]
            goal = self.path[-1]
            cost += cfg.terminal_position_weight * (
                (terminal.x - goal.x) ** 2 + (terminal.y - goal.y) ** 2
            )
            cost += cfg.terminal_heading_weight * angle_error(goal.yaw, terminal.yaw) ** 2
            cost += 8.0 * predicted_speeds[-1] ** 2
            return float(cost)

        def constraints(flat: np.ndarray) -> np.ndarray:
            controls = unpack(flat)
            predicted_poses, predicted_speeds = predict(flat)
            values = list(self._clearance_constraints(predicted_poses))
            for speed_k in predicted_speeds[1:]:
                values.extend((speed_k, cfg.max_speed_mps - speed_k))
            rate_limit = cfg.max_steering_rate_rad_s * dt
            previous = previous_steering
            for _, steering in controls:
                values.append(rate_limit - abs(float(steering) - previous))
                previous = float(steering)
            return np.asarray(values, dtype=float)

        # Use the geometric Pure Pursuit law for the first seed or when the
        # shifted previous solution is disabled; constraints are checked again.
        guess = np.zeros((n, 2), dtype=float)
        warm_speed = float(speed)
        for k in range(n):
            s_ref = min(total, progress_s + max(warm_speed, 0.25) * dt * (k + 1))
            target = sample_path(self.path, self.arc, s_ref)
            guess[k, 0] = np.clip(
                (min(cfg.cruise_speed_mps, math.sqrt(2 * max_accel * max(total - s_ref, 0.0))) - warm_speed) / dt,
                -max_accel,
                max_accel,
            )
            guess[k, 1] = pure_pursuit(
                pose,
                target,
                self.gear,
                wb,
                cfg.lookahead_m,
                max_steer,
            )
            warm_speed = float(np.clip(warm_speed + guess[k, 0] * dt, 0.0, cfg.max_speed_mps))

        warm_start_used = bool(cfg.warm_start and self.previous_controls is not None)
        if warm_start_used:
            previous = cast(np.ndarray, self.previous_controls)
            guess = np.vstack((previous[1:], previous[-1:])).copy()

        bounds = []
        for _ in range(n):
            bounds.extend(((-max_accel, max_accel), (-max_steer, max_steer)))
        if minimize is None:
            raise RuntimeError(
                "SciPy is required for the constrained MPC demo. Install it with: "
                "python -m pip install -e '.[mpc]'"
            )
        started = time.perf_counter()
        result = minimize(
            objective,
            guess.ravel(),
            method="SLSQP",
            bounds=bounds,
            constraints=({"type": "ineq", "fun": constraints},),
            options={"maxiter": cfg.max_iterations, "ftol": 2e-3, "disp": False},
        )
        elapsed = time.perf_counter() - started
        controls = unpack(np.asarray(result.x, dtype=float))
        min_constraint = float(np.min(constraints(result.x)))
        feasible = bool(np.isfinite(result.fun) and min_constraint >= -cfg.feasibility_tolerance)
        self.previous_controls = controls.copy() if cfg.warm_start and feasible else None
        info: dict[str, object] = {
            "solver_success": bool(result.success),
            "feasible": feasible,
            "warm_start_used": warm_start_used,
            "status": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "min_constraint_m": min_constraint,
            "objective": float(result.fun) if np.isfinite(result.fun) else None,
            "solve_time_s": elapsed,
        }
        return (controls if feasible else None), info


def nearest_progress(path: list[Pose], arc: np.ndarray, pose: Pose, minimum_s: float) -> float:
    first = int(np.searchsorted(arc, minimum_s, side="left"))
    candidates = range(min(first, len(path) - 1), len(path))
    best_index = min(candidates, key=lambda i: distance(path[i], pose))
    return max(minimum_s, float(arc[best_index]))


def simulate_mpc(
    world: ParkingWorld,
    path: list[Pose],
    config: MPCConfig,
    max_time_s: float = 25.0,
) -> Rollout:
    controller = PolygonConstrainedMPC(world, path, config)
    pose = path[0]
    speed = 0.0
    progress_s = 0.0
    poses = [pose]
    steerings: list[float] = []
    speeds = [speed]
    solve_times: list[float] = []
    controller_call_times: list[float] = []
    clearances = [world_clearance(world, pose)]
    elapsed = 0.0
    collisions = 0
    previous_steering = 0.0
    max_steps = int(max_time_s / config.dt)

    for _ in range(max_steps):
        goal = path[-1]
        if (
            distance(pose, goal) <= 0.10
            and abs(angle_error(goal.yaw, pose.yaw)) <= math.radians(5.0)
            and speed <= 0.12
        ):
            return Rollout(
                True, poses, steerings, speeds, elapsed, collisions,
                min(clearances), clearances, solve_times, "MPC reached the parked pose",
                controller_call_times,
            )

        call_started = time.perf_counter()
        controls, info = controller.solve(pose, speed, progress_s, previous_steering)
        controller_call_times.append(time.perf_counter() - call_started)
        solve_times.append(cast(float, info["solve_time_s"]))
        if controls is None:
            return Rollout(
                False, poses, steerings, speeds, elapsed, collisions,
                min(clearances), clearances, solve_times,
                "MPC safe-stop: no feasible command; " + str(info["status"]),
                controller_call_times,
            )

        acceleration, steering = map(float, controls[0])
        next_poses, next_speeds = _rollout_controls(
            pose,
            speed,
            np.asarray([[acceleration, steering]], dtype=float),
            controller.gear,
            config.dt,
            world.vehicle.wheelbase,
        )
        next_pose, next_speed = next_poses[-1], next_speeds[-1]
        if world.is_collision(next_pose, extra_margin=config.extra_clearance_m):
            collisions += 1
            return Rollout(
                False, poses, steerings, speeds, elapsed, collisions,
                min(clearances), clearances, solve_times,
                "MPC command rejected by the independent footprint collision check",
                controller_call_times,
            )

        pose, speed = next_pose, float(next_speed)
        elapsed += config.dt
        previous_steering = steering
        progress_s = nearest_progress(path, controller.arc, pose, progress_s)
        poses.append(pose)
        steerings.append(steering)
        speeds.append(speed)
        clearances.append(world_clearance(world, pose))

    return Rollout(
        False, poses, steerings, speeds, elapsed, collisions,
        min(clearances), clearances, solve_times, "MPC timed out before reaching the slot",
        controller_call_times,
    )


def rollout_from_pure_pursuit(
    world: ParkingWorld,
    path: list[Pose],
    dt: float = 0.10,
    speed_mps: float = 0.55,
) -> Rollout:
    plan = PlanResult(
        success=True,
        poses=path,
        gears=[-1] * len(path),
        steerings=[0.0] * len(path),
        cost=0.0,
        expanded=0,
        message="smooth reverse-in demonstration reference",
    )
    result: SimulationResult = simulate_closed_loop(
        world,
        plan,
        ControllerConfig(
            dt=dt,
            speed=speed_mps,
            lookahead=0.55,
            waypoint_tolerance=0.18,
            max_time=25.0,
            goal_position_tolerance=0.10,
            goal_yaw_tolerance=math.radians(5.0),
        ),
    )
    clearances = [world_clearance(world, pose) for pose in result.poses]
    return Rollout(
        result.success,
        result.poses,
        result.steerings,
        [speed_mps] * len(result.poses),
        result.time,
        result.collisions,
        min(clearances, default=-math.inf),
        clearances,
        [],
        result.message,
    )


def _draw_world(ax: Axes, world: ParkingWorld) -> None:
    from matplotlib.patches import Polygon as PolygonPatch

    ax.add_patch(PolygonPatch(world.slot.polygon(), closed=True, fill=False,
                              edgecolor="#4c956c", linewidth=2.0, linestyle="--",
                              label="target slot"))
    for obstacle in world.obstacles:
        polygon = obstacle.polygon()
        ax.add_patch(PolygonPatch(polygon, closed=True, facecolor="#747d86",
                                  edgecolor="#30343b", alpha=0.85))
        center = np.mean(polygon, axis=0)
        ax.text(center[0], center[1], obstacle.name.replace("parked_", ""),
                ha="center", va="center", fontsize=8, color="white")
    xmin, xmax, ymin, ymax = world.bounds
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal",
           xlabel="x [m]", ylabel="y [m]", title="Reverse-in parking: 2.50 m slot")
    ax.grid(alpha=0.2)


def _summary(rollout: Rollout) -> dict[str, object]:
    optimizer_times = np.asarray(rollout.solve_times_s, dtype=float)
    call_times = np.asarray(rollout.controller_call_times_s, dtype=float)
    summary: dict[str, object] = {
        "success": rollout.success,
        "message": rollout.message,
        "elapsed_simulation_s": rollout.elapsed_s,
        "steps": max(0, len(rollout.poses) - 1),
        "collisions": rollout.collisions,
        "minimum_raw_polygon_clearance_m": rollout.min_clearance_m,
        "mean_mpc_solve_time_ms": float(optimizer_times.mean() * 1000) if optimizer_times.size else None,
        "p95_mpc_solve_time_ms": float(np.percentile(optimizer_times, 95) * 1000) if optimizer_times.size else None,
        "p95_controller_call_ms": float(np.percentile(call_times, 95) * 1000) if call_times.size else None,
        "p99_controller_call_ms": float(np.percentile(call_times, 99) * 1000) if call_times.size else None,
    }
    if call_times.size:
        summary["mean_controller_call_ms"] = float(call_times.mean() * 1000)
    return summary


def run_comparison(config: MPCConfig) -> DemoComparison:
    world = tight_slot_world()
    start = world.start_pose
    goal = world.slot.goal_pose(world.vehicle)
    path = make_reference(start, goal)
    extra = config.extra_clearance_m
    for i, pose in enumerate(path):
        if world.is_collision(pose, extra_margin=extra):
            raise RuntimeError(
                f"Reference pose {i} is not collision-free with the requested MPC margin."
            )

    # Give both controllers the same expanded collision envelope.
    baseline_world = replace(
        world,
        collision_margin=world.collision_margin + config.extra_clearance_m,
    )
    pp = rollout_from_pure_pursuit(baseline_world, path)
    mpc = simulate_mpc(world, path, config)
    return DemoComparison(world, path, pp, mpc, config)


def comparison_payload(comparison: DemoComparison) -> dict[str, object]:
    world = comparison.world
    config = comparison.config
    pp = comparison.pure_pursuit
    mpc = comparison.constrained_mpc
    goal = world.slot.goal_pose(world.vehicle)
    pose_list = lambda poses: [asdict(pose) for pose in poses]
    call_times = [round(value * 1000.0, 4) for value in mpc.controller_call_times_s]
    optimizer_times = [round(value * 1000.0, 4) for value in mpc.solve_times_s]
    deadline_misses = sum(value > config.dt for value in mpc.controller_call_times_s)
    return {
        "scenario": {
            "maneuver": "reverse-in perpendicular parking",
            "reference_source": "smooth synthetic reverse-in path shared by both controllers; not a Hybrid A* output",
            "slot_width_m": world.slot.width,
            "vehicle_width_m": world.vehicle.width,
            "world_collision_margin_m": world.collision_margin,
            "shared_extra_clearance_m": config.extra_clearance_m,
            "effective_controller_clearance_m": world.collision_margin + config.extra_clearance_m,
            "moving_obstacles": False,
            "gear": "reverse for the complete maneuver; discrete gear shifts are out of scope",
        },
        "mpc_configuration": asdict(config),
        "pure_pursuit": _summary(pp),
        "constrained_mpc": _summary(mpc),
        "scene": {
            "bounds": list(world.bounds),
            "slot_polygon": world.slot.polygon().tolist(),
            "obstacles": [
                {"name": obstacle.name, "polygon": obstacle.polygon().tolist()}
                for obstacle in world.obstacles
            ],
            "start": asdict(world.start_pose),
            "goal": asdict(goal),
        },
        "trajectories": {
            "reference": pose_list(comparison.reference),
            "pure_pursuit": pose_list(pp.poses),
            "constrained_mpc": pose_list(mpc.poses),
        },
        "clearance_profiles": {
            "pure_pursuit": {"dt_s": 0.10, "clearance_m": pp.clearances_m},
            "constrained_mpc": {"dt_s": config.dt, "clearance_m": mpc.clearances_m},
        },
        "latency": {
            "controller_call_ms": call_times,
            "optimizer_only_ms": optimizer_times,
            "simulation_step_budget_ms": config.dt * 1000.0,
            "controller_calls": len(call_times),
            "misses_simulation_step_budget": int(deadline_misses),
            "miss_rate_percent": 100.0 * deadline_misses / len(call_times) if call_times else None,
            "project_control_periods_ms": {"20_hz": 50.0, "50_hz": 20.0},
        },
        "safety_note": (
            "Research simulation only. Polygon constraints are checked at discrete horizon "
            "steps and each executed step has an independent footprint check. This is not a "
            "continuous-time or real-vehicle safety guarantee."
        ),
    }


def run_demo(output_dir: Path, config: MPCConfig) -> dict[str, object]:
    comparison = run_comparison(config)
    world = comparison.world
    start = world.start_pose
    goal = world.slot.goal_pose(world.vehicle)
    path = comparison.reference
    pp = comparison.pure_pursuit
    mpc = comparison.constrained_mpc
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    _draw_world(ax, world)
    ax.plot([p.x for p in path], [p.y for p in path], "--", color="#9aa0a6", linewidth=1.2, label="reference")
    if pp.poses:
        ax.plot([p.x for p in pp.poses], [p.y for p in pp.poses], color="#168aad", linewidth=2.0, label="Pure Pursuit")
    if mpc.poses:
        ax.plot([p.x for p in mpc.poses], [p.y for p in mpc.poses], color="#e76f51", linewidth=2.0, label="polygon-constrained MPC")
    ax.scatter([start.x], [start.y], marker="o", color="#111111", zorder=5, label="start")
    ax.scatter([goal.x], [goal.y], marker="*", s=100, color="#4c956c", zorder=5, label="goal")
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    for label, rollout, color in (
        ("Pure Pursuit", pp, "#168aad"),
        ("Constrained MPC", mpc, "#e76f51"),
    ):
        if rollout.clearances_m:
            times = np.arange(len(rollout.clearances_m)) * (config.dt if label.startswith("Constrained") else 0.10)
            ax.plot(times, rollout.clearances_m, color=color, label=label)
    ax.axhline(world.collision_margin, color="#444444", linestyle="--", linewidth=1.0,
               label="world collision margin")
    ax.axhline(world.collision_margin + config.extra_clearance_m,
               color="#e76f51", linestyle=":", linewidth=1.2,
               label="shared PP/MPC required margin")
    ax.set(xlabel="simulation time [s]", ylabel="minimum raw footprint clearance [m]",
           title="Clearance along the executed trajectory")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    figure.suptitle("Constrained MPC vs current Pure Pursuit baseline — simulation only")
    figure.tight_layout()
    figure_path = output_dir / "constrained_mpc_tight_parking.png"
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    report = comparison_payload(comparison)
    report["figure"] = str(figure_path)
    report_path = output_dir / "constrained_mpc_tight_parking.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved figure: {figure_path}")
    print(f"Saved metrics: {report_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mpc_demo"))
    parser.add_argument("--horizon", type=int, default=8, help="MPC prediction steps (default: 8)")
    parser.add_argument("--max-iterations", type=int, default=40)
    args = parser.parse_args(argv)
    if args.horizon < 3:
        parser.error("--horizon must be at least 3")
    config = MPCConfig(horizon_steps=args.horizon, max_iterations=args.max_iterations)
    try:
        run_demo(args.output_dir, config)
    except Exception as exc:
        print(f"MPC demo failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
