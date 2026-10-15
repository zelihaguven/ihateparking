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
from typing import Any, Literal, cast

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
    derivative_mode: Literal["finite_difference", "casadi_ad"] = "finite_difference"
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
    solver_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    solver_setup_time_s: float = 0.0


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


def _rollout_arrays(
    pose0: Pose,
    speed0: float,
    controls: np.ndarray,
    gear: int,
    dt: float,
    wheelbase: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized midpoint rollout used by repeated optimizer callbacks."""
    controls = np.asarray(controls, dtype=float).reshape(-1, 2)
    accelerations = controls[:, 0]
    steerings = controls[:, 1]
    speeds = np.empty(len(controls) + 1, dtype=float)
    speeds[0] = speed0
    speeds[1:] = speed0 + np.cumsum(accelerations) * dt
    signed_distances = gear * 0.5 * (speeds[:-1] + speeds[1:]) * dt
    yaw_deltas = signed_distances / wheelbase * np.tan(steerings)
    yaws = np.empty(len(controls) + 1, dtype=float)
    yaws[0] = pose0.yaw
    yaws[1:] = pose0.yaw + np.cumsum(yaw_deltas)
    mid_yaws = yaws[:-1] + 0.5 * yaw_deltas
    xs = np.empty(len(controls) + 1, dtype=float)
    ys = np.empty(len(controls) + 1, dtype=float)
    xs[0], ys[0] = pose0.x, pose0.y
    xs[1:] = pose0.x + np.cumsum(signed_distances * np.cos(mid_yaws))
    ys[1:] = pose0.y + np.cumsum(signed_distances * np.sin(mid_yaws))
    return xs, ys, yaws, speeds


def _rectangle_sat_gap(
    x: float,
    y: float,
    yaw: float,
    vehicle: VehicleSpec,
    obstacle: RectangleObstacle,
    obstacle_axes: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Return the polygon SAT gap using rectangle support intervals."""
    ego_forward = np.array([math.cos(yaw), math.sin(yaw)])
    ego_lateral = np.array([-math.sin(yaw), math.cos(yaw)])
    if obstacle_axes is None:
        obs_forward = np.array([math.cos(obstacle.yaw), math.sin(obstacle.yaw)])
        obs_lateral = np.array([-math.sin(obstacle.yaw), math.cos(obstacle.yaw)])
    else:
        obs_forward, obs_lateral = obstacle_axes
    axes = np.vstack((ego_forward, ego_lateral, obs_forward, obs_lateral))

    axle_to_center = vehicle.length / 2.0 - vehicle.rear_overhang
    ego_center = np.array([x, y]) + axle_to_center * ego_forward
    ego_radius = (
        vehicle.length / 2.0 * np.abs(axes @ ego_forward)
        + vehicle.width / 2.0 * np.abs(axes @ ego_lateral)
    )
    obs_center = np.asarray(obstacle.center, dtype=float)
    obs_radius = (
        obstacle.length / 2.0 * np.abs(axes @ obs_forward)
        + obstacle.width / 2.0 * np.abs(axes @ obs_lateral)
    )
    ego_projection = axes @ ego_center
    obs_projection = axes @ obs_center
    gaps = np.concatenate(
        (
            obs_projection - obs_radius - (ego_projection + ego_radius),
            ego_projection - ego_radius - (obs_projection + obs_radius),
        )
    )
    return float(np.max(gaps))


class PolygonConstrainedMPC:
    """Direct-shooting MPC with cached callbacks and optional CasADi Jacobians."""

    def __init__(self, world: ParkingWorld, path: list[Pose], config: MPCConfig):
        self.world = world
        self.path = path
        self.arc = path_arclength(path)
        self.cfg = config
        self.gear = -1  # gear is discrete and fixed for this one-segment demo
        self.prev_steering = 0.0
        self.previous_controls: np.ndarray | None = None
        self._obstacle_axes = [
            (
                np.array([math.cos(obstacle.yaw), math.sin(obstacle.yaw)]),
                np.array([-math.sin(obstacle.yaw), math.cos(obstacle.yaw)]),
            )
            for obstacle in world.obstacles
        ]
        self._ad_model: Any | None = None
        self.solver_setup_time_s = 0.0
        if config.derivative_mode == "casadi_ad":
            setup_started = time.perf_counter()
            module_name = (
                f"{__package__}.mpc_autodiff" if __package__ else "mpc_autodiff"
            )
            casadi_model = getattr(import_module(module_name), "CasadiMPCAD")
            self._ad_model = casadi_model(
                world, path[-1], config, self.gear, config.horizon_steps
            )
            self.solver_setup_time_s = time.perf_counter() - setup_started

    def _reference_targets(self, progress_s: float) -> list[tuple[Pose, float]]:
        total = float(self.arc[-1])
        targets = []
        for k in range(1, self.cfg.horizon_steps + 1):
            travel_ref = min(
                total,
                progress_s + self.cfg.cruise_speed_mps * self.cfg.dt * k,
            )
            ref = sample_path(self.path, self.arc, travel_ref)
            remaining = max(0.0, total - travel_ref)
            target_speed = min(
                self.cfg.cruise_speed_mps,
                math.sqrt(2.0 * self.cfg.max_accel_mps2 * remaining),
            )
            targets.append((ref, target_speed))
        return targets

    @staticmethod
    def _parameters(
        pose: Pose,
        speed: float,
        previous_steering: float,
        references: list[tuple[Pose, float]],
    ) -> np.ndarray:
        values = [pose.x, pose.y, pose.yaw, speed, previous_steering]
        for ref, target_speed in references:
            values.extend((ref.x, ref.y, ref.yaw, target_speed))
        return np.asarray(values, dtype=float)

    @staticmethod
    def _reference_arrays(
        references: list[tuple[Pose, float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray([ref.x for ref, _ in references], dtype=float),
            np.asarray([ref.y for ref, _ in references], dtype=float),
            np.asarray([ref.yaw for ref, _ in references], dtype=float),
            np.asarray([target_speed for _, target_speed in references], dtype=float),
        )

    def _clearance_constraints(
        self, xs: np.ndarray, ys: np.ndarray, yaws: np.ndarray
    ) -> np.ndarray:
        margin = self.world.collision_margin + self.cfg.extra_clearance_m
        xmin, xmax, ymin, ymax = self.world.bounds
        vehicle = self.world.vehicle
        half_length = vehicle.length / 2.0
        half_width = vehicle.width / 2.0
        axle_to_center = vehicle.length / 2.0 - vehicle.rear_overhang
        values: list[float] = []
        for index in range(1, len(xs)):
            yaw = float(yaws[index])
            forward = np.array([math.cos(yaw), math.sin(yaw)])
            center_x = float(xs[index]) + axle_to_center * forward[0]
            center_y = float(ys[index]) + axle_to_center * forward[1]
            for obstacle, axes in zip(self.world.obstacles, self._obstacle_axes):
                values.append(
                    _rectangle_sat_gap(
                        float(xs[index]), float(ys[index]), yaw, vehicle, obstacle, axes
                    )
                    - margin
                )
            extent_x = half_length * abs(forward[0]) + half_width * abs(forward[1])
            extent_y = half_length * abs(forward[1]) + half_width * abs(forward[0])
            values.extend(
                (
                    center_x - extent_x - xmin - margin,
                    xmax - margin - center_x - extent_x,
                    center_y - extent_y - ymin - margin,
                    ymax - margin - center_y - extent_y,
                )
            )
        return np.asarray(values, dtype=float)

    def _numpy_evaluate(
        self,
        vector: np.ndarray,
        pose: Pose,
        speed: float,
        previous_steering: float,
        reference_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[float, np.ndarray]:
        """Evaluate the exact numerical objective/constraints for regression checks."""
        cfg = self.cfg
        n = cfg.horizon_steps
        controls = np.asarray(vector, dtype=float).reshape(n, 2)
        xs, ys, yaws, speeds = _rollout_arrays(
            pose,
            speed,
            controls,
            self.gear,
            cfg.dt,
            self.world.vehicle.wheelbase,
        )
        ref_x, ref_y, ref_yaw, target_speeds = reference_arrays
        yaw_errors = np.arctan2(
            np.sin(ref_yaw - yaws[1:]), np.cos(ref_yaw - yaws[1:])
        )
        objective_value = float(
            cfg.position_weight
            * np.sum((xs[1:] - ref_x) ** 2 + (ys[1:] - ref_y) ** 2)
            + cfg.heading_weight * np.sum(yaw_errors**2)
            + cfg.speed_weight * np.sum((speeds[1:] - target_speeds) ** 2)
            + cfg.acceleration_weight * np.sum(controls[:, 0] ** 2)
            + cfg.steering_weight * np.sum(controls[:, 1] ** 2)
        )
        steering_before = np.concatenate(([previous_steering], controls[:-1, 1]))
        objective_value += float(
            cfg.steering_rate_weight
            * np.sum((controls[:, 1] - steering_before) ** 2)
        )
        terminal_yaw_error = math.atan2(
            math.sin(self.path[-1].yaw - yaws[-1]),
            math.cos(self.path[-1].yaw - yaws[-1]),
        )
        objective_value += cfg.terminal_position_weight * (
            (xs[-1] - self.path[-1].x) ** 2
            + (ys[-1] - self.path[-1].y) ** 2
        )
        objective_value += cfg.terminal_heading_weight * terminal_yaw_error**2
        objective_value += 8.0 * speeds[-1] ** 2

        values = list(self._clearance_constraints(xs, ys, yaws))
        for speed_k in speeds[1:]:
            values.extend((float(speed_k), cfg.max_speed_mps - float(speed_k)))
        rate_limit = cfg.max_steering_rate_rad_s * cfg.dt
        steering_prev = previous_steering
        for steering in controls[:, 1]:
            values.append(rate_limit - abs(float(steering) - steering_prev))
            steering_prev = float(steering)
        return objective_value, np.asarray(values, dtype=float)

    def solve(
        self,
        pose: Pose,
        speed: float,
        progress_s: float,
        previous_steering: float,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        cfg = self.cfg
        n = cfg.horizon_steps
        dt = cfg.dt
        vehicle = self.world.vehicle
        wb = vehicle.wheelbase
        max_steer = vehicle.max_steering
        max_accel = cfg.max_accel_mps2
        total = float(self.arc[-1])
        references = self._reference_targets(progress_s)
        reference_arrays = self._reference_arrays(references)
        parameters = self._parameters(pose, speed, previous_steering, references)
        stats = {
            "objective_calls": 0,
            "constraint_calls": 0,
            "objective_jacobian_calls": 0,
            "constraint_jacobian_calls": 0,
            "evaluation_cache_hits": 0,
            "evaluation_cache_misses": 0,
        }

        # Construct a geometric initial guess; the optional shifted prior
        # solution remains only an initial guess, never a relaxed constraint.
        warm_start_used = bool(cfg.warm_start and self.previous_controls is not None)
        if warm_start_used:
            previous = cast(np.ndarray, self.previous_controls)
            guess = np.vstack((previous[1:], previous[-1:])).copy()
        else:
            guess = np.zeros((n, 2), dtype=float)
            warm_speed = float(speed)
            for k in range(n):
                s_ref = min(total, progress_s + max(warm_speed, 0.25) * dt * (k + 1))
                target = sample_path(self.path, self.arc, s_ref)
                guess[k, 0] = np.clip(
                    (
                        min(
                            cfg.cruise_speed_mps,
                            math.sqrt(2 * max_accel * max(total - s_ref, 0.0)),
                        )
                        - warm_speed
                    )
                    / dt,
                    -max_accel,
                    max_accel,
                )
                guess[k, 1] = pure_pursuit(
                    pose, target, self.gear, wb, cfg.lookahead_m, max_steer
                )
                warm_speed = float(
                    np.clip(warm_speed + guess[k, 0] * dt, 0.0, cfg.max_speed_mps)
                )

        bounds = []
        for _ in range(n):
            bounds.extend(((-max_accel, max_accel), (-max_steer, max_steer)))
        if minimize is None:
            raise RuntimeError(
                "SciPy is required for the constrained MPC demo. Install it with: "
                "python -m pip install -e '.[mpc]'"
            )

        cached_key: bytes | None = None
        cached_values: tuple[float, np.ndarray, np.ndarray | None, np.ndarray | None] | None = None

        def evaluate(flat: np.ndarray) -> tuple[float, np.ndarray, np.ndarray | None, np.ndarray | None]:
            nonlocal cached_key, cached_values
            vector = np.asarray(flat, dtype=float).reshape(2 * n)
            key = vector.tobytes()
            if key == cached_key and cached_values is not None:
                stats["evaluation_cache_hits"] += 1
                return cached_values
            stats["evaluation_cache_misses"] += 1

            if self._ad_model is not None:
                objective_value, constraint_values, objective_gradient, constraint_jacobian = (
                    self._ad_model.evaluate(vector, parameters)
                )
                result = (
                    objective_value,
                    constraint_values,
                    objective_gradient,
                    constraint_jacobian,
                )
            else:
                objective_value, constraint_values = self._numpy_evaluate(
                    vector, pose, speed, previous_steering, reference_arrays
                )
                result = (
                    objective_value,
                    constraint_values,
                    None,
                    None,
                )
            cached_key = key
            cached_values = result
            return result

        def objective(flat: np.ndarray) -> float:
            stats["objective_calls"] += 1
            return evaluate(flat)[0]

        def constraints(flat: np.ndarray) -> np.ndarray:
            stats["constraint_calls"] += 1
            return evaluate(flat)[1]

        def objective_jacobian(flat: np.ndarray) -> np.ndarray:
            stats["objective_jacobian_calls"] += 1
            gradient = evaluate(flat)[2]
            if gradient is None:
                raise RuntimeError("Automatic objective gradient is unavailable.")
            return gradient

        def constraints_jacobian(flat: np.ndarray) -> np.ndarray:
            stats["constraint_jacobian_calls"] += 1
            jacobian = evaluate(flat)[3]
            if jacobian is None:
                raise RuntimeError("Automatic constraint Jacobian is unavailable.")
            return jacobian

        constraint_spec: dict[str, object] = {"type": "ineq", "fun": constraints}
        solver_kwargs: dict[str, object] = {
            "method": "SLSQP",
            "bounds": bounds,
            "constraints": (constraint_spec,),
            "options": {"maxiter": cfg.max_iterations, "ftol": 2e-3, "disp": False},
        }
        if self._ad_model is not None:
            solver_kwargs["jac"] = objective_jacobian
            constraint_spec["jac"] = constraints_jacobian

        started = time.perf_counter()
        result = minimize(objective, guess.ravel(), **solver_kwargs)
        elapsed = time.perf_counter() - started
        controls = np.asarray(result.x, dtype=float).reshape(n, 2)
        final_objective, final_constraints, _, _ = evaluate(result.x)
        min_constraint = float(np.min(final_constraints))
        feasible = bool(
            np.isfinite(final_objective)
            and np.all(np.isfinite(final_constraints))
            and min_constraint >= -cfg.feasibility_tolerance
        )
        self.previous_controls = controls.copy() if cfg.warm_start and feasible else None
        info: dict[str, Any] = {
            "solver_success": bool(result.success),
            "feasible": feasible,
            "warm_start_used": warm_start_used,
            "derivative_mode": cfg.derivative_mode,
            "status": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "optimizer_nfev": int(getattr(result, "nfev", 0)),
            "optimizer_njev": int(getattr(result, "njev", 0)),
            "min_constraint_m": min_constraint,
            "objective": final_objective if np.isfinite(final_objective) else None,
            "solve_time_s": elapsed,
            "callback_diagnostics": dict(stats),
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
    setup_started = time.perf_counter()
    controller = PolygonConstrainedMPC(world, path, config)
    solver_setup_time = time.perf_counter() - setup_started
    pose = path[0]
    speed = 0.0
    progress_s = 0.0
    poses = [pose]
    steerings: list[float] = []
    speeds = [speed]
    solve_times: list[float] = []
    controller_call_times: list[float] = []
    solver_diagnostics: list[dict[str, object]] = []
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
                controller_call_times, solver_diagnostics, solver_setup_time,
            )

        call_started = time.perf_counter()
        controls, info = controller.solve(pose, speed, progress_s, previous_steering)
        solve_times.append(cast(float, info["solve_time_s"]))
        solver_diagnostics.append(info)
        if controls is None:
            controller_call_times.append(time.perf_counter() - call_started)
            return Rollout(
                False, poses, steerings, speeds, elapsed, collisions,
                min(clearances), clearances, solve_times,
                "MPC safe-stop: no feasible command; " + str(info["status"]),
                controller_call_times, solver_diagnostics, solver_setup_time,
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
            controller_call_times.append(time.perf_counter() - call_started)
            collisions += 1
            return Rollout(
                False, poses, steerings, speeds, elapsed, collisions,
                min(clearances), clearances, solve_times,
                "MPC command rejected by the independent footprint collision check",
                controller_call_times, solver_diagnostics, solver_setup_time,
            )

        controller_call_times.append(time.perf_counter() - call_started)
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
        controller_call_times, solver_diagnostics, solver_setup_time,
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
    message = result.message
    if result.collisions:
        message = (
            "Pure Pursuit command rejected by the independent footprint collision "
            "check before it was applied"
        )
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
        message,
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
    callback_keys = (
        "objective_calls",
        "constraint_calls",
        "objective_jacobian_calls",
        "constraint_jacobian_calls",
        "evaluation_cache_hits",
        "evaluation_cache_misses",
    )
    callback_totals = {key: 0 for key in callback_keys}
    for diagnostic in rollout.solver_diagnostics:
        callbacks = diagnostic.get("callback_diagnostics", {})
        if isinstance(callbacks, dict):
            for key in callback_keys:
                callback_totals[key] += cast(int, callbacks.get(key, 0))
    diagnostic_modes = {
        str(item.get("derivative_mode")) for item in rollout.solver_diagnostics
    }
    summary: dict[str, object] = {
        "success": rollout.success,
        "message": rollout.message,
        "elapsed_simulation_s": rollout.elapsed_s,
        "steps": max(0, len(rollout.poses) - 1),
        "collisions": rollout.collisions,
        "minimum_raw_polygon_clearance_m": rollout.min_clearance_m,
        "solver_setup_time_ms": rollout.solver_setup_time_s * 1000.0,
        "derivative_mode": next(iter(diagnostic_modes)) if len(diagnostic_modes) == 1 else None,
        "solver_iterations_total": sum(cast(int, item.get("iterations", 0)) for item in rollout.solver_diagnostics),
        "optimizer_nfev_total": sum(cast(int, item.get("optimizer_nfev", 0)) for item in rollout.solver_diagnostics),
        "optimizer_njev_total": sum(cast(int, item.get("optimizer_njev", 0)) for item in rollout.solver_diagnostics),
        "callback_totals": callback_totals,
        "mean_mpc_solve_time_ms": float(optimizer_times.mean() * 1000) if optimizer_times.size else None,
        "p95_mpc_solve_time_ms": float(np.percentile(optimizer_times, 95) * 1000) if optimizer_times.size else None,
        "p50_controller_call_ms": float(np.percentile(call_times, 50) * 1000) if call_times.size else None,
        "p95_controller_call_ms": float(np.percentile(call_times, 95) * 1000) if call_times.size else None,
        "p99_controller_call_ms": float(np.percentile(call_times, 99) * 1000) if call_times.size else None,
        "max_controller_call_ms": float(call_times.max() * 1000) if call_times.size else None,
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
    parser.add_argument(
        "--derivative-mode",
        choices=("finite_difference", "casadi_ad"),
        default="finite_difference",
    )
    args = parser.parse_args(argv)
    if args.horizon < 3:
        parser.error("--horizon must be at least 3")
    config = MPCConfig(
        horizon_steps=args.horizon,
        max_iterations=args.max_iterations,
        derivative_mode=args.derivative_mode,
    )
    try:
        run_demo(args.output_dir, config)
    except Exception as exc:
        print(f"MPC demo failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
