"""Optional CasADi automatic-differentiation evaluator for the demo MPC.

This module does not select a different optimizer. It builds the same direct-
shooting objective and inequality constraints symbolically, then supplies their
derivatives to SciPy SLSQP. It is research code, not a real-time guarantee.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from ihateparking.geometry import Pose
from ihateparking.world import ParkingWorld


class CasadiMPCAD:
    """Build a parameterized cost/constraint/Jacobian function for one horizon."""

    def __init__(
        self,
        world: ParkingWorld,
        goal: Pose,
        config: Any,
        gear: int,
        horizon_steps: int,
    ) -> None:
        try:
            ca = import_module("casadi")
        except ImportError as exc:  # pragma: no cover - optional dependency branch
            raise RuntimeError(
                "CasADi is required for the automatic-differentiation mode. "
                "Install it with: python -m pip install -e '.[mpc-ad]'"
            ) from exc

        n = horizon_steps
        controls = ca.MX.sym("controls", 2 * n)
        # Initial rear-axle state, speed and steering; then n (x, y, yaw, speed)
        # tracking targets. Keeping references as parameters avoids rebuilding the
        # graph when the receding-horizon progress estimate changes.
        parameters = ca.MX.sym("parameters", 5 + 4 * n)
        pose_x, pose_y, yaw, speed = (
            parameters[0], parameters[1], parameters[2], parameters[3]
        )
        previous_steering = parameters[4]

        vehicle = world.vehicle
        half_length = vehicle.length / 2.0
        half_width = vehicle.width / 2.0
        axle_to_center = vehicle.length / 2.0 - vehicle.rear_overhang
        margin = world.collision_margin + config.extra_clearance_m
        xmin, xmax, ymin, ymax = world.bounds
        clearance_terms: list[Any] = []
        speed_terms: list[Any] = []
        steering_terms: list[Any] = []
        cost: Any = ca.MX(0.0)

        for index in range(n):
            acceleration = controls[2 * index]
            steering = controls[2 * index + 1]
            next_speed = speed + acceleration * config.dt
            signed_distance = gear * 0.5 * (speed + next_speed) * config.dt
            yaw_delta = signed_distance / vehicle.wheelbase * ca.tan(steering)
            mid_yaw = yaw + 0.5 * yaw_delta
            next_x = pose_x + signed_distance * ca.cos(mid_yaw)
            next_y = pose_y + signed_distance * ca.sin(mid_yaw)
            next_yaw = yaw + yaw_delta

            ref_index = 5 + 4 * index
            ref_x, ref_y, ref_yaw, target_speed = (
                parameters[ref_index + offset] for offset in range(4)
            )
            yaw_error = ca.atan2(
                ca.sin(ref_yaw - next_yaw), ca.cos(ref_yaw - next_yaw)
            )
            cost += config.position_weight * (
                (next_x - ref_x) ** 2 + (next_y - ref_y) ** 2
            )
            cost += config.heading_weight * yaw_error**2
            cost += config.speed_weight * (next_speed - target_speed) ** 2
            cost += config.acceleration_weight * acceleration**2
            cost += config.steering_weight * steering**2
            steering_before = (
                previous_steering
                if index == 0
                else controls[2 * (index - 1) + 1]
            )
            cost += config.steering_rate_weight * (steering - steering_before) ** 2

            # Exact rectangle support intervals on the four separating axes
            # (two ego axes and two obstacle axes). The max is piecewise smooth;
            # derivative tests cover the active-axis regions used by this demo.
            ego_forward = ca.vertcat(ca.cos(next_yaw), ca.sin(next_yaw))
            ego_lateral = ca.vertcat(-ca.sin(next_yaw), ca.cos(next_yaw))
            ego_center = ca.vertcat(
                next_x + axle_to_center * ca.cos(next_yaw),
                next_y + axle_to_center * ca.sin(next_yaw),
            )
            for obstacle in world.obstacles:
                obstacle_yaw = obstacle.yaw
                obstacle_forward = ca.vertcat(
                    float(np.cos(obstacle_yaw)), float(np.sin(obstacle_yaw))
                )
                obstacle_lateral = ca.vertcat(
                    -float(np.sin(obstacle_yaw)), float(np.cos(obstacle_yaw))
                )
                obstacle_center = ca.vertcat(*obstacle.center)
                obstacle_half_length = obstacle.length / 2.0
                obstacle_half_width = obstacle.width / 2.0
                axes = (
                    ego_forward,
                    ego_lateral,
                    obstacle_forward,
                    obstacle_lateral,
                )
                axis_gaps: list[Any] = []
                for axis in axes:
                    ego_center_projection = ca.dot(axis, ego_center)
                    ego_radius = (
                        half_length * ca.fabs(ca.dot(axis, ego_forward))
                        + half_width * ca.fabs(ca.dot(axis, ego_lateral))
                    )
                    obstacle_center_projection = ca.dot(axis, obstacle_center)
                    obstacle_radius = (
                        obstacle_half_length
                        * ca.fabs(ca.dot(axis, obstacle_forward))
                        + obstacle_half_width
                        * ca.fabs(ca.dot(axis, obstacle_lateral))
                    )
                    ego_min = ego_center_projection - ego_radius
                    ego_max = ego_center_projection + ego_radius
                    obstacle_min = obstacle_center_projection - obstacle_radius
                    obstacle_max = obstacle_center_projection + obstacle_radius
                    axis_gaps.append(obstacle_min - ego_max)
                    axis_gaps.append(ego_min - obstacle_max)
                clearance_terms.append(ca.mmax(ca.vertcat(*axis_gaps)) - margin)

            # World-boundary support constraints, matching the numeric SAT path.
            extent_x = half_length * ca.fabs(ca.cos(next_yaw)) + half_width * ca.fabs(ca.sin(next_yaw))
            extent_y = half_length * ca.fabs(ca.sin(next_yaw)) + half_width * ca.fabs(ca.cos(next_yaw))
            clearance_terms.extend(
                (
                    ego_center[0] - extent_x - xmin - margin,
                    xmax - margin - ego_center[0] - extent_x,
                    ego_center[1] - extent_y - ymin - margin,
                    ymax - margin - ego_center[1] - extent_y,
                )
            )

            speed_terms.extend((next_speed, config.max_speed_mps - next_speed))
            steering_limit = config.max_steering_rate_rad_s * config.dt
            steering_terms.append(steering_limit - ca.fabs(steering - steering_before))
            pose_x, pose_y, yaw, speed = next_x, next_y, next_yaw, next_speed

        goal_yaw_error = ca.atan2(
            ca.sin(goal.yaw - yaw), ca.cos(goal.yaw - yaw)
        )
        cost += config.terminal_position_weight * (
            (pose_x - goal.x) ** 2 + (pose_y - goal.y) ** 2
        )
        cost += config.terminal_heading_weight * goal_yaw_error**2
        cost += 8.0 * speed**2

        constraint_vector = ca.vertcat(
            *(clearance_terms + speed_terms + steering_terms)
        )
        objective_gradient = ca.gradient(cost, controls)
        constraint_jacobian = ca.jacobian(constraint_vector, controls)
        name = f"ihateparking_mpc_ad_{id(self)}"
        self._function = ca.Function(
            name,
            [controls, parameters],
            [cost, constraint_vector, objective_gradient, constraint_jacobian],
        )
        self.constraint_count = int(constraint_vector.numel())

    def evaluate(
        self, controls: np.ndarray, parameters: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        objective, constraints, objective_gradient, constraint_jacobian = self._function(
            np.asarray(controls, dtype=float), np.asarray(parameters, dtype=float)
        )
        return (
            float(np.asarray(objective, dtype=float).reshape(-1)[0]),
            np.asarray(constraints, dtype=float).reshape(-1),
            np.asarray(objective_gradient, dtype=float).reshape(-1),
            np.asarray(constraint_jacobian, dtype=float),
        )
