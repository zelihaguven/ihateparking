"""Closed-loop vehicle simulation and a reverse-aware Pure Pursuit controller."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .geometry import Pose, angle_error, distance, wrap_angle
from .planner import PlanResult
from .world import ParkingWorld


@dataclass(frozen=True)
class ControllerConfig:
    dt: float = 0.05
    speed: float = 0.8
    lookahead: float = 0.6
    waypoint_tolerance: float = 0.45
    steering_gain: float = 1.4
    max_time: float = 55.0
    goal_position_tolerance: float = 0.35
    goal_yaw_tolerance: float = math.radians(12.0)


@dataclass
class SimulationResult:
    success: bool
    poses: list[Pose]
    steerings: list[float]
    gears: list[int]
    time: float
    collisions: int
    message: str

    @property
    def distance_travelled(self) -> float:
        return sum(distance(a, b) for a, b in zip(self.poses, self.poses[1:]))


def _target_index(path: list[Pose], current_index: int, pose: Pose, lookahead: float) -> int:
    if not path:
        return 0
    closest = current_index
    closest_distance = math.inf
    for i in range(current_index, len(path)):
        d = distance(path[i], pose)
        if d < closest_distance:
            closest, closest_distance = i, d
    travelled = 0.0
    target = closest
    for i in range(closest + 1, len(path)):
        travelled += distance(path[i - 1], path[i])
        target = i
        if travelled >= lookahead:
            break
    return target


def pure_pursuit(pose: Pose, target: Pose, gear: int, wheelbase: float, lookahead: float, max_steering: float) -> float:
    """Reverse-aware geometric steering law."""
    dx = target.x - pose.x
    dy = target.y - pose.y
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    effective_lookahead = max(lookahead, math.hypot(local_x, local_y), 0.25)
    curvature = 2.0 * local_y / (effective_lookahead * effective_lookahead)
    if gear < 0:
        curvature = -curvature
    heading_term = 0.25 * angle_error(target.yaw, pose.yaw)
    steering = math.atan(wheelbase * curvature) + heading_term
    return max(-max_steering, min(max_steering, steering))


def simulate_closed_loop(world: ParkingWorld, plan: PlanResult, config: ControllerConfig | None = None) -> SimulationResult:
    cfg = config or ControllerConfig()
    if not plan.success or len(plan.poses) < 2:
        return SimulationResult(False, [world.start_pose], [], [], 0.0, 0, "cannot simulate an empty plan")

    pose = world.start_pose
    poses = [pose]
    steerings: list[float] = []
    gears: list[int] = []
    waypoint = 1
    elapsed = 0.0
    collisions = 0

    while elapsed < cfg.max_time:
        while waypoint < len(plan.poses) - 1 and distance(pose, plan.poses[waypoint]) < cfg.waypoint_tolerance:
            waypoint += 1
        goal = world.slot.goal_pose(world.vehicle)
        if world.at_goal(
            pose,
            position_tolerance=cfg.goal_position_tolerance,
            yaw_tolerance=cfg.goal_yaw_tolerance,
        ):
            return SimulationResult(True, poses, steerings, gears, elapsed, collisions, "parking completed")
        if waypoint >= len(plan.poses):
            break

        final_approach = waypoint >= len(plan.poses) - 1
        if final_approach:
            # The planner may stop inside its goal tolerance. Use the exact
            # slot pose for a slow, receding-horizon capture rather than
            # driving past the last graph node.
            target = goal
            dx = target.x - pose.x
            dy = target.y - pose.y
            local_x = math.cos(pose.yaw) * dx + math.sin(pose.yaw) * dy
            gear = -1 if local_x < -0.05 else 1
            speed = cfg.speed * 0.45
        else:
            gear = plan.gears[min(waypoint, len(plan.gears) - 1)]
            target_index = _target_index(plan.poses, max(1, waypoint), pose, cfg.lookahead)
            target = plan.poses[target_index]
            speed = cfg.speed
        steering = pure_pursuit(
            pose,
            target,
            gear,
            world.vehicle.wheelbase,
            cfg.lookahead,
            world.vehicle.max_steering,
        )
        signed_speed = speed * (1 if gear >= 0 else -1)
        yaw_rate = signed_speed / world.vehicle.wheelbase * math.tan(steering)
        next_pose = Pose(
            pose.x + signed_speed * math.cos(pose.yaw) * cfg.dt,
            pose.y + signed_speed * math.sin(pose.yaw) * cfg.dt,
            wrap_angle(pose.yaw + yaw_rate * cfg.dt),
        )
        if world.is_collision(next_pose):
            collisions += 1
            return SimulationResult(False, poses, steerings, gears, elapsed, collisions, "controller entered a collision")
        pose = next_pose
        poses.append(pose)
        steerings.append(steering)
        gears.append(gear)
        elapsed += cfg.dt

    return SimulationResult(False, poses, steerings, gears, elapsed, collisions, "controller timed out before reaching the slot")
