"""Closed-loop vehicle simulation and a reverse-aware Pure Pursuit controller."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .geometry import Pose, angle_error, distance, wrap_angle
from .planner import PlanResult
from .world import ParkingWorld


@dataclass(frozen=True)
class ControllerConfig:
    dt: float = 0.05
    speed: float = 0.8
    lookahead: float = 0.6
    # Retained for API compatibility; segment tracking uses cusp_tolerance.
    waypoint_tolerance: float = 0.45
    steering_gain: float = 1.4
    max_time: float = 55.0
    goal_position_tolerance: float = 0.35
    goal_yaw_tolerance: float = math.radians(12.0)
    # A gear segment ends once the remaining along-track distance is below this.
    cusp_tolerance: float = 0.03
    # Speed ramps down linearly over this distance before every cusp and the goal.
    slowdown_distance: float = 1.0
    min_speed_fraction: float = 0.25
    # "feedforward": planned curvature plus lateral/heading error feedback.
    # "pure_pursuit": geometric lookahead law only (the MPC lab baseline).
    tracking_law: Literal["feedforward", "pure_pursuit"] = "feedforward"
    # Error-feedback gains per metre of travel (critically damped for 1 m).
    lateral_gain: float = 1.0
    heading_gain: float = 2.0


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


def pure_pursuit(pose: Pose, target: Pose, gear: int, wheelbase: float, lookahead: float, max_steering: float) -> float:
    """Reverse-aware geometric steering law for a rear-axle bicycle model.

    The arc through the target is tangent to the heading at the rear axle, so
    its curvature d(yaw)/ds is the same for either gear; only the heading
    correction flips sign because ds is negative in reverse.
    """
    dx = target.x - pose.x
    dy = target.y - pose.y
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    effective_lookahead = max(lookahead, math.hypot(local_x, local_y), 0.25)
    curvature = 2.0 * local_y / (effective_lookahead * effective_lookahead)
    direction = 1.0 if gear >= 0 else -1.0
    heading_term = 0.25 * direction * angle_error(target.yaw, pose.yaw)
    steering = math.atan(wheelbase * curvature) + heading_term
    return max(-max_steering, min(max_steering, steering))


@dataclass(frozen=True)
class _Segment:
    """One constant-gear stretch of a plan with cumulative arc length."""

    points: tuple[Pose, ...]
    gear: int
    arc: tuple[float, ...]
    # Planned steering into each point; index 0 is unused.
    steerings: tuple[float, ...]

    @property
    def length(self) -> float:
        return self.arc[-1]

    def tangent(self, index: int) -> tuple[float, float]:
        """Unit direction of travel at a point (backwards for reverse)."""
        yaw = self.points[index].yaw
        return self.gear * math.cos(yaw), self.gear * math.sin(yaw)


def _split_segments(plan: PlanResult) -> list[_Segment]:
    """Split a plan at every gear change; gears[i] is the gear into pose i."""
    segments: list[_Segment] = []
    index = 1
    while index < len(plan.poses):
        gear = plan.gears[index] if plan.gears[index] != 0 else 1
        end = index
        while end + 1 < len(plan.poses) and (plan.gears[end + 1] or 1) == gear:
            end += 1
        points = tuple(plan.poses[index - 1:end + 1])
        arc = [0.0]
        for a, b in zip(points, points[1:]):
            arc.append(arc[-1] + distance(a, b))
        steerings = (0.0,) + tuple(_planned_steering(plan, i) for i in range(index, end + 1))
        segments.append(_Segment(points, gear, tuple(arc), steerings))
        index = end + 1
    return segments


def _planned_steering(plan: PlanResult, index: int) -> float:
    return plan.steerings[index] if index < len(plan.steerings) else 0.0


def _feedforward_steering(segment: _Segment, s: float, pose: Pose, cfg: ControllerConfig, wheelbase: float) -> float:
    """Planned curvature plus feedback on lateral and heading error.

    With signed travel ds = gear * |ds|, the error dynamics are
    de_y/ds = sin(e_yaw) and de_yaw/ds = k - k_ref. The law
    k = k_ref - k_y * e_y - gear * k_yaw * e_yaw keeps both gears stable.
    """
    reference = _point_at(segment, s)
    span = next((i for i in range(1, len(segment.points)) if segment.arc[i] >= s), len(segment.points) - 1)
    reference_curvature = math.tan(segment.steerings[span]) / wheelbase
    c, sn = math.cos(reference.yaw), math.sin(reference.yaw)
    lateral_error = -sn * (pose.x - reference.x) + c * (pose.y - reference.y)
    heading_error = angle_error(pose.yaw, reference.yaw)
    curvature = reference_curvature - cfg.lateral_gain * lateral_error - segment.gear * cfg.heading_gain * heading_error
    return math.atan(wheelbase * curvature)


def _progress(segment: _Segment, cursor: int, pose: Pose) -> tuple[int, float]:
    """Advance the closest-point cursor and return the arc position of the pose."""
    window_end = min(len(segment.points), cursor + 6)
    cursor = min(range(cursor, window_end), key=lambda i: distance(segment.points[i], pose))
    tx, ty = segment.tangent(cursor)
    anchor = segment.points[cursor]
    along = (pose.x - anchor.x) * tx + (pose.y - anchor.y) * ty
    return cursor, segment.arc[cursor] + along


def _point_at(segment: _Segment, s: float) -> Pose:
    """Interpolate the segment at arc length s, extending past its end."""
    if s >= segment.length:
        end = segment.points[-1]
        tx, ty = segment.tangent(len(segment.points) - 1)
        extra = s - segment.length
        return Pose(end.x + tx * extra, end.y + ty * extra, end.yaw)
    s = max(0.0, s)
    for i in range(1, len(segment.points)):
        if segment.arc[i] >= s:
            span = segment.arc[i] - segment.arc[i - 1]
            ratio = 0.0 if span <= 1e-12 else (s - segment.arc[i - 1]) / span
            a, b = segment.points[i - 1], segment.points[i]
            return Pose(
                a.x + (b.x - a.x) * ratio,
                a.y + (b.y - a.y) * ratio,
                wrap_angle(a.yaw + angle_error(b.yaw, a.yaw) * ratio),
            )
    return segment.points[-1]


@dataclass(frozen=True)
class TrackingCommand:
    """One control step: gear (+1/-1), steering (rad) and unsigned speed (m/s)."""

    gear: int
    steering: float
    speed: float
    remaining_m: float
    final_segment: bool


class PlanTracker:
    """Stateful segment-by-segment plan follower.

    Call command() once per control cycle with the measured pose. It returns
    None once every gear segment has been driven to its end; the caller then
    decides whether the final pose is inside its goal tolerance.
    """

    def __init__(self, plan: PlanResult, wheelbase: float, max_steering: float, config: ControllerConfig | None = None):
        self.config = config or ControllerConfig()
        self.wheelbase = wheelbase
        self.max_steering = max_steering
        self.segments = _split_segments(plan) if plan.success and len(plan.poses) >= 2 else []
        self.segment_index = 0
        self._cursor = 0

    @property
    def finished(self) -> bool:
        return self.segment_index >= len(self.segments)

    def command(self, pose: Pose) -> TrackingCommand | None:
        cfg = self.config
        while not self.finished:
            segment = self.segments[self.segment_index]
            self._cursor, s = _progress(segment, self._cursor, pose)
            remaining = segment.length - s
            if remaining > cfg.cusp_tolerance:
                break
            self.segment_index += 1
            self._cursor = 0
        if self.finished:
            return None
        if cfg.tracking_law == "pure_pursuit":
            target = _point_at(segment, s + cfg.lookahead)
            steering = pure_pursuit(pose, target, segment.gear, self.wheelbase, cfg.lookahead, self.max_steering)
        else:
            steering = _feedforward_steering(segment, s, pose, cfg, self.wheelbase)
            steering = max(-self.max_steering, min(self.max_steering, steering))
        ramp = min(1.0, remaining / cfg.slowdown_distance) if cfg.slowdown_distance > 0 else 1.0
        speed = cfg.speed * max(cfg.min_speed_fraction, ramp)
        final_segment = self.segment_index == len(self.segments) - 1
        return TrackingCommand(segment.gear, steering, speed, remaining, final_segment)


def simulate_closed_loop(world: ParkingWorld, plan: PlanResult, config: ControllerConfig | None = None) -> SimulationResult:
    """Track a plan one gear segment at a time, stopping at every cusp.

    Each segment is followed with the configured tracking law; neither the
    feedback reference nor the Pure Pursuit lookahead crosses into the next
    segment. Speed ramps down before each cusp, and every proposed pose is
    checked against the world before it is applied. The plan is driven to its
    end, which Reeds-Shepp expansion places exactly on the goal, and only the
    final pose is judged against the goal tolerance.
    """
    cfg = config or ControllerConfig()
    if not plan.success or len(plan.poses) < 2:
        return SimulationResult(False, [world.start_pose], [], [], 0.0, 0, "cannot simulate an empty plan")

    vehicle = world.vehicle
    tracker = PlanTracker(plan, vehicle.wheelbase, vehicle.max_steering, cfg)
    pose = world.start_pose
    poses = [pose]
    steerings: list[float] = []
    gears: list[int] = []
    elapsed = 0.0

    def at_goal(candidate: Pose) -> bool:
        return world.at_goal(
            candidate,
            position_tolerance=cfg.goal_position_tolerance,
            yaw_tolerance=cfg.goal_yaw_tolerance,
        )

    while (command := tracker.command(pose)) is not None:
        if elapsed >= cfg.max_time:
            return SimulationResult(False, poses, steerings, gears, elapsed, 0, "controller timed out before reaching the slot")
        signed_speed = command.speed * command.gear
        next_pose = Pose(
            pose.x + signed_speed * math.cos(pose.yaw) * cfg.dt,
            pose.y + signed_speed * math.sin(pose.yaw) * cfg.dt,
            wrap_angle(pose.yaw + signed_speed / vehicle.wheelbase * math.tan(command.steering) * cfg.dt),
        )
        if world.is_collision(next_pose):
            return SimulationResult(False, poses, steerings, gears, elapsed, 1, "controller entered a collision")
        pose = next_pose
        poses.append(pose)
        steerings.append(command.steering)
        gears.append(command.gear)
        elapsed += cfg.dt

    if at_goal(pose):
        return SimulationResult(True, poses, steerings, gears, elapsed, 0, "parking completed")
    return SimulationResult(False, poses, steerings, gears, elapsed, 0, "plan executed but the final pose is outside the goal tolerance")
