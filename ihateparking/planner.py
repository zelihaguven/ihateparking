"""A small Hybrid A* planner for nonholonomic parking manoeuvres."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Optional

from .geometry import Pose, angle_error, distance
from .reeds_shepp import ReedsSheppPath, reeds_shepp_paths, sample_path
from .world import ParkingWorld


@dataclass(frozen=True)
class PlannerConfig:
    # Fine enough for multi-shot parallel manoeuvres in a 6 m bay.
    xy_resolution: float = 0.30
    yaw_bins: int = 48
    primitive_length: float = 0.50
    primitive_sample: float = 0.10
    steering_samples: int = 5
    reverse_penalty: float = 0.35
    gear_switch_penalty: float = 1.8
    steering_change_penalty: float = 0.08
    goal_position_tolerance: float = 0.35
    goal_yaw_tolerance: float = math.radians(12.0)
    max_expansions: int = 100_000
    # Extra clearance (m) beyond the world margin while searching, so closed-loop
    # tracking error does not turn a grazing plan into a guard rejection.
    clearance_margin: float = 0.05
    # Reeds-Shepp shot to the exact goal pose. It is tried on every expansion
    # within analytic_radius of the goal and every analytic_interval otherwise.
    analytic_expansion: bool = True
    analytic_interval: int = 10
    analytic_radius: float = 8.0
    analytic_candidates: int = 4
    # Analytic curves use a slightly larger turning radius than the steering
    # limit so the tracker keeps authority in reserve.
    analytic_steering_fraction: float = 0.85
    analytic_sample: float = 0.10
    # Close to the goal, also try full-lock curves, which fit tighter bays.
    analytic_full_lock_radius: float = 3.0
    # After a node first lands inside the goal tolerance, keep searching this
    # many expansions for an exact analytic connection before settling.
    exact_goal_extra_expansions: int = 3_000


@dataclass
class PlanResult:
    success: bool
    poses: list[Pose]
    gears: list[int]
    steerings: list[float]
    cost: float
    expanded: int
    message: str

    @property
    def path_length(self) -> float:
        return sum(distance(a, b) for a, b in zip(self.poses, self.poses[1:]))


@dataclass
class _Node:
    pose: Pose
    cost: float
    parent: Optional[int]
    gear: int
    steering: float


class HybridAStar:
    def __init__(self, world: ParkingWorld, config: PlannerConfig | None = None):
        self.world = world
        self.config = config or PlannerConfig()
        self._steering_values = self._make_steering_values()

    def _make_steering_values(self) -> list[float]:
        maximum = self.world.vehicle.max_steering
        if self.config.steering_samples == 1:
            return [0.0]
        return [
            -maximum + (2 * maximum * i / (self.config.steering_samples - 1))
            for i in range(self.config.steering_samples)
        ]

    def _blocked(self, pose: Pose) -> bool:
        return self.world.is_collision(pose, extra_margin=self.config.clearance_margin)

    def _analytic_radius(self, steering_fraction: float) -> float:
        steering = self.world.vehicle.max_steering * steering_fraction
        return self.world.vehicle.wheelbase / math.tan(steering)

    def _analytic_cost(self, path: ReedsSheppPath, from_gear: int) -> float:
        cfg = self.config
        cost = path.length + cfg.reverse_penalty * path.reverse_length + cfg.gear_switch_penalty * path.gear_switches
        first_gear = 1 if path.lengths[0] > 0 else -1
        if from_gear not in (0, first_gear):
            cost += cfg.gear_switch_penalty
        return cost

    def _analytic_shot(self, node: "_Node", goal: Pose) -> Optional[tuple[list[Pose], list[int], list[float], float]]:
        """Try collision-free Reeds-Shepp connections, cheapest first."""
        cfg = self.config
        fractions = [cfg.analytic_steering_fraction]
        if distance(node.pose, goal) <= cfg.analytic_full_lock_radius and cfg.analytic_steering_fraction < 1.0:
            fractions.append(1.0)
        wheelbase = self.world.vehicle.wheelbase
        for fraction in fractions:
            paths = sorted(
                reeds_shepp_paths(node.pose, goal, self._analytic_radius(fraction)),
                key=lambda item: self._analytic_cost(item, node.gear),
            )[: cfg.analytic_candidates]
            for path in paths:
                samples = sample_path(node.pose, path, cfg.analytic_sample)
                if any(self._blocked(sample.pose) for sample in samples):
                    continue
                poses, gears, steerings = self._decimate(samples, wheelbase)
                return poses, gears, steerings, self._analytic_cost(path, node.gear)
        return None

    def _decimate(self, samples, wheelbase: float) -> tuple[list[Pose], list[int], list[float]]:
        """Keep roughly primitive-length spacing, plus every cusp and the goal.

        Callers index plan poses as waypoints, so the analytic tail keeps the
        same spacing as graph nodes instead of the fine collision samples.
        """
        poses: list[Pose] = []
        gears: list[int] = []
        steerings: list[float] = []
        travelled = 0.0
        previous = None
        for index, sample in enumerate(samples):
            if previous is not None:
                travelled += distance(previous.pose, sample.pose)
            is_last = index == len(samples) - 1
            is_cusp = not is_last and samples[index + 1].gear != sample.gear
            if travelled >= self.config.primitive_length - 1e-9 or is_cusp or is_last:
                poses.append(sample.pose)
                gears.append(sample.gear)
                steerings.append(math.atan(wheelbase * sample.curvature))
                travelled = 0.0
            previous = sample
        return poses, gears, steerings

    def _key(self, pose: Pose) -> tuple[int, int, int]:
        cfg = self.config
        x_bin = round(pose.x / cfg.xy_resolution)
        y_bin = round(pose.y / cfg.xy_resolution)
        yaw = (pose.yaw + math.pi) % (2 * math.pi)
        yaw_bin = int(yaw / (2 * math.pi) * cfg.yaw_bins) % cfg.yaw_bins
        return x_bin, y_bin, yaw_bin

    def _heuristic(self, pose: Pose, goal: Pose) -> float:
        return distance(pose, goal) + 0.35 * abs(angle_error(goal.yaw, pose.yaw))

    def _propagate(self, pose: Pose, gear: int, steering: float) -> list[Pose]:
        cfg = self.config
        steps = max(2, math.ceil(cfg.primitive_length / cfg.primitive_sample))
        ds = gear * cfg.primitive_length / steps
        current = pose
        states = []
        for _ in range(steps):
            # Rear-axle bicycle model; signed ds naturally handles reverse.
            next_pose = Pose(
                current.x + ds * math.cos(current.yaw),
                current.y + ds * math.sin(current.yaw),
                current.yaw + ds / self.world.vehicle.wheelbase * math.tan(steering),
            )
            next_pose = Pose(next_pose.x, next_pose.y, (next_pose.yaw + math.pi) % (2 * math.pi) - math.pi)
            if self._blocked(next_pose):
                return []
            states.append(next_pose)
            current = next_pose
        return states

    def plan(self, start: Pose | None = None, goal: Pose | None = None) -> PlanResult:
        start = start or self.world.start_pose
        goal = goal or self.world.slot.goal_pose(self.world.vehicle)
        if self.world.is_collision(start):
            return PlanResult(False, [], [], [], math.inf, 0, "start pose is in collision")
        # A goal blocked by e.g. an inserted dynamic obstacle cannot be reached
        # exactly; fall back to the tolerance-only search instead of failing.
        exact_goal_reachable = not self._blocked(goal)

        nodes: list[_Node] = [_Node(start, 0.0, None, 0, 0.0)]
        queue: list[tuple[float, int]] = [(self._heuristic(start, goal), 0)]
        best_cost: dict[tuple[int, int, int, int], float] = {(self._key(start) + (0,)): 0.0}
        expanded = 0
        goal_index: Optional[int] = None
        analytic_tail: Optional[tuple[list[Pose], list[int], list[float], float]] = None
        cfg = self.config
        tolerance_index: Optional[int] = None
        expansion_limit = cfg.max_expansions

        while queue and expanded < expansion_limit:
            _, index = heapq.heappop(queue)
            node = nodes[index]
            expanded += 1
            if cfg.analytic_expansion and exact_goal_reachable and (
                distance(node.pose, goal) <= cfg.analytic_radius or expanded % cfg.analytic_interval == 1
            ):
                analytic_tail = self._analytic_shot(node, goal)
                if analytic_tail is not None:
                    goal_index = index
                    break
            if distance(node.pose, goal) <= cfg.goal_position_tolerance and abs(angle_error(goal.yaw, node.pose.yaw)) <= cfg.goal_yaw_tolerance:
                if not (cfg.analytic_expansion and exact_goal_reachable):
                    goal_index = index
                    break
                if tolerance_index is None:
                    tolerance_index = index
                    expansion_limit = min(cfg.max_expansions, expanded + cfg.exact_goal_extra_expansions)

            for gear in (1, -1):
                for steering in self._steering_values:
                    states = self._propagate(node.pose, gear, steering)
                    if not states:
                        continue
                    next_pose = states[-1]
                    switch_cost = self.config.gear_switch_penalty if node.gear not in (0, gear) else 0.0
                    steering_cost = self.config.steering_change_penalty * abs(steering - node.steering)
                    motion_cost = self.config.primitive_length * (1.0 + (self.config.reverse_penalty if gear < 0 else 0.0))
                    new_cost = node.cost + motion_cost + switch_cost + steering_cost
                    state_key = self._key(next_pose) + (gear,)
                    if new_cost >= best_cost.get(state_key, math.inf):
                        continue
                    best_cost[state_key] = new_cost
                    new_index = len(nodes)
                    nodes.append(_Node(next_pose, new_cost, index, gear, steering))
                    estimate = new_cost + self._heuristic(next_pose, goal)
                    heapq.heappush(queue, (estimate, new_index))

        if goal_index is None:
            goal_index = tolerance_index
        if goal_index is None:
            message = f"no path found after {expanded} expansions"
            return PlanResult(False, [], [], [], math.inf, expanded, message)

        poses: list[Pose] = []
        gears: list[int] = []
        steerings: list[float] = []
        cursor: Optional[int] = goal_index
        while cursor is not None:
            node = nodes[cursor]
            poses.append(node.pose)
            gears.append(node.gear)
            steerings.append(node.steering)
            cursor = node.parent
        poses.reverse()
        gears.reverse()
        steerings.reverse()
        cost = nodes[goal_index].cost
        message = "path found"
        if analytic_tail is not None:
            tail_poses, tail_gears, tail_steerings, tail_cost = analytic_tail
            poses.extend(tail_poses)
            gears.extend(tail_gears)
            steerings.extend(tail_steerings)
            cost += tail_cost
            message = "path found (exact goal via Reeds-Shepp analytic expansion)"
        return PlanResult(True, poses, gears, steerings, cost, expanded, message)
