"""A small Hybrid A* planner for nonholonomic parking manoeuvres."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Optional

from .geometry import Pose, angle_error, distance
from .world import ParkingWorld


@dataclass(frozen=True)
class PlannerConfig:
    xy_resolution: float = 0.45
    yaw_bins: int = 24
    primitive_length: float = 0.75
    primitive_sample: float = 0.15
    steering_samples: int = 5
    reverse_penalty: float = 0.35
    gear_switch_penalty: float = 1.8
    steering_change_penalty: float = 0.08
    goal_position_tolerance: float = 0.35
    goal_yaw_tolerance: float = math.radians(12.0)
    max_expansions: int = 100_000


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
            if self.world.is_collision(next_pose):
                return []
            states.append(next_pose)
            current = next_pose
        return states

    def plan(self, start: Pose | None = None, goal: Pose | None = None) -> PlanResult:
        start = start or self.world.start_pose
        goal = goal or self.world.slot.goal_pose(self.world.vehicle)
        if self.world.is_collision(start):
            return PlanResult(False, [], [], [], math.inf, 0, "start pose is in collision")

        nodes: list[_Node] = [_Node(start, 0.0, None, 0, 0.0)]
        queue: list[tuple[float, int]] = [(self._heuristic(start, goal), 0)]
        best_cost: dict[tuple[int, int, int, int], float] = {(self._key(start) + (0,)): 0.0}
        expanded = 0
        goal_index: Optional[int] = None

        while queue and expanded < self.config.max_expansions:
            _, index = heapq.heappop(queue)
            node = nodes[index]
            expanded += 1
            if distance(node.pose, goal) <= self.config.goal_position_tolerance and abs(angle_error(goal.yaw, node.pose.yaw)) <= self.config.goal_yaw_tolerance:
                goal_index = index
                break

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
        return PlanResult(True, poses, gears, steerings, nodes[goal_index].cost, expanded, "path found")
