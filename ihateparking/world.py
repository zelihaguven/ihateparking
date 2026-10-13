"""Parking-world definitions and collision queries."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import Pose, oriented_box, point_in_bounds, polygons_intersect, vehicle_polygon


@dataclass(frozen=True)
class VehicleSpec:
    length: float = 4.4
    width: float = 1.9
    wheelbase: float = 2.7
    rear_overhang: float = 0.7
    max_steering: float = math.radians(32.0)


@dataclass(frozen=True)
class RectangleObstacle:
    name: str
    center: tuple[float, float]
    length: float
    width: float
    yaw: float = 0.0

    def polygon(self, margin: float = 0.0) -> np.ndarray:
        return oriented_box(self.center, self.length + 2 * margin, self.width + 2 * margin, self.yaw)


@dataclass(frozen=True)
class ParkingSlot:
    center: tuple[float, float]
    length: float
    width: float
    yaw: float = 0.0

    def polygon(self) -> np.ndarray:
        return oriented_box(self.center, self.length, self.width, self.yaw)

    def goal_pose(self, vehicle: VehicleSpec | None = None) -> Pose:
        """Return a rear-axle pose whose footprint is centered in the slot."""
        if vehicle is None:
            return Pose(self.center[0], self.center[1], self.yaw)
        axle_to_footprint_center = (vehicle.length - 2.0 * vehicle.rear_overhang) / 2.0
        return Pose(
            self.center[0] - axle_to_footprint_center * math.cos(self.yaw),
            self.center[1] - axle_to_footprint_center * math.sin(self.yaw),
            self.yaw,
        )


@dataclass
class ParkingWorld:
    bounds: tuple[float, float, float, float]
    obstacles: list[RectangleObstacle]
    slot: ParkingSlot
    start_pose: Pose
    vehicle: VehicleSpec = VehicleSpec()
    collision_margin: float = 0.12

    def vehicle_polygon(self, pose: Pose) -> np.ndarray:
        return vehicle_polygon(pose, self.vehicle.length, self.vehicle.width, self.vehicle.rear_overhang)

    def is_collision(self, pose: Pose, extra_margin: float = 0.0) -> bool:
        footprint = self.vehicle_polygon(pose)
        if not point_in_bounds(footprint, self.bounds, margin=self.collision_margin + extra_margin):
            return True
        for obstacle in self.obstacles:
            if polygons_intersect(footprint, obstacle.polygon(margin=self.collision_margin + extra_margin)):
                return True
        return False

    def at_goal(self, pose: Pose, position_tolerance: float = 0.35, yaw_tolerance: float = math.radians(12.0)) -> bool:
        goal = self.slot.goal_pose(self.vehicle)
        position_error = math.hypot(pose.x - goal.x, pose.y - goal.y)
        yaw_error = abs((pose.yaw - goal.yaw + math.pi) % (2 * math.pi) - math.pi)
        return position_error <= position_tolerance and yaw_error <= yaw_tolerance


def default_world() -> ParkingWorld:
    """A perpendicular parking scenario with a free slot between two parked cars."""
    # Other cars form a top row. The target slot is the middle gap.
    obstacles = [
        RectangleObstacle("parked_left", center=(3.0, 8.0), length=4.5, width=2.0, yaw=math.pi / 2),
        RectangleObstacle("parked_right", center=(12.0, 8.0), length=4.5, width=2.0, yaw=math.pi / 2),
        # A curb-like obstacle at the far end leaves a realistic stopping margin.
        RectangleObstacle("back_curb", center=(7.5, 11.95), length=13.0, width=0.25, yaw=0.0),
    ]
    slot = ParkingSlot(center=(7.5, 8.0), length=4.5, width=2.65, yaw=math.pi / 2)
    return ParkingWorld(
        bounds=(0.0, 16.0, 0.0, 12.0),
        obstacles=obstacles,
        slot=slot,
        start_pose=Pose(14.0, 3.0, math.pi),
    )
