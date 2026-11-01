"""The world the parking stack drives in, behind one interface.

The mission loop only ever sees its own map and pose estimate. What it
does not see (the true pose, how the car actually moves, what the LiDAR
returns, whether it touched something, and the scoring) lives in a plant:

- ``SimPlant``: the built-in 2-D simulator (kinematic bicycle, ray-cast
  LiDAR over rectangles, an independent collision guard);
- ``ihateparking.metadrive_bridge.MetaDrivePlant``: MetaDrive (Bullet
  physics, MetaDrive's LiDAR).
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .geometry import Pose, wrap_angle
from .lidar import LidarConfig, LidarScan, scan
from .procedural_lot import ProceduralLot
from .world import ParkingWorld, VehicleSpec


class Plant(Protocol):
    vehicle: VehicleSpec
    dt: float  # seconds per ``apply``

    def true_pose(self) -> Pose:
        """Rear-axle pose in the world (for the simulation, never for the stack)."""

    def scan(self) -> LidarScan:
        """One LiDAR scan from where the car truly is."""

    def apply(self, steering: float, speed: float) -> bool:
        """Drive for ``dt`` towards a steering angle and a signed speed; False if the
        car was stopped from touching anything."""

    def odometry(self) -> tuple[float, float, float | None]:
        """(signed speed, steering, yaw rate or None) the car actually had during
        the last ``apply``, as its wheel, steering and gyro sensors would see them
        before noise. None: no gyro, heading comes from the steering angle."""

    def evaluate(self, pose: Pose) -> dict:
        """Score a final rear-axle pose against the ground truth."""


class SimPlant:
    """The built-in simulator: kinematic bicycle and ray-cast LiDAR over the lot's rectangles."""

    def __init__(self, lot: ProceduralLot, dt: float, lidar: LidarConfig, rng: np.random.Generator):
        self.lot = lot
        self.vehicle = lot.vehicle
        self.dt = dt
        self.lidar = lidar
        self.rng = rng
        any_slot = lot.bays[0].slot if lot.bays else None
        self.world = ParkingWorld(lot.bounds, list(lot.obstacles), any_slot, lot.start, lot.vehicle)
        self._pose = lot.start
        self._last = (0.0, 0.0)

    def true_pose(self) -> Pose:
        return self._pose

    def scan(self) -> LidarScan:
        return scan(self._pose, self.lot.obstacles, self.vehicle, self.lidar, self.rng)

    def apply(self, steering: float, speed: float) -> bool:
        pose = self._pose
        next_pose = Pose(
            pose.x + speed * math.cos(pose.yaw) * self.dt,
            pose.y + speed * math.sin(pose.yaw) * self.dt,
            wrap_angle(pose.yaw + speed / self.vehicle.wheelbase * math.tan(steering) * self.dt),
        )
        # The guard stops the car before contact, independently of the stack.
        if self.world.is_collision(next_pose, extra_margin=-self.world.collision_margin):
            return False
        self._pose = next_pose
        self._last = (speed, steering)
        return True

    def odometry(self) -> tuple[float, float, float | None]:
        return self._last[0], self._last[1], None

    def evaluate(self, pose: Pose) -> dict:
        from .mission import evaluate_parking

        return evaluate_parking(self.lot, pose)
