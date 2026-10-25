"""Dead reckoning from noisy wheel and steering sensors.

A real car does not know where it is: it integrates what its wheel encoders
and steering-angle sensor report, and both are imperfect. The wheel
circumference is off by a fixed fraction (tyre pressure, wear), each wheel
tick adds noise, and the steering sensor has a fixed bias plus noise. The
estimate therefore drifts with the distance driven, and a heading bias of a
fraction of a degree grows into decimetres of position error over a lot.

``Odometry`` draws one set of fixed errors per mission and then integrates
the measured motion; ``place_scan`` re-places a LiDAR scan, which the sensor
takes relative to the car, at the estimated pose.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .geometry import Pose, wrap_angle
from .lidar import LidarScan, sensor_origin
from .world import VehicleSpec


@dataclass(frozen=True)
class OdometryConfig:
    distance_scale_std: float = 0.01  # fixed wheel-circumference error, fraction of distance
    distance_noise_std: float = 0.02  # per metre driven (random walk), metres
    steering_bias_std_deg: float = 0.3  # fixed steering-sensor offset
    steering_noise_std_deg: float = 0.3  # per measurement


class Odometry:
    def __init__(self, vehicle: VehicleSpec, config: OdometryConfig, rng: np.random.Generator):
        self.vehicle = vehicle
        self.config = config
        self.rng = rng
        self.scale = 1.0 + rng.normal(0.0, config.distance_scale_std)
        self.steering_bias = math.radians(rng.normal(0.0, config.steering_bias_std_deg))

    def step(self, estimate: Pose, signed_speed: float, steering: float, dt: float) -> Pose:
        """Advance ``estimate`` by the motion the sensors report for one step."""
        cfg = self.config
        travelled = signed_speed * dt
        measured = travelled * self.scale + self.rng.normal(0.0, cfg.distance_noise_std * math.sqrt(abs(travelled)))
        angle = steering + self.steering_bias + math.radians(self.rng.normal(0.0, cfg.steering_noise_std_deg))
        return Pose(
            estimate.x + measured * math.cos(estimate.yaw),
            estimate.y + measured * math.sin(estimate.yaw),
            wrap_angle(estimate.yaw + measured / self.vehicle.wheelbase * math.tan(angle)),
        )


def place_scan(measured: LidarScan, true_pose: Pose, estimate: Pose, vehicle: VehicleSpec) -> LidarScan:
    """The scan as the car sees it: taken relative to the car, placed at its estimated pose."""
    return replace(
        measured,
        origin=sensor_origin(estimate, vehicle),
        angles=measured.angles - true_pose.yaw + estimate.yaw,
    )
