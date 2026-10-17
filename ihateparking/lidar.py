"""Simulated 2-D LiDAR: ray casting against the ground-truth obstacles.

The scanner sits at the centre of the vehicle footprint. Each ray returns the
distance to the first obstacle edge it meets, with Gaussian range noise and
random dropouts; rays that hit nothing within range report ``max_range`` and
``hit=False``. Occlusion is physical: a parked car hides what is behind it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .geometry import Pose
from .world import RectangleObstacle, VehicleSpec


@dataclass(frozen=True)
class LidarConfig:
    beams: int = 720
    max_range: float = 25.0
    range_noise_std: float = 0.02
    dropout: float = 0.01


@dataclass(frozen=True)
class LidarScan:
    origin: tuple[float, float]
    angles: np.ndarray  # world-frame beam angles (rad)
    ranges: np.ndarray  # metres
    hits: np.ndarray  # bool: True if the beam returned from an obstacle


def sensor_origin(pose: Pose, vehicle: VehicleSpec) -> tuple[float, float]:
    offset = vehicle.length / 2.0 - vehicle.rear_overhang
    return pose.x + offset * math.cos(pose.yaw), pose.y + offset * math.sin(pose.yaw)


def scan(
    pose: Pose,
    obstacles: Sequence[RectangleObstacle],
    vehicle: VehicleSpec,
    config: LidarConfig = LidarConfig(),
    rng: np.random.Generator | None = None,
) -> LidarScan:
    ox, oy = sensor_origin(pose, vehicle)
    angles = pose.yaw + np.linspace(-math.pi, math.pi, config.beams, endpoint=False)
    dx, dy = np.cos(angles), np.sin(angles)
    best = np.full(config.beams, np.inf)
    for obstacle in obstacles:
        cx, cy = obstacle.center
        reach = math.hypot(obstacle.length, obstacle.width) / 2.0
        if math.hypot(cx - ox, cy - oy) - reach > config.max_range:
            continue
        c, s = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
        # Ray origin and direction in the obstacle frame; slab intersection.
        px, py = c * (ox - cx) + s * (oy - cy), -s * (ox - cx) + c * (oy - cy)
        ux, uy = c * dx + s * dy, -s * dx + c * dy
        hx, hy = obstacle.length / 2.0, obstacle.width / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            tx1, tx2 = (-hx - px) / ux, (hx - px) / ux
            ty1, ty2 = (-hy - py) / uy, (hy - py) / uy
        # A beam parallel to a slab never crosses it: it either always lies
        # within the slab (no constraint) or never does (no hit at all).
        x_parallel, y_parallel = ux == 0.0, uy == 0.0
        tx1 = np.where(x_parallel, -np.inf, tx1)
        tx2 = np.where(x_parallel, np.inf, tx2)
        ty1 = np.where(y_parallel, -np.inf, ty1)
        ty2 = np.where(y_parallel, np.inf, ty2)
        t_near = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
        t_far = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
        outside = (x_parallel & (abs(px) > hx)) | (y_parallel & (abs(py) > hy))
        hit = (t_near <= t_far) & (t_far > 0.0) & ~outside
        distance = np.where(t_near > 0.0, t_near, 0.0)
        best = np.where(hit & (distance < best), distance, best)
    rng = rng or np.random.default_rng()
    hits = best <= config.max_range
    if config.dropout > 0:
        hits &= rng.random(config.beams) >= config.dropout
    noisy = best + rng.normal(0.0, config.range_noise_std, config.beams)
    ranges = np.where(hits, np.clip(noisy, 0.0, config.max_range), config.max_range)
    return LidarScan((ox, oy), angles, ranges, hits)
