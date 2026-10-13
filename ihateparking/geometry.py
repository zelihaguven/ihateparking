"""2-D geometry primitives used by the ihateparking simulator."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_error(target: float, actual: float) -> float:
    return wrap_angle(target - actual)


def distance(a: Pose, b: Pose) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def transform_points(local_points: Iterable[tuple[float, float]], pose: Pose) -> np.ndarray:
    """Transform local 2-D points into world coordinates."""
    points = np.asarray(list(local_points), dtype=float)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return points @ rotation.T + np.array([pose.x, pose.y])


def oriented_box(center: tuple[float, float], length: float, width: float, yaw: float = 0.0) -> np.ndarray:
    """Return rectangle corners counter-clockwise."""
    cx, cy = center
    local = [
        (-length / 2, -width / 2),
        (length / 2, -width / 2),
        (length / 2, width / 2),
        (-length / 2, width / 2),
    ]
    return transform_points(local, Pose(cx, cy, yaw))


def vehicle_polygon(pose: Pose, length: float, width: float, rear_overhang: float = 0.0) -> np.ndarray:
    """Vehicle footprint; pose is defined at the rear axle."""
    front = length - rear_overhang
    local = [
        (-rear_overhang, -width / 2),
        (front, -width / 2),
        (front, width / 2),
        (-rear_overhang, width / 2),
    ]
    return transform_points(local, pose)


def _edge_normals(polygon: np.ndarray) -> list[np.ndarray]:
    normals = []
    for p, q in zip(polygon, np.vstack([polygon[1:], polygon[:1]])):
        edge = q - p
        normal = np.array([-edge[1], edge[0]], dtype=float)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals.append(normal / norm)
    return normals


def polygons_intersect(first: np.ndarray, second: np.ndarray, margin: float = 0.0) -> bool:
    """Separating-axis test for convex polygons."""
    for axis in _edge_normals(first) + _edge_normals(second):
        first_projection = first @ axis
        second_projection = second @ axis
        if first_projection.max() < second_projection.min() - margin:
            return False
        if second_projection.max() < first_projection.min() - margin:
            return False
    return True


def point_in_bounds(polygon: np.ndarray, bounds: tuple[float, float, float, float], margin: float = 0.0) -> bool:
    xmin, xmax, ymin, ymax = bounds
    return bool(
        polygon[:, 0].min() >= xmin + margin
        and polygon[:, 0].max() <= xmax - margin
        and polygon[:, 1].min() >= ymin + margin
        and polygon[:, 1].max() <= ymax - margin
    )
