"""Obstacle-aware distance-to-goal field for the Hybrid A* heuristic.

This is the "holonomic with obstacles" heuristic of Dolgov et al. (2008): an
8-connected grid Dijkstra from the goal, treating the vehicle as a disc of
half its width. It ignores heading and turning radius, so it underestimates
true path length, but it knows that reaching a bay behind a row of cars means
driving around the row. Without it, a straight-line heuristic pulls the
search into the back of the row.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .world import ParkingWorld

_SQRT2 = math.sqrt(2.0)
_NEIGHBOURS = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2),
)


@dataclass(frozen=True)
class DistanceField:
    origin: tuple[float, float]
    resolution: float
    distances: np.ndarray  # metres; inf where blocked or unreachable

    def lookup(self, x: float, y: float) -> float:
        col = int(round((x - self.origin[0]) / self.resolution))
        row = int(round((y - self.origin[1]) / self.resolution))
        rows, cols = self.distances.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return math.inf
        return float(self.distances[row, col])


def _blocked_mask(world: ParkingWorld, xs: np.ndarray, ys: np.ndarray, inflation: float) -> np.ndarray:
    xmin, xmax, ymin, ymax = world.bounds
    blocked = (xs < xmin + inflation) | (xs > xmax - inflation) | (ys < ymin + inflation) | (ys > ymax - inflation)
    for obstacle in world.obstacles:
        c, s = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
        dx, dy = xs - obstacle.center[0], ys - obstacle.center[1]
        local_x = np.abs(c * dx + s * dy) - obstacle.length / 2.0
        local_y = np.abs(-s * dx + c * dy) - obstacle.width / 2.0
        outside = np.hypot(np.maximum(local_x, 0.0), np.maximum(local_y, 0.0))
        blocked |= outside < inflation
    return blocked


def build_distance_field(world: ParkingWorld, goal_xy: tuple[float, float], resolution: float = 0.5, inflation: float | None = None) -> DistanceField:
    """Dijkstra over free cells, seeded from the cell nearest the goal.

    If the goal cell itself is blocked (a rear-axle goal can sit closer to a
    neighbour than the disc radius), the search starts from the nearest free
    cells around it, offset by their distance to the goal.
    """
    if inflation is None:
        inflation = world.vehicle.width / 2.0
    xmin, xmax, ymin, ymax = world.bounds
    cols = int(math.floor((xmax - xmin) / resolution)) + 1
    rows = int(math.floor((ymax - ymin) / resolution)) + 1
    xs, ys = np.meshgrid(xmin + resolution * np.arange(cols), ymin + resolution * np.arange(rows))
    blocked = _blocked_mask(world, xs, ys, inflation)
    distances = np.full((rows, cols), math.inf)

    goal_col = int(round((goal_xy[0] - xmin) / resolution))
    goal_row = int(round((goal_xy[1] - ymin) / resolution))
    queue: list[tuple[float, int, int]] = []
    for radius in range(0, 6):
        for row in range(goal_row - radius, goal_row + radius + 1):
            for col in range(goal_col - radius, goal_col + radius + 1):
                if 0 <= row < rows and 0 <= col < cols and not blocked[row, col]:
                    offset = math.hypot(xs[row, col] - goal_xy[0], ys[row, col] - goal_xy[1])
                    if offset < distances[row, col]:
                        distances[row, col] = offset
                        queue.append((offset, row, col))
        if queue:
            break
    heapq.heapify(queue)
    while queue:
        cost, row, col = heapq.heappop(queue)
        if cost > distances[row, col]:
            continue
        for d_col, d_row, step in _NEIGHBOURS:
            r, c = row + d_row, col + d_col
            if 0 <= r < rows and 0 <= c < cols and not blocked[r, c]:
                candidate = cost + step * resolution
                if candidate < distances[r, c]:
                    distances[r, c] = candidate
                    heapq.heappush(queue, (candidate, r, c))
    return DistanceField((xmin, ymin), resolution, distances)
