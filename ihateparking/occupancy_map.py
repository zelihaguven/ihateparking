"""Log-odds occupancy grid built from LiDAR scans, and a planning world on it.

The grid is the vehicle's only knowledge of its surroundings. ``MapWorld``
exposes the same interface the Hybrid A* planner uses for the ground-truth
world (``is_collision``, ``bounds``, ``vehicle``), but answers from the map:
occupied cells and, by default, unknown cells count as obstacles, so the
planner never routes through space the sensor has not seen.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage

from .geometry import Pose
from .lidar import LidarScan
from .world import VehicleSpec

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


@dataclass(frozen=True)
class GridConfig:
    resolution: float = 0.15
    size_m: float = 100.0
    hit_log_odds: float = 0.85
    miss_log_odds: float = -0.4
    clamp: float = 4.0
    occupied_threshold: float = 0.6
    free_threshold: float = -0.4
    # A beam with no return marks free space only this far: a dropout may be
    # a beam that actually passed through a car.
    free_range_on_miss: float = 10.0
    # Beams either side searched for a return when a beam has none.
    dropout_window: int = 8


class OccupancyGrid:
    def __init__(self, center: tuple[float, float], config: GridConfig = GridConfig()):
        self.config = config
        cells = int(round(config.size_m / config.resolution))
        self.shape = (cells, cells)
        self.origin = (center[0] - config.size_m / 2.0, center[1] - config.size_m / 2.0)
        self.log_odds = np.zeros(self.shape, dtype=np.float32)
        self.observed = np.zeros(self.shape, dtype=bool)

    # -- coordinates --------------------------------------------------------
    def to_cell(self, x: np.ndarray | float, y: np.ndarray | float):
        res = self.config.resolution
        col = np.floor((np.asarray(x) - self.origin[0]) / res).astype(int)
        row = np.floor((np.asarray(y) - self.origin[1]) / res).astype(int)
        return row, col

    def cell_center(self, row: np.ndarray, col: np.ndarray):
        res = self.config.resolution
        return self.origin[0] + (col + 0.5) * res, self.origin[1] + (row + 0.5) * res

    def _inside(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        return (row >= 0) & (row < self.shape[0]) & (col >= 0) & (col < self.shape[1])

    # -- updates ------------------------------------------------------------
    def integrate(self, scan: LidarScan) -> None:
        cfg = self.config
        ox, oy = scan.origin
        free_length = np.where(scan.hits, scan.ranges - cfg.resolution, np.minimum(scan.ranges, cfg.free_range_on_miss))
        # A beam without a return between beams that did return is most
        # likely a dropout, not open space: trust it only as far as its
        # neighbours saw, or it would punch free cells through walls.
        missed = ~scan.hits
        if missed.any() and scan.hits.any():
            hit_ranges = np.where(scan.hits, scan.ranges, np.inf)
            neighbour = np.full_like(hit_ranges, np.inf)
            for shift in range(1, cfg.dropout_window + 1):
                neighbour = np.minimum(neighbour, np.minimum(np.roll(hit_ranges, shift), np.roll(hit_ranges, -shift)))
            bounded = missed & np.isfinite(neighbour)
            free_length = np.where(bounded, np.minimum(free_length, neighbour - 0.5), free_length)
        step = cfg.resolution * 0.7
        longest = float(free_length.max(initial=0.0))
        if longest > 0:
            samples = np.arange(0.0, longest, step)
            dist = samples[None, :]
            mask = dist < free_length[:, None]
            xs = ox + np.cos(scan.angles)[:, None] * dist
            ys = oy + np.sin(scan.angles)[:, None] * dist
            rows, cols = self.to_cell(xs[mask], ys[mask])
            keep = self._inside(rows, cols)
            free_flat = np.unique(rows[keep] * self.shape[1] + cols[keep])
        else:
            free_flat = np.empty(0, dtype=int)
        hit_x = ox + np.cos(scan.angles[scan.hits]) * scan.ranges[scan.hits]
        hit_y = oy + np.sin(scan.angles[scan.hits]) * scan.ranges[scan.hits]
        rows, cols = self.to_cell(hit_x, hit_y)
        keep = self._inside(rows, cols)
        hit_flat = np.unique(rows[keep] * self.shape[1] + cols[keep])
        free_flat = np.setdiff1d(free_flat, hit_flat, assume_unique=True)
        flat = self.log_odds.reshape(-1)
        seen = self.observed.reshape(-1)
        flat[free_flat] = np.clip(flat[free_flat] + cfg.miss_log_odds, -cfg.clamp, cfg.clamp)
        flat[hit_flat] = np.clip(flat[hit_flat] + cfg.hit_log_odds, -cfg.clamp, cfg.clamp)
        seen[free_flat] = True
        seen[hit_flat] = True

    def states(self) -> np.ndarray:
        cfg = self.config
        out = np.full(self.shape, UNKNOWN, dtype=np.uint8)
        out[self.observed & (self.log_odds <= cfg.free_threshold)] = FREE
        out[self.log_odds >= cfg.occupied_threshold] = OCCUPIED
        return out

    def known_bounds(self, margin: float = 1.0) -> tuple[float, float, float, float]:
        rows, cols = np.nonzero(self.observed)
        if rows.size == 0:
            return (self.origin[0], self.origin[0], self.origin[1], self.origin[1])
        res = self.config.resolution
        return (
            self.origin[0] + cols.min() * res - margin,
            self.origin[0] + (cols.max() + 1) * res + margin,
            self.origin[1] + rows.min() * res - margin,
            self.origin[1] + (rows.max() + 1) * res + margin,
        )


@dataclass
class _Slot:
    """Placeholder so planner defaults (``world.slot``) exist; goals are explicit."""

    center: tuple[float, float] = (0.0, 0.0)


class MapWorld:
    """A frozen snapshot of the grid, shaped like ``ParkingWorld`` for the planner."""

    def __init__(self, grid: OccupancyGrid, vehicle: VehicleSpec, start: Pose, collision_margin: float = 0.15, unknown_is_obstacle: bool = True):
        self.grid = grid
        self.vehicle = vehicle
        self.start_pose = start
        self.collision_margin = collision_margin
        self.obstacles: list = []
        self.slot = _Slot()
        states = grid.states()
        self.states = states
        blocked = states == OCCUPIED
        if unknown_is_obstacle:
            blocked |= states == UNKNOWN
        self.blocked = blocked
        res = grid.config.resolution
        # Distance (m) from every cell centre to the nearest blocked cell.
        self.distance = ndimage.distance_transform_edt(~blocked) * res
        boundary = blocked & ndimage.binary_dilation(~blocked)
        rows, cols = np.nonzero(boundary)
        xs, ys = grid.cell_center(rows, cols)
        self._points = np.column_stack([xs, ys])
        self._bucket = 3.0
        keys = np.floor(self._points / self._bucket).astype(int)
        self._buckets: dict[tuple[int, int], np.ndarray] = {}
        if len(keys):
            order = np.lexsort((keys[:, 1], keys[:, 0]))
            keys, points = keys[order], self._points[order]
            splits = np.nonzero(np.any(np.diff(keys, axis=0) != 0, axis=1))[0] + 1
            for chunk_keys, chunk in zip(np.split(keys, splits), np.split(points, splits)):
                self._buckets[(int(chunk_keys[0, 0]), int(chunk_keys[0, 1]))] = chunk
        self.bounds = grid.known_bounds()
        self._half_length = vehicle.length / 2.0
        self._half_width = vehicle.width / 2.0
        self._radius = math.hypot(self._half_length, self._half_width)

    def distance_at(self, x: float, y: float) -> float:
        row, col = self.grid.to_cell(x, y)
        if not (0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]):
            return 0.0
        return float(self.distance[row, col])

    def is_collision(self, pose: Pose, extra_margin: float = 0.0) -> bool:
        margin = self.collision_margin + extra_margin
        offset = self._half_length - self.vehicle.rear_overhang
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        cx, cy = pose.x + offset * c, pose.y + offset * s
        clearance = self.distance_at(cx, cy)
        res = self.grid.config.resolution
        if clearance <= 0.0:
            return True
        if clearance > self._radius + margin + res:
            return False
        reach = self._radius + margin + res
        bx0, bx1 = int(math.floor((cx - reach) / self._bucket)), int(math.floor((cx + reach) / self._bucket))
        by0, by1 = int(math.floor((cy - reach) / self._bucket)), int(math.floor((cy + reach) / self._bucket))
        chunks = [self._buckets[(bx, by)] for bx in range(bx0, bx1 + 1) for by in range(by0, by1 + 1) if (bx, by) in self._buckets]
        if not chunks:
            return False
        points = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        dx, dy = points[:, 0] - cx, points[:, 1] - cy
        along = np.abs(c * dx + s * dy)
        across = np.abs(-s * dx + c * dy)
        # A blocked cell is a square; inflate by half a cell so cells that
        # graze the footprint count.
        pad = margin + res / 2.0
        return bool(np.any((along <= self._half_length + pad) & (across <= self._half_width + pad)))

    def blocked_mask(self, xs: np.ndarray, ys: np.ndarray, inflation: float) -> np.ndarray:
        rows, cols = self.grid.to_cell(xs, ys)
        inside = (rows >= 0) & (rows < self.grid.shape[0]) & (cols >= 0) & (cols < self.grid.shape[1])
        out = np.ones(xs.shape, dtype=bool)
        out[inside] = self.distance[rows[inside], cols[inside]] < inflation
        return out
