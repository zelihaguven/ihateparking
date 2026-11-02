"""Correct the odometry estimate by matching each LiDAR scan to the map.

Odometry drifts; the map the car has built does not move. Each new scan is
taken relative to the car, so placing it at a slightly wrong pose smears it
against the walls and cars already mapped. The matcher tries small shifts
and turns around the odometry estimate and keeps the pose where the scan's
returns land closest to occupied cells (a likelihood field: a Gaussian of
the distance to the nearest occupied cell).

A correction is accepted only when the scan clearly fits the map: enough
returns, most of them close to mapped structure, a better fit than the
odometry pose and a best pose inside the search window (one on its edge
means the true pose may lie outside it). Otherwise odometry stands.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage

from .geometry import Pose, wrap_angle
from .lidar import LidarScan
from .occupancy_map import OCCUPIED, OccupancyGrid
from .world import VehicleSpec


@dataclass(frozen=True)
class ScanMatcherConfig:
    search_xy_m: float = 0.3
    step_xy_m: float = 0.05
    search_yaw_deg: float = 1.5
    step_yaw_deg: float = 0.25
    fine_step_xy_m: float = 0.01
    fine_step_yaw_deg: float = 0.05
    sigma_m: float = 0.1
    min_range_m: float = 0.5
    max_points: int = 360
    min_points: int = 60
    min_matched_fraction: float = 0.5  # returns within ``matched_m`` of a mapped cell
    matched_m: float = 0.2
    min_improvement: float = 1.01  # best score over the odometry pose's score


class ScanMatcher:
    def __init__(self, vehicle: VehicleSpec, config: ScanMatcherConfig = ScanMatcherConfig()):
        self.vehicle = vehicle
        self.config = config
        self.accepted = 0
        self.rejected = 0

    def correct(self, grid: OccupancyGrid, measured: LidarScan, true_yaw: float, estimate: Pose) -> Pose:
        """The estimate corrected against ``grid``, or ``estimate`` unchanged.

        ``true_yaw`` only recovers the beam angles relative to the car, which
        is how a LiDAR reports them.
        """
        cfg = self.config
        keep = measured.hits & (measured.ranges >= cfg.min_range_m)
        if keep.sum() < cfg.min_points:
            self.rejected += 1
            return estimate
        relative = (measured.angles - true_yaw)[keep]
        ranges = measured.ranges[keep]
        if len(ranges) > cfg.max_points:
            index = np.linspace(0, len(ranges) - 1, cfg.max_points).astype(int)
            relative, ranges = relative[index], ranges[index]
        distance = _distance_field(grid)
        if distance is None:
            self.rejected += 1
            return estimate

        def scores(xs: np.ndarray, ys: np.ndarray, yaw: float) -> np.ndarray:
            offset = self.vehicle.length / 2.0 - self.vehicle.rear_overhang
            px = offset * math.cos(yaw) + ranges * np.cos(yaw + relative)
            py = offset * math.sin(yaw) + ranges * np.sin(yaw + relative)
            rows, cols = grid.to_cell(xs[:, None] + px[None, :], ys[:, None] + py[None, :])
            inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
            d = np.full(rows.shape, np.inf)
            d[inside] = distance[rows[inside], cols[inside]]
            return np.exp(-0.5 * (d / cfg.sigma_m) ** 2).sum(axis=1), d

        base_score, base_d = scores(np.array([estimate.x]), np.array([estimate.y]), estimate.yaw)
        best = (float(base_score[0]), estimate, False)
        for pose, on_edge in self._search(estimate, cfg.search_xy_m, cfg.step_xy_m, cfg.search_yaw_deg, cfg.step_yaw_deg, scores):
            if pose[0] > best[0]:
                best = (pose[0], pose[1], on_edge)
        coarse = best[1]
        fine_xy = cfg.step_xy_m
        fine_yaw = cfg.step_yaw_deg
        for pose, _ in self._search(coarse, fine_xy, cfg.fine_step_xy_m, fine_yaw, cfg.fine_step_yaw_deg, scores):
            if pose[0] > best[0]:
                best = (pose[0], pose[1], best[2])
        score, corrected, on_edge = best
        _, d = scores(np.array([corrected.x]), np.array([corrected.y]), corrected.yaw)
        matched = float(np.mean(d[0] <= cfg.matched_m))
        if on_edge or matched < cfg.min_matched_fraction or score < cfg.min_improvement * float(base_score[0]):
            self.rejected += 1
            return estimate
        self.accepted += 1
        return corrected

    @staticmethod
    def _search(center: Pose, reach_xy: float, step_xy: float, reach_yaw_deg: float, step_yaw_deg: float, scores):
        """Best pose per yaw on a grid around ``center``: ((score, pose), on the window edge)."""
        offsets = np.arange(-reach_xy, reach_xy + 1e-9, step_xy)
        dx, dy = np.meshgrid(offsets, offsets)
        dx, dy = dx.ravel(), dy.ravel()
        edge_xy = (np.abs(dx) >= reach_xy - 1e-9) | (np.abs(dy) >= reach_xy - 1e-9)
        turns = np.arange(-reach_yaw_deg, reach_yaw_deg + 1e-9, step_yaw_deg)
        for turn in turns:
            yaw = wrap_angle(center.yaw + math.radians(turn))
            values, _ = scores(center.x + dx, center.y + dy, yaw)
            k = int(np.argmax(values))
            edge = bool(edge_xy[k]) or abs(turn) >= reach_yaw_deg - 1e-9
            yield (float(values[k]), Pose(float(center.x + dx[k]), float(center.y + dy[k]), yaw)), edge


def _distance_field(grid: OccupancyGrid) -> np.ndarray | None:
    occupied = grid.states() == OCCUPIED
    if occupied.sum() < 50:
        return None
    return ndimage.distance_transform_edt(~occupied) * grid.config.resolution
