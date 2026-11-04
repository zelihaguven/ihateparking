"""The mission as the car sees it: its own map, pose estimate, bays and plan.

``StackView`` draws a ``MissionSnapshot`` from above, at a fixed centre and
scale (pixels per metre), so it lines up with a simulator's top-down view
drawn at the same centre and scale:

- the map: occupied black, seen free pale, unknown grey;
- bays found at the last decision yellow, the bay being parked in green,
  bays refused for want of a route red;
- the plan: forward legs blue, reverse legs orange;
- the car where it believes it is (filled blue) and where it truly is
  (thin green outline): the gap between them is the localization error;
- time, mode and the latest event.

Drawing needs OpenCV (``cv2``), loaded only here.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import Pose, vehicle_polygon
from .mission import MissionSnapshot
from .occupancy_map import FREE, OCCUPIED
from .slot_detector import SlotCandidate
from .world import VehicleSpec

UNKNOWN_RGB = (150, 150, 150)
FREE_RGB = (232, 238, 238)
OCCUPIED_RGB = (20, 20, 20)
CANDIDATE_RGB = (230, 180, 0)
TARGET_RGB = (40, 170, 80)
REJECTED_RGB = (220, 40, 40)
FORWARD_RGB = (40, 90, 220)
REVERSE_RGB = (240, 130, 20)
ESTIMATE_RGB = (60, 120, 240)
TRUTH_RGB = (0, 160, 60)
TEXT_RGB = (20, 20, 20)
REJECTED_MARK_M = 0.8  # half the size of the cross on a refused bay


class StackView:
    """Renders snapshots as RGB images of ``size`` pixels, ``scale`` px per metre, centred on ``centre``."""

    def __init__(self, vehicle: VehicleSpec, centre: tuple[float, float], scale: float = 10.0, size: tuple[int, int] = (800, 800)):
        self.vehicle = vehicle
        self.centre = centre
        self.scale = scale
        self.size = size
        self._cells = None  # pixel -> grid cell lookup, built for the first grid seen

    def to_pixel(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """World metres to (column, row) pixels; north is up."""
        width, height = self.size
        column = (np.asarray(x) - self.centre[0]) * self.scale + width / 2.0
        row = height / 2.0 - (np.asarray(y) - self.centre[1]) * self.scale
        return np.stack([column, row], axis=-1)

    def render(self, snapshot: MissionSnapshot) -> np.ndarray:
        import cv2

        image = self._map(cv2, snapshot)
        for candidate in snapshot.candidates:
            self._bay(cv2, image, candidate, CANDIDATE_RGB, 1)
        for centre in snapshot.rejected:
            self._cross(cv2, image, centre)
        if snapshot.target is not None:
            self._bay(cv2, image, snapshot.target, TARGET_RGB, 3)
        if snapshot.plan is not None and snapshot.plan.poses:
            self._plan(cv2, image, snapshot.plan.poses, snapshot.plan.gears)
        self._car(cv2, image, snapshot.true_pose, TRUTH_RGB, filled=False)
        self._car(cv2, image, snapshot.estimate, ESTIMATE_RGB, filled=True)
        self._caption(cv2, image, snapshot)
        return image

    def _map(self, cv2, snapshot: MissionSnapshot) -> np.ndarray:
        grid = snapshot.grid
        if self._cells is None:
            width, height = self.size
            columns, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
            x = self.centre[0] + (columns - width / 2.0) / self.scale
            y = self.centre[1] - (rows - height / 2.0) / self.scale
            grid_rows, grid_cols = grid.to_cell(x, y)
            inside = (grid_rows >= 0) & (grid_rows < grid.shape[0]) & (grid_cols >= 0) & (grid_cols < grid.shape[1])
            self._cells = (np.where(inside, grid_rows, 0), np.where(inside, grid_cols, 0), inside)
        rows, cols, inside = self._cells
        states = grid.states()[rows, cols]
        image = np.empty(states.shape + (3,), dtype=np.uint8)
        image[:] = UNKNOWN_RGB
        image[inside & (states == FREE)] = FREE_RGB
        # A 0.15 m cell is about a pixel wide: thicken walls and car edges to be seen.
        occupied = cv2.dilate((inside & (states == OCCUPIED)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        image[occupied] = OCCUPIED_RGB
        return image

    def _polygon(self, points: np.ndarray) -> np.ndarray:
        return np.round(self.to_pixel(points[:, 0], points[:, 1])).astype(np.int32).reshape(-1, 1, 2)

    def _bay(self, cv2, image: np.ndarray, candidate: SlotCandidate, colour: tuple[int, int, int], thickness: int) -> None:
        half_l, half_w = self.vehicle.length / 2.0, self.vehicle.width / 2.0
        c, s = math.cos(candidate.axis_yaw), math.sin(candidate.axis_yaw)
        corners = np.array([(candidate.center[0] + c * a - s * b, candidate.center[1] + s * a + c * b) for a, b in ((-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w))])
        cv2.polylines(image, [self._polygon(corners)], True, colour, thickness, cv2.LINE_AA)

    def _cross(self, cv2, image: np.ndarray, centre: tuple[float, float]) -> None:
        d = REJECTED_MARK_M
        for a, b in (((-d, -d), (d, d)), ((-d, d), (d, -d))):
            start = self.to_pixel(centre[0] + a[0], centre[1] + a[1])
            end = self.to_pixel(centre[0] + b[0], centre[1] + b[1])
            cv2.line(image, tuple(int(v) for v in np.round(start)), tuple(int(v) for v in np.round(end)), REJECTED_RGB, 2, cv2.LINE_AA)

    def _plan(self, cv2, image: np.ndarray, poses: list[Pose], gears: list[int]) -> None:
        points = self.to_pixel(np.array([p.x for p in poses]), np.array([p.y for p in poses]))
        for index in range(len(points) - 1):
            gear = gears[index] if index < len(gears) else 1
            colour = REVERSE_RGB if gear < 0 else FORWARD_RGB
            start, end = (tuple(int(v) for v in np.round(points[i])) for i in (index, index + 1))
            cv2.line(image, start, end, colour, 2, cv2.LINE_AA)

    def _car(self, cv2, image: np.ndarray, pose: Pose, colour: tuple[int, int, int], filled: bool) -> None:
        polygon = self._polygon(vehicle_polygon(pose, self.vehicle.length, self.vehicle.width, self.vehicle.rear_overhang))
        if filled:
            overlay = image.copy()
            cv2.fillPoly(overlay, [polygon], colour, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0.0, dst=image)
        cv2.polylines(image, [polygon], True, colour, 1 if not filled else 2, cv2.LINE_AA)

    def _caption(self, cv2, image: np.ndarray, snapshot: MissionSnapshot) -> None:
        lines = [f"{snapshot.time_s:6.1f} s   {snapshot.mode}"]
        lines += [f"{event.time_s:6.1f} s  {event.message}"[:90] for event in snapshot.events[-2:]]
        for index, line in enumerate(lines):
            cv2.putText(image, line, (10, 22 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.5 if index else 0.6, TEXT_RGB, 1, cv2.LINE_AA)
