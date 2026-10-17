"""Find parking spaces in an occupancy grid without any prior map.

The detector sees only the LiDAR occupancy grid. Rather than segmenting
cars (fragile with partial views), it searches the map directly for
*bay-shaped* free space: a place the car fits that is enclosed the way a
parking bay is.

* Side-by-side bay (perpendicular or angled): the footprint plus a side
  clearance is observed free; something occupied lies just **behind** it
  (kerb, wall or the car parked opposite); at least one **flank** has a
  neighbour (car side or wall); and the **front** opens onto free space.
* End-to-end bay (parallel): one long side runs along a wall or kerb, the
  other long side opens onto free space, and a car or wall closes at least
  one **end**.

These rules reject the drive-through gap at a row end (nothing behind it)
and open aisle space (no flank, no back). Bay axes come from the dominant
edge directions in the map, so rotated lots and angled rows need no prior.

For each axis the map is resampled on a lattice aligned with it, and
summed-area tables give every box count in constant time, so all positions
are tested at once.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage

from .geometry import Pose, wrap_angle
from .occupancy_map import FREE, OCCUPIED, UNKNOWN, OccupancyGrid
from .world import VehicleSpec

SIDE_CLEARANCE_M = 0.25
MAX_UNKNOWN_FRACTION = 0.06
TENTATIVE_UNKNOWN_FRACTION = 0.35
BACK_SEARCH_M = 1.6
FLANK_SEARCH_M = 1.0
FRONT_OPEN_M = 2.5
KERB_SEARCH_M = 0.9
END_SEARCH_M = 2.5
OPEN_FRACTION = 0.75
ANCHOR_STRIDE = 2  # test every second lattice cell
NMS_RADIUS_M = 2.0
MAX_AXES = 4
PARALLEL_EDGE = 99  # pseudo-state: occupied cell whose edge runs along the lattice axis
PARALLEL_TOLERANCE = math.radians(10.0)


@dataclass(frozen=True)
class SlotCandidate:
    center: tuple[float, float]  # footprint centre
    axis_yaw: float  # side-by-side: heading into the bay (nose-in); parallel: along the kerb
    row: str  # "side_by_side" or "end_to_end"
    score: float

    def headings(self) -> list[float]:
        return [wrap_angle(self.axis_yaw), wrap_angle(self.axis_yaw + math.pi)]

    def goal(self, heading: float, vehicle: VehicleSpec) -> Pose:
        offset = vehicle.length / 2.0 - vehicle.rear_overhang
        return Pose(self.center[0] - offset * math.cos(heading), self.center[1] - offset * math.sin(heading), heading)


def edge_directions(states: np.ndarray) -> np.ndarray:
    """Local edge direction (mod pi) of the occupied structure, per cell.

    The structure tensor of the smoothed occupancy averages gradient
    orientations without their sign, so both faces of a thin line agree.
    """
    smooth = ndimage.gaussian_filter((states == OCCUPIED).astype(float), 1.0)
    gx = ndimage.sobel(smooth, axis=1)
    gy = ndimage.sobel(smooth, axis=0)
    jxx = ndimage.gaussian_filter(gx * gx, 2.0)
    jyy = ndimage.gaussian_filter(gy * gy, 2.0)
    jxy = ndimage.gaussian_filter(gx * gy, 2.0)
    gradient = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    return (gradient + math.pi / 2) % math.pi


def dominant_axes(states: np.ndarray, max_axes: int = MAX_AXES) -> list[float]:
    """Edge directions (mod pi) with the most occupied-edge support."""
    occupied = (states == OCCUPIED).astype(float)
    if occupied.sum() < 20:
        return []
    smooth = ndimage.gaussian_filter(occupied, 1.0)
    gy = ndimage.sobel(smooth, axis=0)
    gx = ndimage.sobel(smooth, axis=1)
    magnitude = np.hypot(gx, gy)
    mask = magnitude > magnitude.max() * 0.1
    # Gradient is normal to the edge; the edge runs at +90 degrees.
    angles = (np.arctan2(gy[mask], gx[mask]) + math.pi / 2) % math.pi
    histogram, edges = np.histogram(angles, bins=90, range=(0.0, math.pi), weights=magnitude[mask])
    histogram = ndimage.gaussian_filter1d(histogram, 1.0, mode="wrap")
    peaks = []
    for _ in range(max_axes):
        index = int(np.argmax(histogram))
        if histogram[index] <= 0 or (peaks and histogram[index] < 0.15 * peaks[0][1]):
            break
        peaks.append(((edges[index] + edges[index + 1]) / 2, histogram[index]))
        for offset in range(-4, 5):  # suppress +-8 degrees
            histogram[(index + offset) % len(histogram)] = 0.0
    return [angle for angle, _ in peaks]


class _AlignedMap:
    """The grid resampled on a lattice whose x axis is ``axis``."""

    def __init__(self, grid: OccupancyGrid, states: np.ndarray, edge_dir: np.ndarray, axis: float, bounds: tuple[float, float, float, float]):
        self.res = grid.config.resolution
        self.axis = axis
        self.u = np.array([math.cos(axis), math.sin(axis)])
        self.v = np.array([-math.sin(axis), math.cos(axis)])
        xmin, xmax, ymin, ymax = bounds
        corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        pu, pv = corners @ self.u, corners @ self.v
        self.u0, self.v0 = pu.min(), pv.min()
        self.nu = int((pu.max() - pu.min()) / self.res) + 1
        self.nv = int((pv.max() - pv.min()) / self.res) + 1
        iu, iv = np.meshgrid(np.arange(self.nu), np.arange(self.nv))
        world = (self.u0 + iu[..., None] * self.res) * self.u + (self.v0 + iv[..., None] * self.res) * self.v
        rows, cols = grid.to_cell(world[..., 0], world[..., 1])
        inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
        image = np.full((self.nv, self.nu), UNKNOWN, dtype=np.uint8)
        image[inside] = states[rows[inside], cols[inside]]
        direction = np.zeros((self.nv, self.nu))
        direction[inside] = edge_dir[rows[inside], cols[inside]]
        offset = np.abs(((direction - axis) + math.pi / 2) % math.pi - math.pi / 2)
        self.tables = {state: self._table(image == state) for state in (FREE, OCCUPIED, UNKNOWN)}
        # Occupied cells whose local edge runs along the u axis.
        self.tables[PARALLEL_EDGE] = self._table((image == OCCUPIED) & (offset <= PARALLEL_TOLERANCE))

    @staticmethod
    def _table(mask: np.ndarray) -> np.ndarray:
        table = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int32)
        table[1:, 1:] = np.cumsum(np.cumsum(mask, axis=0), axis=1)
        return table

    def box(self, state: int, iv: np.ndarray, iu: np.ndarray, du0: float, du1: float, dv0: float, dv1: float) -> np.ndarray:
        """Cell count of ``state`` in [du0,du1]x[dv0,dv1] (m) around each anchor."""
        table = self.tables[state]
        a0 = np.clip(iu + int(round(du0 / self.res)), 0, self.nu)
        a1 = np.clip(iu + int(round(du1 / self.res)), 0, self.nu)
        b0 = np.clip(iv + int(round(dv0 / self.res)), 0, self.nv)
        b1 = np.clip(iv + int(round(dv1 / self.res)), 0, self.nv)
        return table[b1, a1] - table[b0, a1] - table[b1, a0] + table[b0, a0]

    def area(self, du0: float, du1: float, dv0: float, dv1: float) -> int:
        return max(1, int(round((du1 - du0) / self.res)) * int(round((dv1 - dv0) / self.res)))

    def world(self, iv: np.ndarray, iu: np.ndarray) -> np.ndarray:
        return (self.u0 + iu[:, None] * self.res) * self.u + (self.v0 + iv[:, None] * self.res) * self.v


COVERAGE_BIN_M = 0.5
FLANK_COVERAGE = 0.6
BACK_COVERAGE = 0.75
KERB_COVERAGE = 0.85
END_COVERAGE = 0.4
WALL_RUN_M = 3.0
NEAR_BACK_M = 0.6
NEAR_BACK_COVERAGE = 0.5
WALL_CONTINUATION = 0.3


def _coverage(m: _AlignedMap, iv, iu, along_u: bool, a0: float, a1: float, b0: float, b1: float, state: int = OCCUPIED) -> np.ndarray:
    """Fraction of COVERAGE_BIN_M bins along a strip that contain an occupied cell.

    ``along_u`` strips run along the lattice u axis over [a0, a1], with
    [b0, b1] across them; otherwise the roles swap. Counting bins rather than
    cells demands an edge that spans the side, not one corner poking in.
    """
    edges = np.arange(a0, a1, COVERAGE_BIN_M)
    hits = np.zeros(iv.shape, dtype=float)
    for lo in edges:
        hi = min(lo + COVERAGE_BIN_M, a1)
        box = m.box(state, iv, iu, lo, hi, b0, b1) if along_u else m.box(state, iv, iu, b0, b1, lo, hi)
        hits += box > 0
    return hits / max(1, len(edges))


def _side_by_side(m: _AlignedMap, iv, iu, half_l: float, half_w: float, near_back_needed: float = NEAR_BACK_COVERAGE) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Anchors passing the side-by-side rules, for a back at +u and at -u."""
    inner_l = half_l - SIDE_CLEARANCE_M
    inner_w = half_w - SIDE_CLEARANCE_M
    flank_strips = ((half_w, half_w + FLANK_SEARCH_M), (-half_w - FLANK_SEARCH_M, -half_w))
    # A neighbour's side runs along the bay axis over the whole bay depth:
    # demand it in each half, so a short face (a car's front) cannot pass.
    flank_cover = [
        np.minimum(
            _coverage(m, iv, iu, True, -inner_l, 0.0, *strip, PARALLEL_EDGE),
            _coverage(m, iv, iu, True, 0.0, inner_l, *strip, PARALLEL_EDGE),
        )
        for strip in flank_strips
    ]
    results = []
    for back in (1.0, -1.0):
        back_box = (half_l, half_l + BACK_SEARCH_M) if back > 0 else (-half_l - BACK_SEARCH_M, -half_l)
        front_box = (-half_l - FRONT_OPEN_M, -half_l) if back > 0 else (half_l, half_l + FRONT_OPEN_M)
        # Beyond the bay's front, a car's side has ended; a wall carries on.
        beyond = (-inner_l - WALL_RUN_M, -inner_l - 0.5) if back > 0 else (inner_l + 0.5, inner_l + WALL_RUN_M)
        # Both flanks must be finite neighbours (car sides, not a wall that
        # runs on into the aisle). A gap between two parked cars is the least
        # ambiguous bay without line markings; accepting a single flank was
        # measured to halve precision (aisle and row-end false positives).
        flank_ok = np.ones(iv.shape, dtype=bool)
        flank_score = np.zeros(iv.shape)
        for strip, cover in zip(flank_strips, flank_cover):
            continues = _coverage(m, iv, iu, True, *beyond, *strip, PARALLEL_EDGE)
            flank_ok &= (cover >= FLANK_COVERAGE) & (continues < WALL_CONTINUATION)
            flank_score += cover
        back_cover = _coverage(m, iv, iu, False, -inner_w, inner_w, *back_box)
        front_free = m.box(FREE, iv, iu, *front_box, -inner_w, inner_w) / m.area(*front_box, -inner_w, inner_w)
        # The back must be seen close behind the footprint, so the car pulls
        # fully into the bay; a bay whose far end is not yet observed waits
        # until the car has seen it.
        near_box = (half_l, half_l + NEAR_BACK_M) if back > 0 else (-half_l - NEAR_BACK_M, -half_l)
        near_back = _coverage(m, iv, iu, False, -inner_w, inner_w, *near_box)
        ok = (back_cover >= BACK_COVERAGE) & (near_back >= near_back_needed) & (front_free >= OPEN_FRACTION) & flank_ok
        results.append((ok, back, back_cover + flank_score + 2.0 * near_back))
    return results


def _end_to_end(m: _AlignedMap, iv, iu, half_l: float, half_w: float) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Anchors passing the parallel-bay rules, for a kerb at +v and at -v."""
    inner_l = half_l - SIDE_CLEARANCE_M
    inner_w = half_w - SIDE_CLEARANCE_M
    results = []
    for kerb in (1.0, -1.0):
        kerb_box = (half_w, half_w + KERB_SEARCH_M) if kerb > 0 else (-half_w - KERB_SEARCH_M, -half_w)
        open_box = (-half_w - FRONT_OPEN_M, -half_w) if kerb > 0 else (half_w, half_w + FRONT_OPEN_M)
        # The car closing an end shows its aisle-side flank, parallel to the
        # bay, beyond that end; a perpendicular wall at a corner does not.
        lateral = (-half_w - 0.4, inner_w - 0.3) if kerb > 0 else (-inner_w + 0.3, half_w + 0.4)
        end_a = _coverage(m, iv, iu, True, half_l, half_l + END_SEARCH_M, *lateral, PARALLEL_EDGE)
        end_b = _coverage(m, iv, iu, True, -half_l - END_SEARCH_M, -half_l, *lateral, PARALLEL_EDGE)
        end_ok = (end_a >= END_COVERAGE) & (end_b >= END_COVERAGE)
        kerb_cover = _coverage(m, iv, iu, True, -inner_l, inner_l, *kerb_box, PARALLEL_EDGE)
        open_free = m.box(FREE, iv, iu, -inner_l, inner_l, *open_box) / m.area(-inner_l, inner_l, *open_box)
        ok = (kerb_cover >= KERB_COVERAGE) & (open_free >= OPEN_FRACTION) & end_ok
        # Park close behind or ahead of a car, as a driver would, rather than
        # anywhere along a long gap: reward an end car within 0.8 m.
        near_a = _coverage(m, iv, iu, False, -inner_w, inner_w, half_l, half_l + 0.8)
        near_b = _coverage(m, iv, iu, False, -inner_w, inner_w, -half_l - 0.8, -half_l)
        results.append((ok, kerb, kerb_cover + 2.0 * np.maximum(near_a, near_b)))
    return results


def detect_slots(grid: OccupancyGrid, vehicle: VehicleSpec, tentative: bool = False) -> list[SlotCandidate]:
    """Verified bays, or with ``tentative`` the promising gaps worth a closer look.

    Tentative mode tolerates unobserved cells inside the footprint and an
    unseen back edge. It only chooses where to look next; a park decision
    always uses the strict rules.
    """
    max_unknown = TENTATIVE_UNKNOWN_FRACTION if tentative else MAX_UNKNOWN_FRACTION
    near_back_needed = 0.0 if tentative else NEAR_BACK_COVERAGE
    states = grid.states()
    rows, cols = np.nonzero(states != UNKNOWN)
    if rows.size == 0:
        return []
    xs, ys = grid.cell_center(rows, cols)
    bounds = (xs.min() - 1, xs.max() + 1, ys.min() - 1, ys.max() + 1)
    half_l = vehicle.length / 2 + SIDE_CLEARANCE_M
    half_w = vehicle.width / 2 + SIDE_CLEARANCE_M
    raw: list[tuple[float, SlotCandidate]] = []
    axes = dominant_axes(states)
    edge_dir = edge_directions(states)
    tried: list[float] = []
    for edge in axes:
        for axis in (edge, edge + math.pi / 2):
            if any(abs(wrap_angle(2 * (axis - other))) < math.radians(6) for other in tried):
                continue
            tried.append(axis)
            m = _AlignedMap(grid, states, edge_dir, axis, bounds)
            iv, iu = np.mgrid[0:m.nv:ANCHOR_STRIDE, 0:m.nu:ANCHOR_STRIDE]
            iv, iu = iv.ravel(), iu.ravel()
            area = m.area(-half_l, half_l, -half_w, half_w)
            occupied = m.box(OCCUPIED, iv, iu, -half_l, half_l, -half_w, half_w)
            unknown = m.box(UNKNOWN, iv, iu, -half_l, half_l, -half_w, half_w)
            clear = (occupied == 0) & (unknown <= max_unknown * area)
            if not clear.any():
                continue
            # Side-by-side: ego length along the lattice u axis.
            for ok, back, snug in _side_by_side(m, iv, iu, half_l, half_w, near_back_needed):
                hits = np.nonzero(clear & ok)[0]
                heading = axis if back > 0 else axis + math.pi
                for point, score in zip(m.world(iv[hits], iu[hits]), snug[hits]):
                    raw.append((float(score), SlotCandidate((float(point[0]), float(point[1])), wrap_angle(heading), "side_by_side", float(score))))
            # Parallel: ego length also along u, kerb on a long side.
            for ok, _kerb, snug in _end_to_end(m, iv, iu, half_l, half_w):
                hits = np.nonzero(clear & ok)[0]
                for point, score in zip(m.world(iv[hits], iu[hits]), snug[hits]):
                    raw.append((float(score), SlotCandidate((float(point[0]), float(point[1])), wrap_angle(axis), "end_to_end", float(score))))
    refined = [_refine(grid, states, edge_dir, vehicle, candidate, max_unknown) for _, candidate in raw]
    return _non_maximum_suppression([(score, item) for (score, _), item in zip(raw, refined) if item is not None])


def _refine(grid: OccupancyGrid, states: np.ndarray, edge_dir: np.ndarray, vehicle: VehicleSpec, candidate: SlotCandidate, max_unknown: float = MAX_UNKNOWN_FRACTION) -> SlotCandidate | None:
    """Snap the axis to the neighbours' edge direction and re-check clearance.

    The lattice axes come from global histogram peaks; a bay's own
    neighbours give its true axis to within a few degrees.
    """
    if candidate.row != "side_by_side":
        return candidate
    yaw = candidate.axis_yaw
    u = np.array([math.cos(yaw), math.sin(yaw)])
    v = np.array([-math.sin(yaw), math.cos(yaw)])
    center = np.array(candidate.center)
    half_l = vehicle.length / 2
    half_w = vehicle.width / 2 + SIDE_CLEARANCE_M
    along, across = np.meshgrid(np.arange(-half_l, half_l, 0.15), np.arange(half_w, half_w + FLANK_SEARCH_M, 0.15))
    directions = []
    for side in (1.0, -1.0):
        points = center + along.ravel()[:, None] * u + side * across.ravel()[:, None] * v
        rows, cols = grid.to_cell(points[:, 0], points[:, 1])
        inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
        rows, cols = rows[inside], cols[inside]
        occupied = states[rows, cols] == OCCUPIED
        directions.append(edge_dir[rows[occupied], cols[occupied]])
    angles = np.concatenate(directions)
    if angles.size < 8:
        return None
    # Circular mean of axial (mod pi) directions via angle doubling.
    mean = 0.5 * math.atan2(np.mean(np.sin(2 * angles)), np.mean(np.cos(2 * angles)))
    refined = mean if abs(wrap_angle(mean - yaw)) <= math.pi / 2 else mean + math.pi
    if abs(wrap_angle(refined - yaw)) > math.radians(25.0):
        return None
    samples_u, samples_v = np.meshgrid(np.arange(-half_l - SIDE_CLEARANCE_M, half_l + SIDE_CLEARANCE_M + 1e-9, 0.15), np.arange(-half_w, half_w + 1e-9, 0.15))
    c, s = math.cos(refined), math.sin(refined)
    xs = center[0] + c * samples_u - s * samples_v
    ys = center[1] + s * samples_u + c * samples_v
    rows, cols = grid.to_cell(xs.ravel(), ys.ravel())
    inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    if not inside.all():
        return None
    cells = states[rows, cols]
    if np.any(cells == OCCUPIED) or np.mean(cells == UNKNOWN) > max_unknown:
        return None
    return SlotCandidate(candidate.center, wrap_angle(refined), candidate.row, candidate.score)


def _non_maximum_suppression(raw: list[tuple[float, SlotCandidate]]) -> list[SlotCandidate]:
    kept: list[SlotCandidate] = []
    for _, candidate in sorted(raw, key=lambda item: -item[0]):
        if all(math.dist(candidate.center, other.center) > NMS_RADIUS_M for other in kept):
            kept.append(candidate)
    return kept
