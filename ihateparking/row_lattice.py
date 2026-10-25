"""Place a bay in a run of free bays from the parked cars along its row.

A gap between two parked cars is unambiguous; a run of several free bays is
not, because without line markings the bay boundaries are invisible. The
row still gives them away: bays repeat at a fixed pitch, so the cars that
close the run on either side, divided by the pitch measured between two
adjacent parked cars, fix how many bays the run holds and where each one is.

A candidate is accepted only when parked cars are seen on *both* sides
along the row, so the open end of a row is never taken for a bay.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import wrap_angle
from .occupancy_map import OCCUPIED, OccupancyGrid

ROW_SEARCH_M = 16.0
STEP_M = 0.1
CAR_RUN_RANGE_M = (0.3, 4.0)  # extent along the row of one parked car's visible outline
BAY_WIDTH_RANGE_M = (2.45, 2.95)
MIN_LATTICE_CARS = 4
LATTICE_COHERENCE = 0.75
LATTICE_MAX_RMS_M = 0.4
LATTICE_MARGIN_M = 0.1  # the bay may overlap a closing car's outline by this much
WHOLE_CAR_WIDTH_M = (1.6, 2.2)
SIDE_ALIGNMENT = math.radians(15.0)
MAX_SNAP_M = 1.0
ROW_DIRECTION_TOLERANCE = math.radians(50.0)
CLOSING_SIDE_M = 2.5  # a closing car's side seen along the bay depth
MAX_CAR_SIDE_M = 6.0
WALL_PROBE_EXTRA_M = 3.5


def row_direction(grid: OccupancyGrid, states: np.ndarray, edge_dir: np.ndarray, center: np.ndarray, u: np.ndarray, back_from: float, back_to: float, half_w: float) -> np.ndarray:
    """Unit vector along the row, from the edge behind the bay.

    The back edge (kerb, wall or the opposite row's fronts) runs along the
    row. In a perpendicular row that is square to the bay axis; in an angled
    row it is not. Falls back to square when the back gives no clear answer.
    """
    v = np.array([-u[1], u[0]])
    along, across = np.meshgrid(np.arange(back_from, back_to, 0.15), np.arange(-half_w * 2.0, half_w * 2.0, 0.15))
    points = center + along.ravel()[:, None] * u + across.ravel()[:, None] * v
    rows, cols = grid.to_cell(points[:, 0], points[:, 1])
    inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    rows, cols = rows[inside], cols[inside]
    angles = edge_dir[rows, cols][states[rows, cols] == OCCUPIED]
    if angles.size < 8 or np.hypot(np.mean(np.cos(2 * angles)), np.mean(np.sin(2 * angles))) < 0.6:
        return v
    mean = 0.5 * math.atan2(np.mean(np.sin(2 * angles)), np.mean(np.cos(2 * angles)))
    direction = np.array([math.cos(mean), math.sin(mean)])
    if direction @ v < 0:
        direction = -direction
    # The row cannot run along the bay axis.
    if abs(wrap_angle(2 * (mean - math.atan2(v[1], v[0])))) / 2 > ROW_DIRECTION_TOLERANCE:
        return v
    return direction


def car_centres(grid: OccupancyGrid, states: np.ndarray, edge_dir: np.ndarray, center: np.ndarray, u: np.ndarray, r: np.ndarray, depth: tuple[float, float], aisle: tuple[float, float]) -> list[tuple[float, float, bool]] | None:
    """Parked car outlines along ``r`` as (first, last, could close a run), or None if a wall is in the way.

    A probe parallel to the bay axis, spanning the bay depth, slides along
    the row; each run of positions where it touches an occupied cell is one
    car's outline (its front face and sides). The outline can close a run of
    free bays only if one unbroken side, running along the bay axis, shows
    along most of the depth (a nose poking in, a car end seen across an
    aisle or a car in another orientation does not) and nothing at that
    position carries on into the aisle beyond ``aisle`` (a wall does).
    """
    ts = np.arange(-ROW_SEARCH_M, ROW_SEARCH_M + 1e-9, STEP_M)
    us = np.arange(depth[0], depth[1], 0.15)
    occupied = _probe(grid, states, center, u, r, ts, us)
    along_axis = _probe(grid, states, center, u, r, ts, us, edge_dir, math.atan2(u[1], u[0]))
    walled = _probe(grid, states, center, u, r, ts, np.arange(aisle[0], aisle[1], 0.15)).any(axis=1)
    # A car side ends within a car length; a wall along the bay axis runs on
    # past the back of the bay.
    long_probe = _probe(grid, states, center, u, r, ts, np.arange(depth[0], depth[1] + WALL_PROBE_EXTRA_M, 0.15))
    walled |= _longest_run(long_probe) * 0.15 > MAX_CAR_SIDE_M
    hit = occupied.any(axis=1)
    side_m = _longest_run(along_axis) * 0.15
    cars = []
    start = None
    for index, value in enumerate(np.append(hit, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            length = (index - start) * STEP_M
            if length > CAR_RUN_RANGE_M[1]:
                return None
            if length >= CAR_RUN_RANGE_M[0]:
                closes = side_m[start:index].max() >= CLOSING_SIDE_M and not walled[start:index].any()
                cars.append((float(ts[start]), float(ts[index - 1]), bool(closes)))
            start = None
    return cars


def _longest_run(occupied: np.ndarray) -> np.ndarray:
    """Longest unbroken stretch of occupied samples along each probe (one-sample gaps bridged)."""
    filled = occupied.copy()
    filled[:, 1:-1] |= occupied[:, :-2] & occupied[:, 2:]
    best = np.zeros(len(filled), dtype=int)
    current = np.zeros(len(filled), dtype=int)
    for column in filled.T:
        current = np.where(column, current + 1, 0)
        best = np.maximum(best, current)
    return best


def _probe(grid: OccupancyGrid, states: np.ndarray, center: np.ndarray, u: np.ndarray, r: np.ndarray, ts: np.ndarray, us: np.ndarray, edge_dir: np.ndarray | None = None, axis: float = 0.0) -> np.ndarray:
    """Occupied samples of probes parallel to ``u`` at offsets ``ts`` along ``r``.

    With ``edge_dir``, only cells whose edge runs along ``axis`` count.
    """
    points = center + ts[:, None, None] * r + us[None, :, None] * u
    rows, cols = grid.to_cell(points[..., 0], points[..., 1])
    inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    occupied = np.zeros(rows.shape[:2], dtype=bool)
    hit = states[rows[inside], cols[inside]] == OCCUPIED
    if edge_dir is not None:
        offset = np.abs(((edge_dir[rows[inside], cols[inside]] - axis) + math.pi / 2) % math.pi - math.pi / 2)
        hit &= offset <= SIDE_ALIGNMENT
    occupied[inside] = hit
    return occupied


def snap_to_row(cars: list[tuple[float, float, bool]], across: float, car_width: float) -> float | None:
    """Offset along the row from the candidate to the bay it falls in, or None.

    ``across`` is the cosine between the row and the normal of the bay axis
    (1 in a perpendicular row): bay widths and car widths are measured along
    that normal. Parked cars sit up to half a metre off their bay centre, so
    two cars alone cannot fix a long run; the bay lattice (width and phase)
    is fitted to every parked car seen across its full width, which averages
    those offsets out. The candidate snaps to the lattice bay it falls in,
    which must lie clear of the cars that close the run.
    """
    behind = [car for car in cars if car[1] < 0]
    ahead = [car for car in cars if car[0] > 0]
    if not behind or not ahead or across < 0.5:
        return None  # an open end of the row, not a run between parked cars
    last_behind, first_ahead = max(behind), min(ahead)
    if not (last_behind[2] and first_ahead[2]):
        return None
    whole = [(first + last) / 2.0 * across for first, last, _ in cars if WHOLE_CAR_WIDTH_M[0] <= (last - first) * across <= WHOLE_CAR_WIDTH_M[1]]
    lattice = fit_lattice(whole)
    if lattice is None:
        return None
    width, phase = lattice
    bay = phase + round(-phase / width) * width
    # The bay must lie inside the row the lattice was fitted to, with whole
    # parked cars on both sides: past the last car of a row, the "car" that
    # seems to close the run can be a wall at a slight angle.
    if not min(whole) < bay < max(whole):
        return None
    margin = car_width / 2.0 - LATTICE_MARGIN_M
    if bay - margin < last_behind[1] * across or bay + margin > first_ahead[0] * across:
        return None
    offset = bay / across
    return offset if abs(offset) <= MAX_SNAP_M else None


def fit_lattice(centres: list[float]) -> tuple[float, float] | None:
    """(bay width, phase) of the regular lattice the car centres sit on, or None.

    Each width is scored by how tightly the centres agree in phase (the
    length of their mean unit phasor); the best one must be clearly regular.
    """
    if len(centres) < MIN_LATTICE_CARS:
        return None
    x = np.asarray(centres)
    widths = np.arange(BAY_WIDTH_RANGE_M[0], BAY_WIDTH_RANGE_M[1] + 1e-9, 0.005)
    phasors = np.exp(2j * np.pi * x[None, :] / widths[:, None]).mean(axis=1)
    best = int(np.argmax(np.abs(phasors)))
    if abs(phasors[best]) < LATTICE_COHERENCE:
        return None
    width = float(widths[best])
    phase = float(np.angle(phasors[best]) / (2 * np.pi) * width)
    residual = (x - phase + width / 2) % width - width / 2
    if float(np.sqrt(np.mean(residual ** 2))) > LATTICE_MAX_RMS_M:
        return None
    return width, phase
