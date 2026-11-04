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
LATTICE_EDGE_M = 0.03  # a best width this close to the range edge is an alias
LATTICE_MARGIN_M = 0.1  # the bay may overlap a closing car's outline by this much
WHOLE_CAR_WIDTH_M = (1.6, 2.4)  # a car's width plus a few degrees of skew over its length
ANGLED_ROW_RANGE = (math.radians(20.0), math.radians(75.0))
SIDE_ALIGNMENT = math.radians(15.0)
KERB_SEARCH_M = 20.0
PARKED_CAR_LENGTH_M = (3.8, 5.3)  # a parked car's aisle-side flank, fully seen
PARALLEL_BAY_LENGTH_M = (6.0, 7.2)
MIN_PARALLEL_CARS = 3
MAX_KERB_SNAP_M = 3.5
FLANK_GAP_M = 0.4  # gaps between parked cars along a kerb are wider than this
END_WALL_THICKNESS_M = 0.8
END_WALL_SLACK_M = 1.0  # missed cells along a wall's length
END_WALL_REACH_M = 5.5  # past the parked-car band into the aisle: longer than any car, as a wall across the row end is
END_CLEARANCE_M = 0.6
TYPICAL_PARALLEL_BAY_M = 6.6
MAX_SNAP_M = 1.5  # over half a bay: any anchor in the run reaches its nearest lattice bay
CLOSING_SIDE_M = 2.1  # a closing car's side seen along the bay depth; a car nose is ~1.9 m
MAX_CAR_SIDE_M = 6.0
WALL_PROBE_EXTRA_M = 3.5


def row_candidates(u: np.ndarray, axes: list[float]) -> list[np.ndarray]:
    """Possible row directions for a bay with axis ``u``.

    Square to the bay in a perpendicular row; in an angled row, the kerb or
    aisle direction, which is one of the map's dominant edge directions at
    20-75 degrees to the bay axis.
    """
    v = np.array([-u[1], u[0]])
    candidates = [v]
    bay = math.atan2(u[1], u[0])
    for axis in axes:
        angle = abs((axis - bay + math.pi / 2) % math.pi - math.pi / 2)
        if ANGLED_ROW_RANGE[0] <= angle <= ANGLED_ROW_RANGE[1]:
            direction = np.array([math.cos(axis), math.sin(axis)])
            candidates.append(direction if direction @ v > 0 else -direction)
    return candidates


def whole_cars(cars: list[tuple[float, float, bool]], across: float) -> list[float]:
    """Centres (in bay-width units along the bay normal) of cars seen across their full width."""
    return [(first + last) / 2.0 * across for first, last, _ in cars if WHOLE_CAR_WIDTH_M[0] <= (last - first) * across <= WHOLE_CAR_WIDTH_M[1]]


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
    # The cars closing the run anchor the lattice next to it: each must be
    # seen across its whole width, not a merged outline (12014: two cars
    # seen as one 3.3 m outline, the lattice half a bay off beside it).
    if not all(WHOLE_CAR_WIDTH_M[0] <= (car[1] - car[0]) * across <= WHOLE_CAR_WIDTH_M[1] for car in (last_behind, first_ahead)):
        return None
    whole = whole_cars(cars, across)
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


def fit_lattice(centres: list[float], width_range: tuple[float, float] = BAY_WIDTH_RANGE_M, min_cars: int = MIN_LATTICE_CARS) -> tuple[float, float] | None:
    """(bay width, phase) of the regular lattice the car centres sit on, or None.

    Each width is scored by how tightly the centres agree in phase (the
    length of their mean unit phasor); the best one must be clearly regular.
    """
    if len(centres) < min_cars:
        return None
    x = np.asarray(centres)
    widths = np.arange(width_range[0], width_range[1] + 1e-9, 0.005)
    phasors = np.exp(2j * np.pi * x[None, :] / widths[:, None]).mean(axis=1)
    coherence = np.abs(phasors)
    best = int(np.argmax(coherence))
    if coherence[best] < LATTICE_COHERENCE:
        return None
    # Cars far apart fit several widths almost equally well (7 x 2.45 m is
    # nearly 6 x 2.83 m); a best width on the edge of the search range is
    # such an alias, not a measurement.
    if widths[best] - widths[0] < LATTICE_EDGE_M or widths[-1] - widths[best] < LATTICE_EDGE_M:
        return None
    width = float(widths[best])
    phase = float(np.angle(phasors[best]) / (2 * np.pi) * width)
    residual = (x - phase + width / 2) % width - width / 2
    if float(np.sqrt(np.mean(residual ** 2))) > LATTICE_MAX_RMS_M:
        return None
    return width, phase


def kerb_profile(grid: OccupancyGrid, states: np.ndarray, center: np.ndarray, u: np.ndarray, aisle: np.ndarray, band: tuple[float, float]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Occupied runs along the kerb in the parked-car band, and which of them are walls.

    A probe across the band (``band`` metres along ``aisle``) slides along
    the kerb. A parked car shows its aisle-side flank: one line across the
    band, as long as the car. A wall across the kerb fills the band and the
    aisle beyond it, but is thin. Returns (all runs, wall runs), as (first, last) offsets along ``u``.
    """
    ts = np.arange(-KERB_SEARCH_M, KERB_SEARCH_M + 1e-9, STEP_M)
    occupied = _probe(grid, states, center, aisle, u, ts, np.arange(band[0], band[1], 0.15))
    # A wall runs on across the aisle; the end face of a car is only a car wide.
    # A wall is unbroken; the fronts of a row of parked cars have gaps.
    filled = _longest_run(_probe(grid, states, center, aisle, u, ts, np.arange(band[0], band[1] + END_WALL_REACH_M, 0.15))) * 0.15 >= band[1] - band[0] + END_WALL_REACH_M - END_WALL_SLACK_M
    runs: list[tuple[float, float]] = []
    walls: list[tuple[float, float]] = []
    start = None
    for index, value in enumerate(np.append(occupied.any(axis=1), False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            first, last = float(ts[start]), float(ts[index - 1])
            if last - first <= END_WALL_THICKNESS_M and filled[start:index].any():
                walls.append((first, last))
                runs.append((first, last))
                start = None
                continue
            # A flank seen in pieces (a few cells missed) is still one car.
            if runs and runs[-1] not in walls and first - runs[-1][1] <= FLANK_GAP_M:
                first = runs.pop()[0]
            runs.append((first, last))
            start = None
    return runs, walls


def wall_end_offset(runs: list[tuple[float, float]], walls: list[tuple[float, float]], car_length: float) -> tuple[float, tuple[float, float]] | None:
    """(offset along the kerb to a parallel bay between a wall and a parked car, that car's run), or None.

    At the wall end of a parallel row the bays cannot be counted from the
    cars, so the car does what a driver does: with a wall across the kerb
    on one side and a car parked along the kerb (its whole flank seen) on
    the other, it parks one bay length from that car, as far as the gap
    allows.
    """
    behind = [run for run in runs if run[1] < 0]
    ahead = [run for run in runs if run[0] > 0]
    if not behind or not ahead:
        return None
    near_behind, near_ahead = max(behind), min(ahead)

    def whole_car(run):
        return run not in walls and PARKED_CAR_LENGTH_M[0] <= run[1] - run[0] <= PARKED_CAR_LENGTH_M[1]

    if near_behind in walls and whole_car(near_ahead):
        wall_side, car, toward_wall = near_behind[1], near_ahead, -1.0
    elif near_ahead in walls and whole_car(near_behind):
        wall_side, car, toward_wall = near_ahead[0], near_behind, 1.0
    else:
        return None
    car_end = car[0] if toward_wall < 0 else car[1]
    if abs(car_end - wall_side) < car_length + 2.0 * END_CLEARANCE_M:
        return None
    # The neighbouring bay is one bay length from the parked car's centre
    # (it sits roughly centred in its own bay), kept clear of the car and
    # of the wall.
    target = (car[0] + car[1]) / 2.0 + toward_wall * TYPICAL_PARALLEL_BAY_M
    half = car_length / 2.0 + END_CLEARANCE_M
    low, high = sorted((wall_side - toward_wall * half, car_end + toward_wall * half))
    return float(np.clip(target, low, high)), car


def kerb_slot_offset(grid: OccupancyGrid, states: np.ndarray, center: np.ndarray, u: np.ndarray, aisle: np.ndarray, band: tuple[float, float], car_length: float) -> float | None:
    """Offset along the kerb from ``center`` to the parallel bay it falls in, or None.

    The same lattice idea along a kerb: a probe across the parked-car band
    (``band`` metres along ``aisle``, short of the kerb) slides along the
    kerb; each parked car shows its aisle-side flank as a run as long as
    the car. Whole cars fix the bay length and phase; the bay must lie
    between whole cars and clear of every parked car. A run longer than any
    car is a wall along the kerb line: no bay is placed.
    """
    runs, _ = kerb_profile(grid, states, center, u, aisle, band)
    if any(last - first > MAX_CAR_SIDE_M for first, last in runs):
        return None
    whole = [(first + last) / 2.0 for first, last in runs if PARKED_CAR_LENGTH_M[0] <= last - first <= PARKED_CAR_LENGTH_M[1]]
    lattice = fit_lattice(whole, PARALLEL_BAY_LENGTH_M, MIN_PARALLEL_CARS)
    if lattice is None:
        return None
    length, phase = lattice
    bay = phase + round(-phase / length) * length
    if not min(whole) < bay < max(whole) or abs(bay) > MAX_KERB_SNAP_M:
        return None
    half = car_length / 2.0 - LATTICE_MARGIN_M
    if any(first < bay + half and bay - half < last for first, last in runs):
        return None
    return bay
