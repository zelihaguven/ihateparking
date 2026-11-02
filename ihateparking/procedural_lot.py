"""Procedurally generated, never-seen parking lots for perception-driven parking.

Every seed gives a different lot: size, number and type of rows (parallel,
perpendicular, angled at 45-70 degrees leaning either way), bay dimensions,
aisle widths, open or closed row ends, pillars, occupancy, parked-car sizes
and crooked parking, plus a random rotation of the whole lot.

This module is the *ground truth*. The autonomous stack never reads it: the
LiDAR simulator ray-casts against its obstacles, and the evaluator uses its
bay list to score where the car ended up.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal

from .geometry import Pose, wrap_angle
from .world import ParkingSlot, RectangleObstacle, VehicleSpec

BayKind = Literal["parallel", "perpendicular", "angled"]
WALL_THICKNESS_M = 0.3
KERB_THICKNESS_M = 0.25


@dataclass(frozen=True)
class TruthBay:
    bay_id: str
    kind: BayKind
    slot: ParkingSlot  # yaw points into the bay (nose-in heading)
    occupied: bool


@dataclass(frozen=True)
class ProceduralLot:
    seed: int
    bounds: tuple[float, float, float, float]
    obstacles: tuple[RectangleObstacle, ...]
    bays: tuple[TruthBay, ...]
    start: Pose
    rotation: float
    vehicle: VehicleSpec = VehicleSpec()

    @property
    def free_bays(self) -> tuple[TruthBay, ...]:
        return tuple(bay for bay in self.bays if not bay.occupied)


@dataclass
class _Builder:
    rng: random.Random
    width: float
    obstacles: list[RectangleObstacle]
    bays: list[TruthBay]
    occupancy: float
    free_kinds: tuple[BayKind, ...] | None = None

    def add_box(self, name: str, x0: float, x1: float, y0: float, y1: float) -> None:
        self.obstacles.append(RectangleObstacle(name, ((x0 + x1) / 2, (y0 + y1) / 2), x1 - x0, y1 - y0, 0.0))

    def add_bay(self, kind: BayKind, slot: ParkingSlot) -> None:
        bay_id = f"{kind[0].upper()}{len(self.bays) + 1:02d}"
        occupied = self.rng.random() < self.occupancy
        if self.free_kinds is not None and kind not in self.free_kinds:
            occupied = True
        self.bays.append(TruthBay(bay_id, kind, slot, occupied))
        if occupied:
            self.obstacles.append(self._parked_car(bay_id, kind, slot))

    def _parked_car(self, bay_id: str, kind: BayKind, slot: ParkingSlot) -> RectangleObstacle:
        rng = self.rng
        length = rng.uniform(4.2, 4.9)
        width = rng.uniform(1.75, 2.0)
        # Narrow parallel bays sit against a wall; keep crooked cars inside.
        skew = 1.5 if kind == "parallel" else 4.0
        yaw = slot.yaw + math.radians(rng.uniform(-skew, skew))
        along_room = max(0.0, (slot.length - length) / 2 - 0.1)
        across_room = max(0.0, (slot.width - width) / 2 - 0.08)
        along = rng.uniform(-along_room, along_room)
        across = rng.uniform(-across_room, across_room)
        c, s = math.cos(slot.yaw), math.sin(slot.yaw)
        center = (slot.center[0] + c * along - s * across, slot.center[1] + s * along + c * across)
        return RectangleObstacle(f"car_{bay_id}", center, length, width, yaw)


@dataclass(frozen=True)
class _RowSpec:
    kind: BayKind
    length: float  # along the bay axis
    width: float  # across the bay axis
    angle: float = math.pi / 2  # angled rows only
    lean: float = 1.0  # angled rows: lean along +x or -x
    parallel_heading: float = 0.0

    @property
    def depth(self) -> float:
        """Extent of the row away from its back edge."""
        if self.kind == "parallel":
            return self.width
        if self.kind == "perpendicular":
            return self.length
        return self.length * math.sin(self.angle) + self.width * math.cos(self.angle) + 0.05


def _row_spec(rng: random.Random, kind: BayKind) -> _RowSpec:
    if kind == "parallel":
        return _RowSpec(kind, rng.uniform(6.2, 7.0), rng.uniform(2.3, 2.6), parallel_heading=rng.choice([0.0, math.pi]))
    if kind == "perpendicular":
        return _RowSpec(kind, rng.uniform(5.0, 5.5), rng.uniform(2.5, 2.9))
    return _RowSpec(kind, rng.uniform(4.8, 5.3), rng.uniform(2.5, 2.8), math.radians(rng.uniform(45.0, 70.0)), rng.choice([1.0, -1.0]))


def _place_row(builder: _Builder, spec: _RowSpec, base_y: float, opens_up: bool, x0: float, x1: float) -> None:
    """Fill a row whose back edge lies on ``base_y``.

    ``opens_up`` means the aisle is on the +y side, so bays extend upwards from
    the back edge and the nose-in heading points down (towards the back).
    """
    sign = 1.0 if opens_up else -1.0
    if spec.kind == "parallel":
        count = int((x1 - x0) // spec.length)
        start = x0 + ((x1 - x0) - count * spec.length) / 2
        for index in range(count):
            center = (start + spec.length * (index + 0.5), base_y + sign * spec.width / 2)
            builder.add_bay(spec.kind, ParkingSlot(center, spec.length, spec.width, spec.parallel_heading))
        return
    if spec.kind == "perpendicular":
        count = int((x1 - x0) // spec.width)
        start = x0 + ((x1 - x0) - count * spec.width) / 2
        heading = -sign * math.pi / 2
        for index in range(count):
            center = (start + spec.width * (index + 0.5), base_y + sign * spec.length / 2)
            builder.add_bay(spec.kind, ParkingSlot(center, spec.length, spec.width, heading))
        return
    angle = spec.angle
    pitch = spec.width / math.sin(angle)
    half_x = spec.length / 2 * math.cos(angle) + spec.width / 2 * math.sin(angle)
    half_y = spec.length / 2 * math.sin(angle) + spec.width / 2 * math.cos(angle)
    count = int((x1 - x0 - 2 * half_x) // pitch) + 1
    if x1 - x0 < 2 * half_x:
        return
    start = x0 + half_x + ((x1 - x0 - 2 * half_x) - (count - 1) * pitch) / 2
    heading = math.atan2(-sign * math.sin(angle), spec.lean * math.cos(angle))
    for index in range(count):
        center = (start + pitch * index, base_y + sign * (half_y + 0.05))
        builder.add_bay(spec.kind, ParkingSlot(center, spec.length, spec.width, heading))


def _pick_kind(rng: random.Random, allow_parallel: bool) -> BayKind:
    kinds: list[BayKind] = ["perpendicular", "angled"] + (["parallel"] if allow_parallel else [])
    return rng.choice(kinds)


def generate_lot(
    seed: int,
    occupancy: float | None = None,
    vehicle: VehicleSpec | None = None,
    wall_row_kind: BayKind | None = None,
    free_kinds: tuple[BayKind, ...] | None = None,
) -> ProceduralLot:
    """Build a random lot; ``occupancy`` defaults to a seeded value in 0.4-0.8.

    ``wall_row_kind`` forces both rows along the walls to exist and be of that
    kind; ``free_kinds`` leaves only bays of those kinds free. Both default to
    the ordinary random lot, unchanged for every seed.
    """
    rng = random.Random(seed)
    vehicle = vehicle or VehicleSpec()
    width = rng.uniform(34.0, 48.0)
    occupancy = rng.uniform(0.4, 0.8) if occupancy is None else occupancy
    builder = _Builder(rng, width, [], [], occupancy, free_kinds)
    aisles: list[tuple[float, float]] = []

    y = 0.0
    if rng.random() < 0.75 or wall_row_kind:  # row along the bottom wall, opening upwards
        spec = _row_spec(rng, wall_row_kind or _pick_kind(rng, allow_parallel=True))
        _place_row(builder, spec, 0.0, True, 1.0, width - 1.0)
        y = spec.depth
    for block in range(rng.choice([1, 2])):
        aisle = rng.uniform(6.5, 8.0)
        aisles.append((y, y + aisle))
        y += aisle
        # A double row with a kerb between its halves. At least one end is
        # open so the aisles on either side connect.
        open_left, open_right = rng.choice([(True, True), (True, False), (False, True)])
        x0 = rng.uniform(6.5, 8.0) if open_left else 1.0
        x1 = width - (rng.uniform(6.5, 8.0) if open_right else 1.0)
        lower = _row_spec(rng, _pick_kind(rng, allow_parallel=False))
        y_kerb = y + lower.depth + KERB_THICKNESS_M / 2
        _place_row(builder, lower, y_kerb - KERB_THICKNESS_M / 2, False, x0, x1)
        builder.add_box(f"kerb_{block}", x0, x1, y_kerb - KERB_THICKNESS_M / 2, y_kerb + KERB_THICKNESS_M / 2)
        if rng.random() < 0.6:
            if open_left:
                builder.add_box(f"pillar_{block}w", x0 - 0.9, x0 - 0.3, y_kerb - 0.3, y_kerb + 0.3)
            if open_right:
                builder.add_box(f"pillar_{block}e", x1 + 0.3, x1 + 0.9, y_kerb - 0.3, y_kerb + 0.3)
        upper = _row_spec(rng, _pick_kind(rng, allow_parallel=False))
        _place_row(builder, upper, y_kerb + KERB_THICKNESS_M / 2, True, x0, x1)
        y = y_kerb + KERB_THICKNESS_M / 2 + upper.depth
    aisle = rng.uniform(6.5, 8.0)
    aisles.append((y, y + aisle))
    y += aisle
    if rng.random() < 0.75 or wall_row_kind:  # row along the top wall, opening downwards
        spec = _row_spec(rng, wall_row_kind or _pick_kind(rng, allow_parallel=True))
        y += spec.depth
        _place_row(builder, spec, y, False, 1.0, width - 1.0)
    height = y

    t = WALL_THICKNESS_M
    builder.add_box("wall_s", -t, width + t, -t, 0.0)
    builder.add_box("wall_n", -t, width + t, height, height + t)
    builder.add_box("wall_w", -t, 0.0, 0.0, height)
    builder.add_box("wall_e", width, width + t, 0.0, height)

    aisle_lo, aisle_hi = rng.choice(aisles)
    start = Pose(rng.uniform(5.0, width - 5.0), (aisle_lo + aisle_hi) / 2, rng.choice([0.0, math.pi]))
    rotation = rng.uniform(-math.pi, math.pi)
    return _rotated(seed, builder, start, rotation, width, height, vehicle)


def _rotated(seed: int, builder: _Builder, start: Pose, rotation: float, width: float, height: float, vehicle: VehicleSpec) -> ProceduralLot:
    cx, cy = width / 2, height / 2
    c, s = math.cos(rotation), math.sin(rotation)

    def point(x: float, y: float) -> tuple[float, float]:
        dx, dy = x - cx, y - cy
        return (cx + c * dx - s * dy, cy + s * dx + c * dy)

    obstacles = tuple(
        RectangleObstacle(item.name, point(*item.center), item.length, item.width, wrap_angle(item.yaw + rotation))
        for item in builder.obstacles
    )
    bays = tuple(
        TruthBay(bay.bay_id, bay.kind, ParkingSlot(point(*bay.slot.center), bay.slot.length, bay.slot.width, wrap_angle(bay.slot.yaw + rotation)), bay.occupied)
        for bay in builder.bays
    )
    sx, sy = point(start.x, start.y)
    start_pose = Pose(sx, sy, wrap_angle(start.yaw + rotation))
    corners = [point(x, y) for x, y in ((0, 0), (width, 0), (width, height), (0, height))]
    margin = 1.5
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    bounds = (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)
    return ProceduralLot(seed, bounds, obstacles, bays, start_pose, rotation, vehicle)
