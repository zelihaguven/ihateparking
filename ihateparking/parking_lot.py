"""A seeded multi-row parking lot with mixed bay types and occupancy.

Unlike the single-slot scenarios, the lot does not say how to park: it only
lists bays (id, type, footprint) and which ones hold a parked car. Deciding
the approach, the final heading and the gear sequence is the job of
``ihateparking.autopark``.

Layout (metres, x east, y north)::

    y=28 ───────────────────────────────── wall
          aisle 2 (8 m)
    y=20  ╱╱╱╱╱╱╱╱  row C: angled bays, open to aisle 2
    y=15  ────────  separator curb
          ▮▮▮▮▮▮▮▮  row B: perpendicular bays, open to aisle 1
    y=10
          aisle 1 (7 m)            the aisles connect east of the rows
    y=3   ▭ ▭ ▭ ▭ ▭  row A: parallel bays along the south wall
    y=0 ───────────────────────────────── wall
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal

from .geometry import Pose
from .world import ParkingSlot, RectangleObstacle, VehicleSpec

BayKind = Literal["parallel", "perpendicular", "angled"]

LOT_BOUNDS = (0.0, 40.0, 0.0, 28.0)
DEFAULT_START = Pose(3.0, 6.5, 0.0)
PARKED_CAR_LENGTH_M = 4.5
PARKED_CAR_WIDTH_M = 1.9

PARALLEL_BAY_LENGTH_M = 6.5
PARALLEL_BAY_WIDTH_M = 2.5
PERPENDICULAR_BAY_DEPTH_M = 5.0
PERPENDICULAR_BAY_WIDTH_M = 2.7
ANGLED_BAY_DEPTH_M = 5.0
ANGLED_BAY_WIDTH_M = 2.7
ANGLED_BAY_ANGLE = math.radians(60.0)
SEPARATOR_Y = 15.0
ROW_WEST_X = 5.0


@dataclass(frozen=True)
class Bay:
    bay_id: str
    kind: BayKind
    slot: ParkingSlot

    @property
    def center(self) -> tuple[float, float]:
        return self.slot.center


@dataclass(frozen=True)
class ParkingLot:
    seed: int
    bounds: tuple[float, float, float, float]
    bays: tuple[Bay, ...]
    occupied: frozenset[str]
    parked_cars: tuple[RectangleObstacle, ...]
    fixtures: tuple[RectangleObstacle, ...]
    vehicle: VehicleSpec = VehicleSpec()

    def bay(self, bay_id: str) -> Bay:
        for bay in self.bays:
            if bay.bay_id == bay_id:
                return bay
        raise KeyError(f"unknown bay: {bay_id}")

    def is_free(self, bay_id: str) -> bool:
        self.bay(bay_id)
        return bay_id not in self.occupied

    @property
    def free_bays(self) -> tuple[Bay, ...]:
        return tuple(bay for bay in self.bays if bay.bay_id not in self.occupied)

    @property
    def obstacles(self) -> list[RectangleObstacle]:
        """Everything the ego must not touch: parked cars and fixed curbs."""
        return [*self.fixtures, *self.parked_cars]


def _parallel_row() -> list[Bay]:
    bays = []
    for index in range(5):
        x = 6.0 + PARALLEL_BAY_LENGTH_M * (index + 0.5)
        slot = ParkingSlot((x, PARALLEL_BAY_WIDTH_M / 2.0 + 0.25), PARALLEL_BAY_LENGTH_M, PARALLEL_BAY_WIDTH_M, 0.0)
        bays.append(Bay(f"A{index + 1}", "parallel", slot))
    return bays


def _perpendicular_row() -> list[Bay]:
    bays = []
    y = SEPARATOR_Y - PERPENDICULAR_BAY_DEPTH_M / 2.0
    for index in range(9):
        x = ROW_WEST_X + PERPENDICULAR_BAY_WIDTH_M * (index + 0.5)
        # Slot yaw points into the bay (north): nose-in heading.
        slot = ParkingSlot((x, y), PERPENDICULAR_BAY_DEPTH_M, PERPENDICULAR_BAY_WIDTH_M, math.pi / 2.0)
        bays.append(Bay(f"B{index + 1}", "perpendicular", slot))
    return bays


def _angled_row() -> list[Bay]:
    bays = []
    # Bays open to aisle 2 (north) and lean east; nose-in heads south-west.
    yaw = -(math.pi - ANGLED_BAY_ANGLE)
    spacing = ANGLED_BAY_WIDTH_M / math.sin(ANGLED_BAY_ANGLE)
    # The bay's lowest corner rests on top of the separator curb.
    half_height = (ANGLED_BAY_DEPTH_M / 2.0) * math.sin(ANGLED_BAY_ANGLE) + (ANGLED_BAY_WIDTH_M / 2.0) * math.cos(ANGLED_BAY_ANGLE)
    center_y = SEPARATOR_Y + 0.15 + half_height
    for index in range(7):
        x = ROW_WEST_X + 3.0 + spacing * index
        slot = ParkingSlot((x, center_y), ANGLED_BAY_DEPTH_M, ANGLED_BAY_WIDTH_M, yaw)
        bays.append(Bay(f"C{index + 1}", "angled", slot))
    return bays


def _parked_car(bay: Bay, rng: random.Random) -> RectangleObstacle:
    """A parked car roughly centred in its bay, with small placement noise."""
    slot = bay.slot
    yaw = slot.yaw + rng.uniform(-0.03, 0.03)
    if bay.kind == "parallel":
        along = rng.uniform(-0.4, 0.4)
        across = rng.uniform(-0.1, 0.1)
    else:
        along = rng.uniform(-0.2, 0.2)
        across = rng.uniform(-0.15, 0.15)
    c, s = math.cos(slot.yaw), math.sin(slot.yaw)
    center = (slot.center[0] + c * along - s * across, slot.center[1] + s * along + c * across)
    return RectangleObstacle(f"car_{bay.bay_id}", center, PARKED_CAR_LENGTH_M, PARKED_CAR_WIDTH_M, yaw)


def build_lot(seed: int = 0, occupancy: float = 0.5, keep_free: tuple[str, ...] = ()) -> ParkingLot:
    """Build the lot; each bay is occupied with probability ``occupancy``.

    ``keep_free`` forces bays to stay empty, which is useful for tests and
    demos that need a specific target.
    """
    if not 0.0 <= occupancy <= 1.0:
        raise ValueError("occupancy must be between 0 and 1")
    rng = random.Random(seed)
    bays = (*_parallel_row(), *_perpendicular_row(), *_angled_row())
    occupied = frozenset(
        bay.bay_id for bay in bays if bay.bay_id not in keep_free and rng.random() < occupancy
    )
    parked = tuple(_parked_car(bay, rng) for bay in bays if bay.bay_id in occupied)
    row_b_east = ROW_WEST_X + PERPENDICULAR_BAY_WIDTH_M * 9
    fixtures = (
        RectangleObstacle("separator_curb", ((ROW_WEST_X + row_b_east) / 2.0, SEPARATOR_Y), row_b_east - ROW_WEST_X, 0.3, 0.0),
    )
    return ParkingLot(seed, LOT_BOUNDS, bays, occupied, parked, fixtures)
