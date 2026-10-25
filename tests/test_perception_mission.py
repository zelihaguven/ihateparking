import math
import unittest

import numpy as np

from ihateparking.geometry import Pose, polygons_intersect
from ihateparking.lidar import LidarConfig, scan, sensor_origin
from ihateparking.mission import evaluate_parking, run_mission, scenario_lot
from ihateparking.occupancy_map import FREE, OCCUPIED, OccupancyGrid
from ihateparking.procedural_lot import generate_lot
from ihateparking.row_lattice import fit_lattice, snap_to_row
from ihateparking.slot_detector import detect_slots
from ihateparking.world import ParkingWorld, RectangleObstacle, VehicleSpec

EXACT = LidarConfig(beams=360, range_noise_std=0.0, dropout=0.0)


def _march(origin, angle, obstacles, max_range, step=0.005):
    for k in range(1, int(max_range / step)):
        x, y = origin[0] + math.cos(angle) * k * step, origin[1] + math.sin(angle) * k * step
        for obstacle in obstacles:
            c, s = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
            dx, dy = x - obstacle.center[0], y - obstacle.center[1]
            if abs(c * dx + s * dy) <= obstacle.length / 2 and abs(-s * dx + c * dy) <= obstacle.width / 2:
                return k * step
    return math.inf


class LidarTests(unittest.TestCase):
    def test_ranges_match_ray_marching(self) -> None:
        obstacles = [RectangleObstacle("a", (6.0, 1.0), 4.5, 1.9, 0.4), RectangleObstacle("b", (-3.0, -5.0), 0.3, 12.0, 0.0)]
        pose = Pose(0.0, 0.0, 0.2)
        result = scan(pose, obstacles, VehicleSpec(), EXACT, np.random.default_rng(0))
        for index in range(0, 360, 9):
            expected = _march(result.origin, result.angles[index], obstacles, EXACT.max_range)
            if math.isinf(expected):
                self.assertFalse(result.hits[index])
            else:
                self.assertAlmostEqual(result.ranges[index], expected, delta=0.01)

    def test_beam_parallel_to_a_wall_does_not_hit_it(self) -> None:
        # Regression: an exactly parallel beam used to report range 0.
        wall = RectangleObstacle("wall", (0.0, 5.0), 20.0, 0.3, 0.0)
        result = scan(Pose(0.0, 0.0, 0.0), [wall], VehicleSpec(), EXACT, np.random.default_rng(0))
        self.assertGreater(result.ranges[result.hits].min(), 4.0)


class OccupancyGridTests(unittest.TestCase):
    def test_scan_marks_free_space_and_the_obstacle(self) -> None:
        wall = RectangleObstacle("wall", (0.0, 6.0), 30.0, 0.3, 0.0)
        pose = Pose(0.0, 0.0, 0.0)
        grid = OccupancyGrid((0.0, 0.0))
        for _ in range(3):
            grid.integrate(scan(pose, [wall], VehicleSpec(), EXACT, np.random.default_rng(0)))
        states = grid.states()
        ox, oy = sensor_origin(pose, VehicleSpec())
        row, col = grid.to_cell(ox, oy + 3.0)
        self.assertEqual(states[row, col], FREE)
        row, col = grid.to_cell(ox, 5.9)
        self.assertEqual(states[row, col], OCCUPIED)

    def test_dropouts_do_not_punch_free_space_through_walls(self) -> None:
        wall = RectangleObstacle("wall", (0.0, 6.0), 60.0, 0.3, 0.0)
        noisy = LidarConfig(dropout=0.2)
        grid = OccupancyGrid((0.0, 0.0))
        rng = np.random.default_rng(3)
        for _ in range(20):
            grid.integrate(scan(Pose(0.0, 0.0, 0.0), [wall], VehicleSpec(), noisy, rng))
        states = grid.states()
        rows, cols = np.nonzero(states == FREE)
        _, ys = grid.cell_center(rows, cols)
        self.assertLess(ys.max(), 6.2)


def _aisle_sweep(seed: int):
    """The lot and the map from driving straight along the start aisle."""
    lot = generate_lot(seed)
    world = ParkingWorld(lot.bounds, list(lot.obstacles), lot.bays[0].slot, lot.start, lot.vehicle)
    grid = OccupancyGrid((lot.start.x, lot.start.y))
    rng = np.random.default_rng(seed)
    for d in np.arange(-20.0, 20.1, 1.0):
        pose = Pose(lot.start.x + d * math.cos(lot.start.yaw), lot.start.y + d * math.sin(lot.start.yaw), lot.start.yaw)
        if not world.is_collision(pose):
            grid.integrate(scan(pose, lot.obstacles, lot.vehicle, rng=rng))
    return lot, grid


class SlotDetectorTests(unittest.TestCase):
    def test_finds_bays_in_angled_rows(self) -> None:
        # Regression: staggered neighbours in angled rows used to be read as
        # a wall and a missing flank, so angled bays were never detected.
        for seed in (105, 112):
            lot, grid = _aisle_sweep(seed)
            kinds = [evaluate_parking(lot, c.goal(c.axis_yaw, lot.vehicle)).get("bay_kind") for c in detect_slots(grid, lot.vehicle)]
            with self.subTest(seed=seed):
                self.assertIn("angled", kinds)

    def test_finds_bays_in_a_run_of_free_bays(self) -> None:
        # Bays with no parked neighbour on one or both sides are placed from
        # the lattice of the row's parked cars.
        for seed in (131, 133):
            lot, grid = _aisle_sweep(seed)
            found = []
            for candidate in detect_slots(grid, lot.vehicle):
                verdict = evaluate_parking(lot, candidate.goal(candidate.axis_yaw, lot.vehicle))
                bay = next((b for b in lot.bays if b.bay_id == verdict.get("bay")), None)
                if bay is None or not verdict["in_bay"]:
                    continue
                neighbours = [o for o in lot.bays if o is not bay and o.kind == bay.kind and math.dist(o.slot.center, bay.slot.center) < bay.slot.width * 1.45]
                found.append(sum(o.occupied for o in neighbours))
            with self.subTest(seed=seed):
                self.assertTrue(any(flanks < 2 for flanks in found), found)

    def test_detections_on_unseen_lots_are_real_free_bays(self) -> None:
        checked = 0
        for seed in (101, 102, 103, 104, 105, 112, 123, 131, 133):
            lot, grid = _aisle_sweep(seed)
            for candidate in detect_slots(grid, lot.vehicle):
                pose = candidate.goal(candidate.axis_yaw, lot.vehicle)
                verdict = evaluate_parking(lot, pose)
                with self.subTest(seed=seed, center=candidate.center):
                    self.assertTrue(verdict["in_bay"], verdict)
                checked += 1
        self.assertGreater(checked, 0)


class RowLatticeTests(unittest.TestCase):
    def test_fit_recovers_bay_width_and_phase_from_jittered_cars(self) -> None:
        rng = np.random.default_rng(0)
        bays = [k for k in range(-6, 7) if k not in (0, 1, 2)]  # a run of three free bays
        centres = [0.4 + 2.62 * k + rng.uniform(-0.45, 0.45) for k in bays]
        width, phase = fit_lattice(centres)
        self.assertAlmostEqual(width, 2.62, delta=0.05)
        self.assertAlmostEqual((phase - 0.4 + width / 2) % width - width / 2, 0.0, delta=0.2)

    def test_no_bay_past_the_last_car_of_a_row(self) -> None:
        # Regression (seed 9021, parallel scenario): past the end of a row a
        # wall at a slight angle looked like a car closing a run, and the car
        # parked beyond the last parked car. Cars at -2..-14 m, "closer" at 6 m.
        cars = [(-1.95 - 2.65 * k, -0.05 - 2.65 * k, True) for k in range(5)] + [(6.0, 6.4, True)]
        self.assertIsNone(snap_to_row(sorted(cars), 1.0, 1.9))

    def test_fit_rejects_irregular_spacing(self) -> None:
        self.assertIsNone(fit_lattice([0.0, 2.1, 5.9, 7.3, 11.8]))
        self.assertIsNone(fit_lattice([0.0, 2.6, 5.2]))  # too few cars


class ProceduralLotTests(unittest.TestCase):
    def test_parallel_scenario_leaves_only_parallel_bays_free(self) -> None:
        for seed in range(9000, 9010):
            lot = scenario_lot(seed, "parallel")
            with self.subTest(seed=seed):
                self.assertTrue(any(bay.kind == "parallel" for bay in lot.bays))
                self.assertTrue(all(bay.kind == "parallel" for bay in lot.free_bays))
                self.assertEqual(scenario_lot(seed), generate_lot(seed))

    def test_lots_differ_and_are_consistent(self) -> None:
        first, second = generate_lot(1), generate_lot(2)
        self.assertNotEqual(first.rotation, second.rotation)
        for seed in range(20):
            lot = generate_lot(seed)
            fixed = [item for item in lot.obstacles if not item.name.startswith("car")]
            cars = [item for item in lot.obstacles if item.name.startswith("car")]
            world = ParkingWorld(lot.bounds, list(lot.obstacles), lot.bays[0].slot, lot.start, lot.vehicle)
            with self.subTest(seed=seed):
                self.assertFalse(world.is_collision(lot.start))
                self.assertFalse(any(polygons_intersect(car.polygon(), item.polygon()) for car in cars for item in fixed))


class MissionTests(unittest.TestCase):
    def test_parks_in_an_unseen_lot_using_only_lidar(self) -> None:
        lot = generate_lot(8)
        result = run_mission(lot, seed=8)
        self.assertEqual(result.outcome, "PARKED", result.message)
        self.assertTrue(result.success, result.evaluation)
        self.assertEqual(result.collisions, 0)
        self.assertGreaterEqual(result.evaluation["overlap"], 0.8)

    def test_evaluation_rejects_a_pose_in_the_aisle(self) -> None:
        lot = generate_lot(8)
        verdict = evaluate_parking(lot, lot.start)
        self.assertFalse(verdict["in_bay"])


if __name__ == "__main__":
    unittest.main()
