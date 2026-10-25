import math
import unittest

import numpy as np

from ihateparking.geometry import Pose, wrap_angle
from ihateparking.lidar import LidarConfig, scan
from ihateparking.mission import MissionConfig, run_mission
from ihateparking.occupancy_map import OccupancyGrid
from ihateparking.odometry import Odometry, OdometryConfig, place_scan
from ihateparking.scan_matcher import ScanMatcher
from ihateparking.procedural_lot import generate_lot
from ihateparking.world import RectangleObstacle, VehicleSpec

VEHICLE = VehicleSpec()


def _drive(odometry: Odometry, steps: int = 400, speed: float = 1.5, steering: float = 0.1, dt: float = 0.05):
    """Drive an arc; return the true pose and the odometry estimate."""
    true_pose = estimate = Pose(0.0, 0.0, 0.0)
    for _ in range(steps):
        true_pose = Pose(
            true_pose.x + speed * dt * math.cos(true_pose.yaw),
            true_pose.y + speed * dt * math.sin(true_pose.yaw),
            wrap_angle(true_pose.yaw + speed * dt / VEHICLE.wheelbase * math.tan(steering)),
        )
        estimate = odometry.step(estimate, speed, steering, dt)
    return true_pose, estimate


class OdometryTests(unittest.TestCase):
    def test_noise_free_odometry_matches_the_true_motion(self) -> None:
        perfect = OdometryConfig(0.0, 0.0, 0.0, 0.0)
        true_pose, estimate = _drive(Odometry(VEHICLE, perfect, np.random.default_rng(0)))
        self.assertAlmostEqual(estimate.x, true_pose.x, places=9)
        self.assertAlmostEqual(estimate.y, true_pose.y, places=9)
        self.assertAlmostEqual(estimate.yaw, true_pose.yaw, places=9)

    def test_error_grows_with_distance_driven(self) -> None:
        errors = {}
        for steps in (100, 800):
            samples = []
            for seed in range(20):
                true_pose, estimate = _drive(Odometry(VEHICLE, OdometryConfig(), np.random.default_rng(seed)), steps=steps)
                samples.append(math.hypot(estimate.x - true_pose.x, estimate.y - true_pose.y))
            errors[steps] = float(np.median(samples))
        self.assertGreater(errors[800], 3 * errors[100])
        self.assertGreater(errors[800], 0.2)  # 60 m of driving drifts by decimetres


class PlaceScanTests(unittest.TestCase):
    def test_scan_follows_the_estimated_pose(self) -> None:
        wall = RectangleObstacle("wall", (10.0, 0.0), 0.3, 30.0, 0.0)
        exact = LidarConfig(beams=360, range_noise_std=0.0, dropout=0.0)
        true_pose = Pose(0.0, 0.0, 0.0)
        measured = scan(true_pose, [wall], VEHICLE, exact, np.random.default_rng(0))
        # The car believes it stands 1 m further along and turned by 90 degrees.
        estimate = Pose(1.0, 0.0, math.pi / 2)
        placed = place_scan(measured, true_pose, estimate, VEHICLE)
        ahead = int(np.argmin(np.abs(measured.angles)))
        x = placed.origin[0] + placed.ranges[ahead] * math.cos(placed.angles[ahead])
        y = placed.origin[1] + placed.ranges[ahead] * math.sin(placed.angles[ahead])
        # The wall straight ahead now appears straight ahead of the believed pose.
        self.assertAlmostEqual(x, placed.origin[0], delta=0.05)
        self.assertGreater(y, placed.origin[1] + 5.0)


class ScanMatcherTests(unittest.TestCase):
    OBSTACLES = [
        RectangleObstacle("wall", (0.0, 8.0), 30.0, 0.3, 0.0),
        RectangleObstacle("side", (12.0, 0.0), 0.3, 20.0, 0.0),
        RectangleObstacle("car", (4.0, -4.0), 4.5, 1.9, 0.3),
    ]

    def _mapped_grid(self) -> OccupancyGrid:
        grid = OccupancyGrid((0.0, 0.0))
        rng = np.random.default_rng(1)
        for x in (-3.0, -1.5, 0.0):
            grid.integrate(scan(Pose(x, 0.0, 0.0), self.OBSTACLES, VEHICLE, rng=rng))
        return grid

    def test_recovers_the_pose_from_a_drifted_estimate(self) -> None:
        grid = self._mapped_grid()
        true_pose = Pose(1.0, 0.5, 0.05)
        measured = scan(true_pose, self.OBSTACLES, VEHICLE, rng=np.random.default_rng(2))
        drifted = Pose(true_pose.x + 0.15, true_pose.y - 0.1, true_pose.yaw + math.radians(0.8))
        corrected = ScanMatcher(VEHICLE).correct(grid, measured, true_pose.yaw, drifted)
        self.assertLess(math.hypot(corrected.x - true_pose.x, corrected.y - true_pose.y), 0.04)
        self.assertLess(abs(math.degrees(wrap_angle(corrected.yaw - true_pose.yaw))), 0.2)

    def test_keeps_odometry_when_the_scan_sees_nothing_mapped(self) -> None:
        grid = self._mapped_grid()
        true_pose = Pose(1.0, 0.5, 0.05)
        empty = scan(true_pose, [], VEHICLE, rng=np.random.default_rng(2))
        drifted = Pose(1.2, 0.4, 0.06)
        matcher = ScanMatcher(VEHICLE)
        self.assertEqual(matcher.correct(grid, empty, true_pose.yaw, drifted), drifted)
        self.assertEqual(matcher.rejected, 1)


class MissionLocalizationTests(unittest.TestCase):
    def test_parks_on_odometry_alone_while_the_estimate_drifts(self) -> None:
        result = run_mission(generate_lot(8), MissionConfig(localization="odometry"), seed=8)
        self.assertTrue(result.success, result.evaluation)
        self.assertEqual(result.collisions, 0)
        self.assertGreater(result.localization_error_m, 0.0)

    def test_unknown_localization_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_mission(generate_lot(8), MissionConfig(localization="gps"), seed=8)


if __name__ == "__main__":
    unittest.main()
