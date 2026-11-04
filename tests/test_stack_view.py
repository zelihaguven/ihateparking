import math
import unittest

import numpy as np

from ihateparking.geometry import Pose
from ihateparking.mission import MissionConfig, MissionEvent, MissionSnapshot, run_mission
from ihateparking.occupancy_map import OccupancyGrid
from ihateparking.procedural_lot import generate_lot
from ihateparking.slot_detector import SlotCandidate
from ihateparking.stack_view import ESTIMATE_RGB, OCCUPIED_RGB, TARGET_RGB, UNKNOWN_RGB, StackView
from ihateparking.world import VehicleSpec


class MissionWatcherTests(unittest.TestCase):
    def test_watching_does_not_change_the_mission(self) -> None:
        snapshots = []
        watched = run_mission(generate_lot(8), MissionConfig(localization="scan_matching"), seed=8, watcher=snapshots.append)
        plain = run_mission(generate_lot(8), MissionConfig(localization="scan_matching"), seed=8)
        self.assertEqual((watched.outcome, watched.time_s, watched.distance_m), (plain.outcome, plain.time_s, plain.distance_m))
        self.assertGreater(len(snapshots), 10)
        # One snapshot per scan, in time order, and the last one is the outcome.
        self.assertEqual([s.time_s for s in snapshots], sorted(s.time_s for s in snapshots))
        self.assertEqual(snapshots[-1].mode, watched.outcome)
        self.assertEqual(snapshots[-1].estimate, watched.estimate_trace[-1])


class StackViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vehicle = VehicleSpec()
        self.grid = OccupancyGrid((0.0, 0.0))
        self.view = StackView(self.vehicle, centre=(0.0, 0.0), scale=10.0, size=(400, 300))

    def snapshot(self, **changes) -> MissionSnapshot:
        fields = dict(time_s=1.0, mode="explore", grid=self.grid, estimate=Pose(0.0, 0.0, 0.0), true_pose=Pose(0.0, 0.0, 0.0), plan=None, target=None, candidates=(), rejected=(), events=(MissionEvent(0.5, "explore", "exploring"),))
        fields.update(changes)
        return MissionSnapshot(**fields)

    def test_world_to_pixel_puts_north_up_at_the_given_scale(self) -> None:
        np.testing.assert_allclose(self.view.to_pixel(0.0, 0.0), (200.0, 150.0))
        np.testing.assert_allclose(self.view.to_pixel(5.0, 2.0), (250.0, 130.0))

    def test_draws_the_map_the_car_and_its_bay_where_they_are(self) -> None:
        row, col = self.grid.to_cell(-10.0, 5.0)
        self.grid.log_odds[row, col] = 10.0  # one occupied cell
        self.grid.observed[row, col] = True
        car_centre_x = self.vehicle.length / 2 - self.vehicle.rear_overhang
        target = SlotCandidate((8.0, -6.0), math.pi / 2, "side_by_side", 1.0)
        image = self.view.render(self.snapshot(target=target))
        self.assertEqual(image.shape, (300, 400, 3))
        self.assertEqual(tuple(image[5, 395]), UNKNOWN_RGB)
        self.assertEqual(tuple(image[100, 100]), OCCUPIED_RGB)  # (-10, 5) m
        # The estimate is drawn filled over the unknown map, blended towards its colour.
        x, y = self.view.to_pixel(car_centre_x, 0.0).round().astype(int)
        blended = image[y, x].astype(int)
        self.assertLess(np.abs(blended - np.array(ESTIMATE_RGB)).sum(), np.abs(blended - np.array(UNKNOWN_RGB)).sum())
        # The bay's long edge, half a car width right of its centre (it points north).
        x, y = self.view.to_pixel(8.0 + self.vehicle.width / 2, -6.0).round().astype(int)
        self.assertTrue(any(tuple(image[y, x + dx]) == TARGET_RGB for dx in (-1, 0, 1)))


if __name__ == "__main__":
    unittest.main()
