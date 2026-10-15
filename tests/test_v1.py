import math
import unittest

from ihateparking.controller import ControllerConfig, simulate_closed_loop
from ihateparking.geometry import Pose, oriented_box, polygons_intersect
from ihateparking.planner import HybridAStar, PlannerConfig
from ihateparking.world import default_world


class GeometryTests(unittest.TestCase):
    def test_separated_boxes_do_not_intersect(self):
        left = oriented_box((0.0, 0.0), 2.0, 2.0)
        right = oriented_box((3.0, 0.0), 2.0, 2.0)
        self.assertFalse(polygons_intersect(left, right))

    def test_overlapping_boxes_intersect(self):
        first = oriented_box((0.0, 0.0), 2.0, 2.0)
        second = oriented_box((0.5, 0.0), 2.0, 2.0)
        self.assertTrue(polygons_intersect(first, second))


class ParkingV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = default_world()
        cls.plan = HybridAStar(cls.world).plan()

    def test_start_and_goal_are_safe(self):
        self.assertFalse(self.world.is_collision(self.world.start_pose))
        self.assertFalse(self.world.is_collision(self.world.slot.goal_pose(self.world.vehicle)))

    def test_hybrid_a_star_finds_a_plan(self):
        self.assertTrue(self.plan.success, self.plan.message)
        self.assertGreater(len(self.plan.poses), 2)
        self.assertEqual(len(self.plan.poses), len(self.plan.gears))
        self.assertLess(self.plan.path_length, 20.0)

    def test_closed_loop_parking_has_no_collision(self):
        result = simulate_closed_loop(self.world, self.plan, ControllerConfig())
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.collisions, 0)
        self.assertLess(result.time, 55.0)
        self.assertTrue(self.world.at_goal(result.poses[-1], 0.35, math.radians(12)))


if __name__ == "__main__":
    unittest.main()
